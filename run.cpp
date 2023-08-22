/* Inference for Llama-2 Transformer model in pure C/C++ */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <vector>
#include <sys/stat.h>

#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

using namespace std;

static const int BOS = 1; // BOS token

// ----------------------------------------------------------------------------
// Memory mapping facility to load model and tokenizer files

class MMap {

    size_t size;
    void* data;
    char* cur; // pointer arithmetic on void* is not standard

public:
    MMap(const string& file_str) {
        struct stat fileInfo;
        const char* file = file_str.c_str();
        int fd = open(file, O_RDONLY);
        if (fd == -1 || stat(file, &fileInfo) == -1) {
            cerr << "Couldn't open file " << file_str << endl;
            exit(EXIT_FAILURE);
        }
        data = mmap(NULL, fileInfo.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            cerr << "mmap failed!" << endl;
            exit(EXIT_FAILURE);
        }
        cur = static_cast<char*>(data);
        size = fileInfo.st_size;
        close(fd); // we can close the file once mapped
    }
   ~MMap() { munmap(data, size); }

    template<typename T = float>
    T* next(size_t n = 1) {
        T* ptr = reinterpret_cast<T*>(cur);
        cur += n * sizeof(T);
        if (cur > (char*)data + size) {
            cerr << "Mapping after end of file!" << endl;
            exit(EXIT_FAILURE);
        }
        return ptr;
    }
};

// ----------------------------------------------------------------------------
// A dynamic multi-dimensional array M(x,y) -> &M[x][y], M(x) -> M[x][...]

template <size_t N> struct Array;

template<>
struct Array<1> {

   ~Array() { if (!mem_mapped) { free(base); } }

    // implicit decay to pointer as a native C array
    operator float*() const { return base; }

    // return a pointer to the indexed item
    size_t addr(size_t x) { return x; } // used by derived classes
    float* operator()(size_t x) { return base + addr(x); }

    void alloc(size_t n) {
        if (!mem_mapped) {
            // we calloc instead of malloc to keep valgrind happy
            base = (float*)calloc(n, sizeof(float));
            if (!base) {
                cerr << "Cannot allocate run state!" << endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    float* base = NULL;
    bool mem_mapped = false;
};

// Helper to multiply args with a fold expressions
template<typename... Args>
size_t argmul(Args... args) { return (size_t(1) * ... * args); }

template <size_t N>
struct Array : Array<N-1> {

    template<typename... Args>
    void alloc(size_t a, size_t b, Args... args) { d = b * argmul(args...); Array<N-1>::alloc(a * b, args...); }

    size_t addr(size_t x) { return d * x; }

    template<typename... Args>
    size_t addr(size_t x, Args... args) { return d * x + Array<N-1>::addr(args...); }

    template<typename... Args>
    float* operator()(Args... args) { return this->base + addr(args...); }

    size_t d;
};

// ----------------------------------------------------------------------------
// Transformer model

struct Transformer {
    int32_t dim;        // transformer dimension
    int32_t hidden_dim; // for ffn layers
    int32_t n_layers;   // number of layers
    int32_t n_heads;    // number of query heads
    int32_t n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int32_t vocab_size; // vocabulary size, usually 256 (byte-level)
    int32_t seq_len;    // max sequence length

    // token embedding table
    Array<2> token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    Array<2> rms_att_weight; // (layer, dim)
    Array<2> rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    Array<4> wq; // (layer, dim, n_heads, head_size)
    Array<4> wk; // (layer, dim, n_kv_heads, head_size)
    Array<4> wv; // (layer, dim, n_kv_heads, head_size)
    Array<4> wo; // (layer, n_heads, head_size, dim)
    // weights for ffn
    Array<3> w1; // (layer, hidden_dim, dim)
    Array<3> w2; // (layer, dim, hidden_dim)
    Array<3> w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    Array<1> rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    // freq_cis_real (seq_len, head_size / 2) we don't use it
    // freq_cis_imag (seq_len, head_size / 2) we don't use it
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;

    // Current wave of activations
    Array<1> x;   // activation at current time stamp (dim,)
    Array<2> xb;  // same, but inside a residual branch (dim,)
    Array<1> xb2; // an additional buffer just for convenience (dim,)
    Array<1> hb;  // buffer for hidden dimension in the ffn (hidden_dim,)
    Array<1> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    Array<2> query; // query (dim,)
    Array<2> key;   // key (kv_dim,)
    Array<2> value; // value (kv_dim,)
    Array<2> attention; // buffer for scores/attention values (n_heads, seq_len)
    Array<1> logits; // output logits (vocab_size)
    // Key and Value cache
    Array<4> key_cache;   // (layer, seq_len, n_kv_heads, head_size)
    Array<4> value_cache; // (layer, seq_len, n_kv_heads, head_size)
    // freq_cis for RoPE relatively positional embeddings (not used anymore)
    Array<2> freq_cis; // (seq_len, head_size);

    MMap mmap;

    Transformer(const string& model_file);
    float* forward(int token, int pos);

    // Helper to map an Array into a memory mapped file
    template<size_t N, typename... Args>
    void map_array(Array<N>& a, Args... args) {
        a.base = mmap.next(argmul(args...));
        a.mem_mapped = true; // prevent new memory allocation
        a.alloc(args...);
    }
};

Transformer::Transformer(const string& model_file) : mmap(model_file) {

    // read in config header
    dim        = *mmap.next<int32_t>();
    hidden_dim = *mmap.next<int32_t>();
    n_layers   = *mmap.next<int32_t>();
    n_heads    = *mmap.next<int32_t>();
    n_kv_heads = *mmap.next<int32_t>();
    vocab_size = *mmap.next<int32_t>();
    seq_len    = *mmap.next<int32_t>();

    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    bool shared_weights = (vocab_size > 0);
    vocab_size = abs(vocab_size);

    // memory map the Transformer weights into the data pointer
    int head_size = dim / n_heads;

    map_array(token_embedding_table, vocab_size, dim);
    map_array(rms_att_weight, n_layers, dim);
    map_array(wq, n_layers, dim, n_heads, head_size);
    map_array(wk, n_layers, dim, n_kv_heads, head_size);
    map_array(wv, n_layers, dim, n_kv_heads, head_size);
    map_array(wo, n_layers, n_heads, head_size, dim);
    map_array(rms_ffn_weight, n_layers, dim);
    map_array(w1, n_layers, hidden_dim, dim);
    map_array(w2, n_layers, dim, hidden_dim);
    map_array(w3, n_layers, hidden_dim, dim);
    map_array(rms_final_weight, dim);

 /* freq_cis_real */ mmap.next(seq_len * head_size / 2);
 /* freq_cis_imag */ mmap.next(seq_len * head_size / 2);

    wcls = shared_weights ? token_embedding_table : mmap.next(vocab_size * dim);

    // allocate the run-state buffers
    x.alloc(dim);
    xb.alloc(n_heads, head_size); // dim == n_heads * head_size
    xb2.alloc(dim);
    hb.alloc(hidden_dim);
    hb2.alloc(hidden_dim);
    query.alloc(n_heads, head_size);
    key.alloc(n_kv_heads, head_size);
    value.alloc(n_kv_heads, head_size);
    attention.alloc(n_heads, seq_len);
    logits.alloc(vocab_size);
    key_cache.alloc(n_layers, seq_len, n_kv_heads, head_size);
    value_cache.alloc(n_layers, seq_len, n_kv_heads, head_size);
    freq_cis.alloc(seq_len, head_size);

    // compute freq_cis
    float* ptr = freq_cis;
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < head_size; i += 2) {
            float freq = 1.0f / powf(10000.0f, float(i) / head_size);
            freq *= pos;
            *ptr++ = cosf(freq); // real part
            *ptr++ = sinf(freq); // imaginary part
        }
    }
};

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

int argmax(float* prob, int n) {
    // return the index with the highest value
    return std::distance(prob, max_element(prob, prob + n));
}

void rmsnorm(float* o, float* x, float* w, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = w[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size, float temperature = 1.0f) {
    // find max value (for numerical stability)
    int id = argmax(x, size);
    float max_val = x[id];

    // exp, sum and apply temperature
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf((x[i] - max_val) / temperature);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void complexmul(float* a, float* b, float c, float d) {
    // (a+ib)(c+id) -> (ac-bd) + i(ad+bd)
    float a0 = *a;
    *a = *a * c - *b * d;
    *b = a0 * d + *b * c;
}

float* Transformer::forward(int token, int pos) {

    // a few convenience variables
    int kv_dim = (dim * n_kv_heads) / n_heads;
    int kv_mul = n_heads / n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int head_size = dim / n_heads;

    // copy the token embedding into x
    memcpy(x, token_embedding_table(token), dim * sizeof(float));

    // pluck out the "pos" row of freq_cis
    float* freq = freq_cis(pos);

    // forward all the layers
    for (int l = 0; l < n_layers; l++) {

        // attention rmsnorm (Root Mean Square normalization)
        rmsnorm(xb, x, rms_att_weight(l), dim);

        // qkv matmuls for this position (V X M = V)
        matmul(query, xb, wq(l), dim, dim);
        matmul(key, xb, wk(l), dim, kv_dim);
        matmul(value, xb, wv(l), dim, kv_dim);

        // RoPE relative positional encoding: complex-valued
        // rotate q and k by freq_cis in each head
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < dim; h += 2) {
            int k = (h % head_size);
            complexmul(query+h, query+h+1, freq[k], freq[k+1]);
            if (h < kv_dim) {
                complexmul(key+h, key+h+1, freq[k], freq[k+1]);
            }
        }

        // save key,value at this time step (pos) to our kv cache
        memcpy(key_cache(l, pos), key, kv_dim * sizeof(float));
        memcpy(value_cache(l, pos), value, kv_dim * sizeof(float));

        // multihead attention. iterate over all heads
        #pragma omp parallel for private(h)
        for (h = 0; h < n_heads; h++) {
            // get the query vector for this head
            float* q = query(h);
            // attention scores for this head
            float* att = attention(h);
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = key_cache(l, t, h / kv_mul);
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                // scale down attention score before softmax
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb_h = xb(h);
            memset(xb_h, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = value_cache(l, t, h / kv_mul);
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb_h
                for (int i = 0; i < head_size; i++) {
                    xb_h[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(xb2, xb, wo(l), dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(xb, x, rms_ffn_weight(l), dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(hb, xb, w1(l), dim, hidden_dim);
        matmul(hb2, xb, w3(l), dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= hb2[i];
            hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(xb, hb, w2(l), hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, rms_final_weight, dim);

    // classifier into logits
    matmul(logits, x, wcls, dim, vocab_size);

    return logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

struct Tokenizer {

    Tokenizer(const string& tokenizer_file, int vocab_size);
    int str_lookup(const string& str);
    void encode(vector<int>* tokens_ptr, const string& text);
    string decode(int prev_token, int token);

    MMap mmap;

    vector<string> vocab;
    vector<float> vocab_scores;
    map<string, int> sorted_vocab;
};

Tokenizer::Tokenizer(const string& tokenizer_file, int vocab_size) : mmap(tokenizer_file) {

    // Read in the tokenizer .bin file
    mmap.next<float>(); // ignore max_token_length
    for (int i = 0; i < vocab_size; i++) {
        float score = *mmap.next<float>();
        int32_t len = *mmap.next<int32_t>();
        char* c = mmap.next<char>(len);
        vocab.push_back(string(c, len));
        vocab_scores.push_back(score);
    }

    // sort vocabulary
    for (size_t i = 0; i < vocab.size(); i++)
        sorted_vocab[vocab[i]] = i;
}

int Tokenizer::str_lookup(const string& str) {

    auto it = sorted_vocab.find(str);
    return it != sorted_vocab.end() ? it->second : -1;
}

void Tokenizer::encode(vector<int>* tokens_ptr, const string& text) {

    vector<int>& tokens = *tokens_ptr; // syntactic sugar
    string str;

    // add_dummy_prefix is true by default
    tokens.push_back(str_lookup(" "));

    // first encode every individual character in the input (UTF-8) string
    for (size_t i = 0; i < text.size(); ) {

        // find the [start, end) of the current UTF-8 character
        size_t start = i;

        // in a UTF-8 character any byte but the first has format 10xxxxxx
        do {
            i++;
        } while (i < text.size() && (text[i] & 0xC0) == 0x80);

        // extract the UTF-8 character and look it up in the vocabulary
        str = text.substr(start, i - start);
        int id = str_lookup(str);
        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens.push_back(id);
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (unsigned char c : str)
                tokens.push_back(c + 3);
        }
    }

    vector<float> sv(tokens.size(), -1e10);
    int start = 0;
    int end = sv.size() - 1;

    while (true) {

        // find all possible merges between two consecutive tokens in [start, end)
        for (int i = start; i < end; i++) {
            str = vocab[tokens[i]] + vocab[tokens[i+1]];
            int id = str_lookup(str);
            sv[i] = (id != -1 ? vocab_scores[id] : -1e10);
        }
        // pick the best one with the highest score
        int best_idx = argmax(sv.data(), sv.size());

        if (sv[best_idx] <= -1e10)
            break; // we couldn't find any more pairs to merge, so we're done

        // merge the consecutive pair (best_idx, best_idx+1) into a new token
        str = vocab[tokens[best_idx]] + vocab[tokens[best_idx+1]];
        tokens[best_idx] = str_lookup(str);

        // delete token at position best_idx+1, shift the entire sequence back 1
        tokens.erase(tokens.begin() + best_idx + 1);
        sv.erase(sv.begin() + best_idx + 1);
        sv[best_idx] = -1e10; // reset stale score

        // update scores at previous and current position
        start = std::max(best_idx - 1, 0);
        end = std::min(best_idx + 1, static_cast<int>(sv.size()) - 1);
    }
}

string Tokenizer::decode(int prev_token, int token) {

    // following BOS token, sentencepiece decoder strips any leading whitespace (see PR #89)
    string token_str = vocab[token];
    if (prev_token == BOS && token_str[0] == ' ')
        token_str.erase(0, 1);

    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    unsigned char byte_val;
    if (sscanf(token_str.c_str(), "<0x%02hhX>", &byte_val) == 1) {
        // ok this token is a raw byte token, carefuly to only print printable chars or whitespace
        // some of the other bytes can be various control codes, backspace, etc. => skip
        token_str = isprint(byte_val) || isspace(byte_val) ? string(1, byte_val) : "";
    }
    return token_str;
}

// ----------------------------------------------------------------------------
// utilities: time / rng

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

namespace RNG {

    unsigned long long seed; // should be set before use, cannot be 0

    unsigned int random_u32() {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return (seed * 0x2545F4914F6CDD1Dull) >> 32;
    }

    float random_f32() { // random float32 in [0,1)
        return (random_u32() >> 8) / 16777216.0f;
    }
};

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

struct Sampler {

    Sampler(int n, float temp, float tp) {
        vocab_size = n;
        temperature = temp;
        topp = tp;
    }

    int sample_topp(float* prob);
    int sample(float* logits);

    int vocab_size;
    float temperature;
    float topp;
};

// top-p sampling (or "nucleus sampling") samples from the smallest set of
// tokens that exceed probability topp. This way we never sample tokens that
// have very low probabilities and are less likely to go "off the rails".
//
// if topp <= 0 or > 1 simply sample from the predicted probability distribution
int Sampler::sample_topp(float* prob) {

    vector<int> v;
    v.reserve(vocab_size);
    float cumulative_prob = 1.0f;

    if (topp <= 0 || topp > 1)
        topp = 1.0f;

    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (vocab_size - 1);
    for (int i = 0; i < vocab_size; i++) {
        if (prob[i] >= cutoff)
            v.push_back(i);
    }

    if (topp < 1) {
        // sort in descending order of indexed probabilities
        sort(v.begin(), v.end(), [prob](int a, int b) { return prob[a] > prob[b]; });

        // truncate the list where cumulative probability exceeds topp
        cumulative_prob = 0.0f;
        for (size_t i = 0; i < v.size(); i++) {
            cumulative_prob += prob[v[i]];
            if (cumulative_prob > topp) {
                v.resize(i+1);
                break; // we've exceeded topp by including this last item
            }
        }
    }

    // sample index from probabilities (they must sum to 1!)
    float r = RNG::random_f32() * cumulative_prob;

    float cdf = 0.0f;
    for (size_t i = 0; i < v.size(); i++) {
        cdf += prob[v[i]];
        if (r < cdf)
            return v[i];
    }

    return v.back(); // in case of rounding errors
}

int Sampler::sample(float* logits) {

    // sample the token given the logits and some hyperparameters
    int next;
    if (temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = argmax(logits, vocab_size);
    } else {
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, vocab_size, temperature);

        // sample from this distribution to get the next token
        // if topp > 0 we (also) perform top-p (nucleus) sampling, clamping the least likely
        // tokens to zero. Othewise sample from the predicted probability distribution.
        next = sample_topp(logits);
    }
    return next;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer& t, Tokenizer& tok, Sampler& sampler, const string& prompt, int steps) {

    // encode the (string) prompt into tokens sequence, if any is given
    vector<int> prompt_tokens;
    if (!prompt.empty())
        tok.encode(&prompt_tokens, prompt);

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = BOS; // init with token BOS, as done in Llama-2 sentencepiece tokenizer
    int pos = 0;     // position in the sequence

    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = t.forward(token, pos);

        // advance the state machine
        if (pos < (int)prompt_tokens.size()) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
        } else {
            // otherwise sample the next token from the logits
            next = sampler.sample(logits);
        }

        pos++; // increment before a possible early exit due to a BOS token

        // data-dependent terminating condition: the BOS token delimits sequences
        if (next == BOS)
            break;

        // print the token as string, decode it with the Tokenizer object
        string next_str = tok.decode(token, next);
        cout << next_str << std::flush;
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0)
            start = time_in_ms();
    }
    cout << endl;

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long elapsed = time_in_ms() - start;
        cerr << "\nachieved tok/s: " << (pos-1) / (double)(elapsed)*1000 << endl;
    }
}

// ----------------------------------------------------------------------------
// int main

void print_model_info(const Transformer& t) {
    cerr << "\nModel parameters:\n"
         << "\n    Vocab size " << t.vocab_size
         << "\n    Dimension " << t.dim
         << "\n    Hidden dim " << t.hidden_dim
         << "\n    Num heads " << t.n_heads
         << "\n    Num kv heads " << t.n_kv_heads
         << "\n    Head size (dim / num heads) " << t.dim / t.n_heads
         << "\n    Num layers " << t.n_layers
         << "\n    Max context (tokens) " << t.seq_len
         << "\n" << endl;
}

void error_usage() {
    cerr << "Usage:   run <checkpoint> [options]\n"
            "Example: run model.bin -n 256 -i \"Once upon a time\"\n"
            "Options:\n"
            "  -t <float>  temperature in [0,inf], default 1.0\n"
            "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n"
            "  -s <int>    random seed, default time(NULL)\n"
            "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n"
            "  -c <bool>   colour the probability of the next token, default 0 (false)\n"
            "  -i <string> input prompt\n"
            "  -z <string> optional path to custom tokenizer\n" << endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {

    // default parameters
    string checkpoint_path;  // e.g. out/model.bin
    string tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;        // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;          // number of steps to run for
    string prompt;            // prompt string
    RNG::seed = 0;            // seed rng with time by default

    // 'checkpoint' is necessary, optional arguments and their values
    // come in pairs, so argc must be even.
    if (argc % 2 != 0)
        error_usage();

    checkpoint_path = string(argv[1]);

    // Read in any optional argument
    if (argc > 2) {
        vector<string> opt(argv + 2, argv + argc);
        for (size_t i = 0; i < opt.size(); i += 2) {
            if      (opt[i] == "-t") temperature = stof(opt[i+1]);
            else if (opt[i] == "-p") topp = stof(opt[i+1]);
            else if (opt[i] == "-s") RNG::seed = stoi(opt[i+1]);
            else if (opt[i] == "-n") steps = stoi(opt[i+1]);
            else if (opt[i] == "-i") prompt = opt[i+1];
            else if (opt[i] == "-z") tokenizer_path = opt[i+1];
            else
                error_usage();
        }
    }

    // parameter validation/overrides
    if (RNG::seed <= 0) RNG::seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer(checkpoint_path);
    if (steps == 0) steps = transformer.seq_len; // ovrerride to ~max length

    print_model_info(transformer);

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer(tokenizer_path, transformer.vocab_size);

    // build the Sampler
    Sampler sampler(transformer.vocab_size, temperature, topp);

    // run!
    generate(transformer, tokenizer, sampler, prompt, steps);

    return 0;
}
