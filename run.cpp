/*
Inference for Llama-2 Transformer model in C/C++

Example compile: (see README for more details)
$ g++ -O3 -o run run.cpp

Then run with:
$ ./run
*/

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
    float* operator()(size_t x) { return base + x; }

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

// Helper to map an Array into a memory mapped file
template<size_t N, typename... Args>
void map_array(MMap& m, Array<N>& a, Args... args) {
    a.base = m.next(argmul(args...));
    a.mem_mapped = true; // prevent new memory allocation
    a.alloc(args...);
}

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
// Transformer and RunState structs, and related initializations

struct Config {
    int32_t dim;        // transformer dimension
    int32_t hidden_dim; // for ffn layers
    int32_t n_layers;   // number of layers
    int32_t n_heads;    // number of query heads
    int32_t n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int32_t vocab_size; // vocabulary size, usually 256 (byte-level)
    int32_t seq_len;    // max sequence length
};

struct TransformerWeights {
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
};

struct RunState {
    // Current wave of activations
    Array<1> x;   // activation at current time stamp (dim,)
    Array<2> xb;  // same, but inside a residual branch (dim,)
    Array<1> xb2; // an additional buffer just for convenience (dim,)
    Array<1> hb;  // buffer for hidden dimension in the ffn (hidden_dim,)
    Array<1> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    Array<2> q;   // query (dim,)
    Array<2> k;   // key (kv_dim,)
    Array<2> v;   // value (kv_dim,)
    Array<2> att; // buffer for scores/attention values (n_heads, seq_len)
    Array<1> logits; // output logits (vocab_size)
    // Key and Value cache
    Array<4> key_cache;   // (layer, seq_len, n_kv_heads, head_size)
    Array<4> value_cache; // (layer, seq_len, n_kv_heads, head_size)
    // freq_cis for RoPE relatively positional embeddings
    Array<2> freq_cis; // (seq_len, head_size);

    RunState(const Config&);
};

RunState::RunState(const Config& p) {

    int head_size = p.dim / p.n_heads;

    x.alloc(p.dim);
    xb.alloc(p.n_heads, head_size); // dim == n_heads * head_size
    xb2.alloc(p.dim);
    hb.alloc(p.hidden_dim);
    hb2.alloc(p.hidden_dim);
    q.alloc(p.n_heads, head_size);
    k.alloc(p.n_kv_heads, head_size);
    v.alloc(p.n_kv_heads, head_size);
    att.alloc(p.n_heads, p.seq_len);
    logits.alloc(p.vocab_size);
    key_cache.alloc(p.n_layers, p.seq_len, p.n_kv_heads, head_size);
    value_cache.alloc(p.n_layers, p.seq_len, p.n_kv_heads, head_size);
    freq_cis.alloc(p.seq_len, head_size);

    // Compute freq_cis
    float* ptr = freq_cis;
    float theta = 1e+08;
    for (int pos = 0; pos < p.seq_len; pos++) {
        for (int i = 0; i < head_size / 2; i++) {
            float freq = 1.0 / pow(theta, float(i) / head_size);
            freq *= pos;
            *ptr++ = cos(freq); // real part
            *ptr++ = sin(freq); // imaginary part
        }
    }
}

void init_from_mmap(MMap& m, MMap& t, Config* p, TransformerWeights* w, vector<string>* vocab) {

    // Read in config header, all Config fields are int32_t,
    // so define a union with an array to iterate over them
    static const size_t N = sizeof(Config) / sizeof(int32_t);
    union U { struct Config; int32_t vec[N]; };
    for (auto& v : reinterpret_cast<U*>(p)->vec)
        v = *m.next<int32_t>();

    // Negative vocab size is hacky way of signaling unshared weights. bit yikes.
    bool shared_weights = (p->vocab_size > 0);
    p->vocab_size = abs(p->vocab_size);

    // Memory map the Transformer weights into the data pointer
    int head_size = p->dim / p->n_heads;

    map_array(m, w->token_embedding_table, p->vocab_size, p->dim);
    map_array(m, w->rms_att_weight, p->n_layers, p->dim);
    map_array(m, w->wq, p->n_layers, p->dim, p->n_heads, head_size);
    map_array(m, w->wk, p->n_layers, p->dim, p->n_kv_heads, head_size);
    map_array(m, w->wv, p->n_layers, p->dim, p->n_kv_heads, head_size);
    map_array(m, w->wo, p->n_layers, p->n_heads, head_size, p->dim);
    map_array(m, w->rms_ffn_weight, p->n_layers, p->dim);
    map_array(m, w->w1, p->n_layers, p->hidden_dim, p->dim);
    map_array(m, w->w2, p->n_layers, p->dim, p->hidden_dim);
    map_array(m, w->w3, p->n_layers, p->hidden_dim, p->dim);
    map_array(m, w->rms_final_weight, p->dim);

 /* w->freq_cis_real */ m.next(p->seq_len * head_size / 2);
 /* w->freq_cis_imag */ m.next(p->seq_len * head_size / 2);

    w->wcls = shared_weights ? w->token_embedding_table : m.next(p->vocab_size * p->dim);

    // Read in the tokenizer .bin file
    t.next<float>(); // ignore max_token_length
    for (int i = 0; i < p->vocab_size; i++) {
        t.next<float>(); // ignore score
        int32_t len = *t.next<int32_t>();
        char* c = t.next<char>(len);
        vocab->push_back(string(c, len));
    }
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
// sampling can be done in a few ways: greedy argmax, sampling or top-p sampling

int argmax(float* prob, int n) {
    // return the index with the highest probability
    return std::distance(prob, max_element(prob, prob + n));
}

// top-p sampling (or "nucleus sampling") samples from the smallest set of
// tokens that exceed probability topp. This way we never sample tokens that
// have very low probabilities and are less likely to go "off the rails".
//
// if topp <= 0 or > 1 simply sample from the predicted probability distribution
int sample(float* prob, float topp, int n) {

    vector<int> v;
    v.reserve(n);
    float cumulative_prob = 1.0f;

    if (topp <= 0 || topp > 1)
        topp = 1.0f;

    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
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

// ----------------------------------------------------------------------------
// neural net blocks

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

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {

    // a few convenience variables
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table(token), dim * sizeof(float));

    // pluck out the "pos" row of freq_cis
    float* freq = s->freq_cis(pos);

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm (Root Mean Square normalization)
        rmsnorm(s->xb, x, w->rms_att_weight(l), dim);

        // qkv matmuls for this position (V X M = V)
        matmul(s->q, s->xb, w->wq(l), dim, dim);
        matmul(s->k, s->xb, w->wk(l), dim, kv_dim);
        matmul(s->v, s->xb, w->wv(l), dim, kv_dim);

        // RoPE relative positional encoding: complex-valued
        // rotate q and k by freq_cis in each head
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < dim; h += 2) {
            int k = (h % head_size);
            complexmul(s->q+h, s->q+h+1, freq[k], freq[k+1]);
        }
        for (h = 0; h < kv_dim; h += 2) {
            int k = (h % head_size);
            complexmul(s->k+h, s->k+h+1, freq[k], freq[k+1]);
        }

        // save key,value at this time step (pos) to our kv cache
        memcpy(s->key_cache(l, pos), s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache(l, pos), s->v, kv_dim * sizeof(float));

        // multihead attention. iterate over all heads
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q(h);
            // attention scores for this head
            float* att = s->att(h);
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache(l, t, h / kv_mul);
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
            float* xb = s->xb(h);
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache(l, t, h / kv_mul);
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo(l), dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight(l), dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1(l), dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3(l), dim, hidden_dim);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
        }

        // elementwise multiply with w3(x)
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * s->hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2(l), hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
}

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

int str_lookup(const string& str, const map<string, int>& sorted_vocab) {

    auto it = sorted_vocab.find(str);
    return it != sorted_vocab.end() ? it->second : -1;
}

void bpe_encode(vector<int>* tokens_ptr, const string& text, const vector<string>& vocab) {

    vector<int>& tokens = *tokens_ptr; // syntactic sugar
    map<string, int> sorted_vocab;
    string str;

    // sort vocabulary
    for (size_t i = 0; i < vocab.size(); i++)
        sorted_vocab[vocab[i]] = i;

    // first encode every individual character in the input (UTF-8) string
    for (size_t i = 0, start = 0; i < text.size(); start = i) {
        // in a UTF-8 character any byte but the first has format 10xxxxxx
        while ((text[++i] & 0xc0) == 0x80) {}
        str = text.substr(start, i - start); // extract a single UTF-8 char
        int id = str_lookup(str, sorted_vocab);
        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens.push_back(id);
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (unsigned char c : str) {
                tokens.push_back(c + 3);
            }
        }
    }

    // Greedy merge consecutive tokens until there are no more new merges
    while (true) {
        size_t i = 0;
        for (size_t next = i + 1; next < tokens.size(); next++) {
            // check if we can merge the pair (token[i], token[next])
            str = vocab[tokens[i]] + vocab[tokens[next]];
            int id = str_lookup(str, sorted_vocab);
            if (id == -1)
                tokens[++i] = tokens[next]; // can't merge further, move to next
            else
                tokens[i] = id; // merge next token
        }
        if (tokens.size() == i+1)
            break; // no new merges in the last iteration
        tokens.resize(i+1);
    }
}

// ----------------------------------------------------------------------------
// main loop, runs model inference

long run_model(int* steps, float temperature, float topp, const vector<int>& prompt_tokens,
               RunState& state, Config& config, TransformerWeights& weights, const vector<string>& vocab) {

    static const int BOS = 1; // BOS token

    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = BOS; // init with token BOS, as done in Llama-2 sentencepiece tokenizer
    int pos = 0;     // position in the sequence

    while (pos < *steps) {

        // forward the transformer to get logits for the next token
        transformer(token, pos, &config, &state, &weights);

        // advance the state machine
        if (pos < (int)prompt_tokens.size()) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
        } else {
            // sample the next token
            if (temperature == 0.0f) {
                // greedy argmax sampling: take the token with the highest probability
                next = argmax(state.logits, config.vocab_size);
            } else {
                // apply softmax with temperature to the logits to get the
                // probabilities for next token.
                softmax(state.logits, config.vocab_size, temperature);

                // sample from this distribution to get the next token
                // if topp > 0 we (also) perform top-p (nucleus) sampling, clamping the least likely
                // tokens to zero. Othewise sample from the predicted probability distribution.
                next = sample(state.logits, topp, config.vocab_size);
            }
        }

        pos++; // increment before a possible early exit due to a BOS token

        // data-dependent terminating condition: the BOS token delimits sequences
        if (next == BOS)
            break;

        // following BOS token, sentencepiece decoder strips any leading whitespace (see PR #89)
        string next_str = vocab[next];
        if (token == BOS && next_str[0] == ' ')
            next_str.erase(0, 1);

        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        unsigned char byte_val;
        if (sscanf(next_str.c_str(), "<0x%02hhX>", &byte_val) == 1) {
            // ok this token is a raw byte token, carefuly to only print printable chars or whitespace
            // some of the other bytes can be various control codes, backspace, etc. => skip
            if (isprint(byte_val) || isspace(byte_val))
                next_str = string(1, byte_val);
        }

        cout << next_str << std::flush;
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0)
            start = time_in_ms();
    }
    cout << endl;
    *steps = pos;
    return time_in_ms() - start; // elapsed time in ms
}

void print_model_info(const Config& config, vector<string>& vocab) {
    cerr << "\nModel parameters:\n"
         << "\n    Vocab size " << vocab.size()
         << "\n    Dimension " << config.dim
         << "\n    Hidden dim " << config.dim
         << "\n    Num heads " << config.n_heads
         << "\n    Num kv heads " << config.n_kv_heads
         << "\n    Head size (dim / num heads) " << config.dim / config.n_heads
         << "\n    Num layers " << config.n_layers
         << "\n    Max context (tokens) " << config.seq_len
         << "\n" << endl;
}

void error_usage() {
    cerr << "Usage:   run <checkpoint> [options]\n"
            "Example: run model.bin -n 256 -i \"Once upon a time\"\n"
            "Options:\n"
            "  -t <float>  temperature, default 1.0\n"
            "  -p <float>  p value in top-p (nucleus) sampling. default 0.9\n"
            "  -s <int>    random seed, default time(NULL)\n"
            "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n"
            "  -i <string> input prompt\n"
            "  -z <string> optional path to custom tokenizer\n" << endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {

    // default inits
    float temperature = 1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;        // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    RNG::seed = 0;            // seed rng with time by default
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    string prompt;            // prompt string
    string checkpoint;        // e.g. out/model.bin
    string tokenizer = "tokenizer.bin";

    // 'checkpoint' is necessary, optional arguments and their values
    // come in pairs, so argc must be even.
    if (argc % 2 != 0)
        error_usage();

    checkpoint = string(argv[1]);

    // Read in any optional argument
    if (argc > 2) {
        vector<string> opt(argv + 2, argv + argc);
        for (size_t i = 0; i < opt.size(); i += 2) {
            if      (opt[i] == "-t") temperature = stof(opt[i+1]);
            else if (opt[i] == "-p") topp = stof(opt[i+1]);
            else if (opt[i] == "-s") RNG::seed = stoi(opt[i+1]);
            else if (opt[i] == "-n") steps = stoi(opt[i+1]);
            else if (opt[i] == "-i") prompt = opt[i+1];
            else if (opt[i] == "-z") tokenizer = opt[i+1];
            else
                error_usage();
        }
    }

    if (RNG::seed == 0)
        RNG::seed = (unsigned int)time(NULL);

    // Read in model.bin
    Config config;
    TransformerWeights weights;
    vector<string> vocab;

    // Memory map the checkpoint file and init weights
    MMap mmap(checkpoint);
    MMap tkmap(tokenizer);
    init_from_mmap(mmap, tkmap, &config, &weights, &vocab);

    // Create and init the application RunState
    RunState state(config);

    print_model_info(config, vocab);

    // Process the prompt, if any
    vector<int> prompt_tokens;
    if (!prompt.empty())
        bpe_encode(&prompt_tokens, prompt, vocab);

    // Right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len)
        steps = config.seq_len;

    // Run the model for the given number of steps or until BOS token
    long elapsed = run_model(&steps, temperature, topp, prompt_tokens, state, config, weights, vocab);

    // report achieved tok/s (steps-1 because the timer starts after first iteration)
    if (steps > 1)
        cerr << "\nachieved tok/s: " << (steps-1) / (double)(elapsed)*1000 << endl;

    return 0;
}
