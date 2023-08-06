/*
Inference for Llama-2 Transformer model in C++

Example compile: (see README for more details)
$ g++ -O3 -o run run.cpp -lm

Then run with:
$ ./run
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#include <algorithm>
#include <complex>
#include <iostream>
#include <vector>
#include <sys/stat.h>

using namespace std;

// ----------------------------------------------------------------------------
// Memory mapping facility to load model file

class MMap {
public:
    MMap(const char* file) {
        struct stat fileInfo;
        int fd = open(file, O_RDONLY);
        if (fd == -1 || stat(file, &fileInfo) == -1) {
            printf("Couldn't open file %s\n", file);
            exit(EXIT_FAILURE);
        }
        size = fileInfo.st_size;
        data = mmap(NULL, fileInfo.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            printf("mmap failed!\n");
            exit(EXIT_FAILURE);
        }
        cur = static_cast<char*>(data);
        close(fd); // we can close the file once mapped
    }
    ~MMap() { munmap(data, size); }

    template<typename T>
    T* get_ptr(size_t len) {
        T* ptr = reinterpret_cast<T*>(cur);
        cur += len * sizeof(T);
        return ptr;
    }

    template<typename T>
    MMap& operator>> (T& rhs) {
        rhs = *this->get_ptr<T>(1);
        return *this;
    }

private:
    size_t size;
    void* data;
    char* cur; // pointer arithmetic on void* is not standard
};

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related initializations

struct Config {
    size_t dim; // transformer dimension
    size_t hidden_dim; // for ffn layers
    size_t n_layers; // number of layers
    size_t n_heads; // number of query heads
    size_t n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    size_t vocab_size; // vocabulary size, usually 256 (byte-level)
    size_t seq_len; // max sequence length
};

struct TransformerWeights {
    // token embedding table
    float* token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float* wq; // (layer, dim, dim)
    float* wk; // (layer, dim, dim)
    float* wv; // (layer, dim, dim)
    float* wo; // (layer, dim, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real; // (seq_len, head_size / 2)
    float* freq_cis_imag; // (seq_len, head_size / 2)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;

    void init(MMap&, const Config& p, bool shared_weights);
};

struct ProbIndex { // used when sorting probabilities during top-p sampling
    float prob;
    int index;
};

struct RunState {
    // Current wave of activations
    float* x;   // activation at current time stamp (dim,)
    float* xb;  // same, but inside a residual branch (dim,)
    float* xb2; // an additional buffer just for convenience (dim,)
    float* hb;  // buffer for hidden dimension in the ffn (hidden_dim,)
    float* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* q;   // query (dim,)
    float* k;   // key (dim,)
    float* v;   // value (dim,)
    float* att; // buffer for scores/attention values (n_heads, seq_len)
    float* logits; // output logits
    ProbIndex* probindex; // buffer used in top-p sampling
    // Key and Value cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
    // Helper complex vector for RoPE
    complex<float>* freq_cis; // (seq_len, head_size / 2)

    // Poor's man memory management
    void* mem_vec[20] = { NULL };
    int len = 0;

    template<typename T>
    T* alloc(size_t n) {
        // We calloc instead of malloc to keep valgrind happy
        mem_vec[len] = calloc(n, sizeof(T));
        if (!mem_vec[len]) {
            printf("Cannot allocate run state!\n");
            exit(EXIT_FAILURE);
        }
        return static_cast<T*>(mem_vec[len++]);
    }

    RunState(const Config&, const TransformerWeights&);
   ~RunState() { for (int i = 0; i < len; i++) free(mem_vec[i]); }
};

RunState::RunState(const Config& p, const TransformerWeights& w) {

    size_t head_size = p.dim / p.n_heads;

    x   = alloc<float>(p.dim);
    xb  = alloc<float>(p.dim);
    xb2 = alloc<float>(p.dim);
    hb  = alloc<float>(p.hidden_dim);
    hb2 = alloc<float>(p.hidden_dim);
    q   = alloc<float>(p.dim);
    k   = alloc<float>(p.dim);
    v   = alloc<float>(p.dim);
    att = alloc<float>(p.n_heads * p.seq_len);
    logits      = alloc<float>(p.vocab_size);
    probindex   = alloc<ProbIndex>(p.vocab_size);
    key_cache   = alloc<float>(p.n_layers * p.seq_len * p.dim);
    value_cache = alloc<float>(p.n_layers * p.seq_len * p.dim);

    freq_cis = alloc<complex<float>>(p.seq_len * head_size / 2);

    // Copy the 2 loaded RoPE vectors into a single complex vector
    for (size_t i = 0; i < p.seq_len * head_size / 2; i++)
        freq_cis[i] = complex<float>(w.freq_cis_real[i], w.freq_cis_imag[i]);
}

void TransformerWeights::init(MMap& m, const Config& p, bool shared_weights) {

    size_t head_size = p.dim / p.n_heads;

    token_embedding_table = m.get_ptr<float>(p.vocab_size * p.dim);
    rms_att_weight        = m.get_ptr<float>(p.n_layers * p.dim);
    wq                    = m.get_ptr<float>(p.n_layers * p.dim * p.dim);
    wk                    = m.get_ptr<float>(p.n_layers * p.dim * p.dim);
    wv                    = m.get_ptr<float>(p.n_layers * p.dim * p.dim);
    wo                    = m.get_ptr<float>(p.n_layers * p.dim * p.dim);
    rms_ffn_weight        = m.get_ptr<float>(p.n_layers * p.dim);
    w1                    = m.get_ptr<float>(p.n_layers * p.dim * p.hidden_dim);
    w2                    = m.get_ptr<float>(p.n_layers * p.dim * p.hidden_dim);
    w3                    = m.get_ptr<float>(p.n_layers * p.dim * p.hidden_dim);
    rms_final_weight      = m.get_ptr<float>(p.dim);
    freq_cis_real         = m.get_ptr<float>(p.seq_len * head_size / 2);
    freq_cis_imag         = m.get_ptr<float>(p.seq_len * head_size / 2);

    wcls = shared_weights ? token_embedding_table : m.get_ptr<float>(1);
}

void init_from_mmap(MMap& m, Config* config, TransformerWeights* weights, vector<string>* vocab) {

    int32_t x, len;
    float score;
    bool shared_weights;

    // Read in the config header, values are int32_t on disk
    // but we want them to be size_t.
    static const size_t N = sizeof(Config) / sizeof(size_t);

    // All Config fields are size_t, so define a union with
    // an array to iterate over them.
    union U { struct Config; size_t vec[N]; };
    size_t* ptr = reinterpret_cast<U*>(config)->vec;

    for (size_t i = 0; i < N; i++) {
        m >> x;
        *ptr++ = static_cast<size_t>(x);
    }

    // Negative vocab size is hacky way of signaling unshared weights. bit yikes.
    x = static_cast<int32_t>(config->vocab_size);
    shared_weights = (x > 0);
    config->vocab_size = abs(x);

    // Memory map the Transformer weights into the data pointer
    weights->init(m, *config, shared_weights);

    // Read in the tokenizer.bin file
    MMap t("tokenizer.bin");
    t >> x; // ignore max_token_length

    for (size_t i = 0; i < config->vocab_size; i++) {
        t >> score >> len; // ignore scores
        char* c = t.get_ptr<char>(len);
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

    unsigned long long seed; // Should be set before use, cannot be 0

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
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

int argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

// top-p sampling (or "nucleus sampling") samples from the smallest set of
// tokens that exceed probability topp. This way we never sample tokens that
// have very low probabilities and are less likely to go "off the rails".
//
// if topp <= 0 simply sample from the predicted probability distribution
int sample2(float* prob, float topp, const vector<int>& range_vec, float rnd) {

    vector v(range_vec); // init with 1..vocab_size range
    float cumulative_prob = 1.0f;

    if (topp > 0) {
        // sort v in descending order of indexed probabilities
        std::sort(v.begin(), v.end(), [&prob](int i1, int i2) { return prob[i1] > prob[i2]; });

        // truncate the list where cumulative probability exceeds topp
        cumulative_prob = 0.0f;
        for (size_t i = 0; i < v.size(); i++) {
            cumulative_prob += prob[v[i]];
            if (cumulative_prob > topp) {
                v.resize(i+1);
                break; // we've exceeded topp by including last_idx
            }
        }
    }

    // sample index from probabilities (they must sum to 1 * cumulative_prob!)
    //float r = RNG::random_f32() * cumulative_prob;
    float r = rnd * cumulative_prob; // Debug!!

    float cdf = 0.0f;
    for (size_t i = 0; i < v.size(); i++) {
        cdf += prob[v[i]];
        if (r < cdf) {
            return v[i];
        }
    }

    return v.back(); // in case of rounding errors
}

int sample(float* probabilities, int n, float rnd) {
    // sample index from probabilities (they must sum to 1!)
    float r = rnd; // RNG::random_f32();
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float rnd) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".

    // quicksort indices in descending order of probabilities
    for (int i = 0; i < n; i++) {
        probindex[i].index = i;
        probindex[i].prob = probabilities[i];
    }
    qsort(probindex, n, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = 0;
    for (int i = 0; i < n; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    //float r = RNG::random_f32() * cumulative_prob;
    float r = rnd * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// neural net blocks

void accum(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
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
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
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

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {

    // a few convenience variables
    float *x = s->x;
    size_t dim = p->dim;
    size_t hidden_dim =  p->hidden_dim;
    size_t head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = &(w->token_embedding_table[token * dim]);
    memcpy(x, content_row, dim*sizeof(*x));

    // pluck out the "pos" row of freq_cis
    complex<float>* freq_cis_row = s->freq_cis + pos * head_size / 2;

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
        matmul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the q and k complex vectors for this head
            complex<float>* q_c = (complex<float>*)(s->q + h * head_size);
            complex<float>* k_c = (complex<float>*)(s->k + h * head_size);
            // rotate q and k by the freq_cis_real and freq_cis_imag
            for (int i = 0; i < head_size / 2; i++) {
                q_c[i] *= freq_cis_row[i];
                k_c[i] *= freq_cis_row[i];
            }
        }

        // save key,value at this time step (pos) to our kv cache
        size_t loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * dim;
        float* value_cache_row = s->value_cache + loff + pos * dim;
        memcpy(key_cache_row, s->k, dim*sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, dim*sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * dim + h * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * dim + h * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
        }

        // elementwise multiply with w3(x)
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * s->hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        accum(x, s->xb, dim);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
}

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

int str_lookup(const string& str, const vector<string>& vocab) {
    for (int i = 0; i < vocab.size(); i++) {
        if (str == vocab[i])
            return i;
    }
    return -1;
}

void bpe_encode(vector<int>* tokens_ptr, const char* text, const vector<string>& vocab) {

    vector<int>& tokens = *tokens_ptr; // syntactic sugar

    // First encode every individual character in the input string
    while (*text) {
        // In UTF-8 character any byte but the first has format 10xxxxxx
        const char* start = text;
        do text++; while ((*text & 0xc0) == 0x80);
        int id = str_lookup(string(start, text - start), vocab);
        if (id == -1) {
            printf("First character in <%s> not in vocab\n", start);
            exit(EXIT_FAILURE);
        }
        tokens.push_back(id);
    }

    // Merge consecutive tokens until there are no more new merges
    while (true) {
        int i = 0;
        for (int next = i+1; next < tokens.size(); next++) {
            // check if we can merge the pair (token[i], token[next])
            int id = str_lookup(vocab[tokens[i]] + vocab[tokens[next]], vocab);
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

long run_model(int steps, float temperature, float topp, const vector<int>& prompt_tokens, RunState& state,
               Config& config, TransformerWeights& weights, const vector<string>& vocab) {

    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    int pos = 0;     // position in the sequence

    printf("<s>\n"); // explicit print the initial BOS token for stylistic symmetry reasons

    vector<int> range_vec(config.vocab_size);
    for (int i = 0; i < range_vec.size(); i++) { range_vec[i] = i; }

    while (pos < steps) {

        // forward the transformer to get logits for the next token
        transformer(token, pos, &config, &state, &weights);

        if (pos < prompt_tokens.size()) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
        } else {
            // sample the next token
            if (temperature == 0.0f) {
                // greedy argmax sampling: take the token with the highest probability
                next = argmax(state.logits, config.vocab_size);
            } else {
                // apply the temperature to the logits
                for (int q = 0; q < config.vocab_size; q++)
                    state.logits[q] /= temperature;

                // apply softmax to the logits to get the probabilities for next token
                softmax(state.logits, config.vocab_size);

                float rnd = RNG::random_f32();

                // we sample from this distribution to get the next token
                if (topp <= 0) {
                    // simply sample from the predicted probability distribution
                    next = sample(state.logits, config.vocab_size, rnd);

                    int next2 = sample2(state.logits, topp, range_vec, rnd);

                    if (next != next2) {
                        printf("\nSampling error 1 <%s> <%s>\n", vocab[next].c_str(), vocab[next2].c_str());
                        exit(EXIT_FAILURE);
                    }
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    next = sample_topp(state.logits, config.vocab_size, topp, state.probindex, rnd);

                    int next2 = sample2(state.logits, topp, range_vec, rnd);

                    if (next != next2) {
                        printf("\nSampling error 2 <%s> <%s>\n", vocab[next].c_str(), vocab[next2].c_str());
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
        // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
        string token_str = vocab[next];
        if (token == 1 && token_str[0] == ' ')
            token_str.erase(0, 1);

        cout << token_str << std::flush;

        // advance forward
        token = next;
        pos++;

        // init our timer here because the first iteration is slow due to memmap
        if (start == 0)
            start = time_in_ms();
    }

    return time_in_ms() - start; // elapsed time in ms
}

void error_usage() {
    cout << "Usage:   run <checkpoint> [options]\n"
            "Example: run model.bin -n 256 -i \"Once upon a time\"\n"
            "Options:\n"
            "  -t <float>  temperature, default 1.0\n"
            "  -p <float>  p value in top-p (nucleus) sampling. default 0.9, 0 = off\n"
            "  -s <int>    random seed, default time(NULL)\n"
            "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n"
            "  -i <string> input prompt\n" << endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {

    // default inits
    char* checkpoint;         // e.g. out/model.bin
    float temperature = 1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;        // top-p in nucleus sampling
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    string prompt;            // prompt string
    RNG::seed = (unsigned int)time(NULL); // seed rng with time by default, ensures != 0

    // 'checkpoint' is necessary, optional arguments and their values
    // come in pairs, so argc must be even.
    if (argc % 2 != 0)
        error_usage();

    checkpoint = argv[1];

    // Read in any optional argument
    if (argc > 2) {
        vector<string> opt(argv + 2, argv + argc);
        for (int i = 0; i < opt.size(); i += 2) {
            if      (opt[i] == "-t") temperature = stof(opt[i+1]);
            else if (opt[i] == "-p") topp = stof(opt[i+1]);
            else if (opt[i] == "-s") RNG::seed = stoi(opt[i+1]);
            else if (opt[i] == "-n") steps = stoi(opt[i+1]);
            else if (opt[i] == "-i") prompt = opt[i+1];
            else
                error_usage();
        }
    }

    // Read in the model.bin file
    Config config;
    TransformerWeights weights;
    vector<string> vocab;

    // Memory map the checkpoint file and init weights
    MMap mmap(checkpoint);
    init_from_mmap(mmap, &config, &weights, &vocab);

    // Create and init the application RunState
    RunState state(config, weights);

    // Process the prompt, if any
    vector<int> prompt_tokens;
    if (!prompt.empty())
        bpe_encode(&prompt_tokens, prompt.c_str(), vocab);

    // Right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len)
        steps = config.seq_len;

    // Run the model for the given number of steps
    long elapsed = run_model(steps, temperature, topp, prompt_tokens, state, config, weights, vocab);

    // Report achieved tok/s
    printf("\nachieved tok/s: %f\n", (steps-1) / (double)(elapsed)*1000);

    return 0;
}
