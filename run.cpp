/*
Inference for Llama-2 Transformer model in pure C.

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

#include <complex>
#include <iostream>
#include <vector>
#include <sys/stat.h>

class MMap {
public:
    MMap(const char* file) {
        struct stat fileInfo;
        int fd = open(file, O_RDONLY);
        if (fd == -1 || stat(file, &fileInfo) == -1) {
            printf("Couldn't open file %s\n", file);
            exit(1);
        }
        size = fileInfo.st_size;
        data = mmap(NULL, fileInfo.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        cur = static_cast<char*>(data);
        close(fd); // we can close the file once mapped
        if (data == MAP_FAILED) {
            printf("mmap failed!\n");
            exit(1);
        }
    }
    ~MMap() { munmap(data, size); }

    template<typename T>
    T* get(size_t len = 1) {
        cur += len * sizeof(T);
        return reinterpret_cast<T*>(cur - len * sizeof(T));
    }

    template<typename T>
    MMap& operator>> (T& rhs) {
        rhs = *this->get<T>();
        return *this;
    }

private:
    size_t size;
    void* data;
    char* cur;
};

using namespace std;

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

typedef struct {
    size_t dim; // transformer dimension
    size_t hidden_dim; // for ffn layers
    size_t n_layers; // number of layers
    size_t n_heads; // number of query heads
    size_t n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    size_t vocab_size; // vocabulary size, usually 256 (byte-level)
    size_t seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
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
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(p->dim, sizeof(float));
    s->xb2 = (float*)calloc(p->dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    s->q = (float*)calloc(p->dim, sizeof(float));
    s->k = (float*)calloc(p->dim, sizeof(float));
    s->v = (float*)calloc(p->dim, sizeof(float));
    s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    s->key_cache = (float*)calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

void init_weights(MMap& m, Config* p, TransformerWeights* w, bool shared_weights) {

    int head_size = p->dim / p->n_heads;
    w->token_embedding_table = m.get<float>(p->vocab_size * p->dim);
    w->rms_att_weight        = m.get<float>(p->n_layers * p->dim);
    w->wq                    = m.get<float>(p->n_layers * p->dim * p->dim);
    w->wk                    = m.get<float>(p->n_layers * p->dim * p->dim);
    w->wv                    = m.get<float>(p->n_layers * p->dim * p->dim);
    w->wo                    = m.get<float>(p->n_layers * p->dim * p->dim);
    w->rms_ffn_weight        = m.get<float>(p->n_layers * p->dim);
    w->w1                    = m.get<float>(p->n_layers * p->dim * p->hidden_dim);
    w->w2                    = m.get<float>(p->n_layers * p->dim * p->hidden_dim);
    w->w3                    = m.get<float>(p->n_layers * p->dim * p->hidden_dim);
    w->rms_final_weight      = m.get<float>(p->dim);
    w->freq_cis_real         = m.get<float>(p->seq_len * head_size / 2);
    w->freq_cis_imag         = m.get<float>(p->seq_len * head_size / 2);
    w->wcls = shared_weights ? w->token_embedding_table : m.get<float>(0);
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

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    float* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    float* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
        matmul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        int freq_cis_size = head_size / 2;
        for (int h = 0; h < p->n_heads; h++) {
            // get the q and k complex vectors for this head
            complex<float>* q_c = (complex<float>*)(s->q + h * head_size);
            complex<float>* k_c = (complex<float>*)(s->k + h * head_size);
            // rotate q and k by the freq_cis_real and freq_cis_imag
            for (int i = 0; i < freq_cis_size; i++) {
                complex<float> f(freq_cis_real_row[i], freq_cis_imag_row[i]);
                q_c[i] *= f;
                k_c[i] *= f;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        size_t loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * dim;
        float* value_cache_row = s->value_cache + loff + pos * dim;
        memcpy(key_cache_row, s->k, dim*sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, dim*sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        int h;
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
    // find the first perfect match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < vocab.size(); i++) {
        if (str == vocab[i])
            return i;
    }
    return -1;
}

void bpe_encode(char* text, const vector<string>& vocab, vector<int>& tokens) {

    // first encode every individual byte in the input string
    for (char *c = text; *c != '\0'; c++) {
        int id = str_lookup(string(c, 1), vocab);
        if (id == -1) { printf("not good\n"); exit(1); }
        tokens.push_back(id);
    }

    // merge consecutive tokens until there are no more new merges
    int merge_found = 1;
    while (merge_found) {
        merge_found = 0;
        int i = 0;
        for (int next = i+1; next < tokens.size(); next++) {
            // check if we can merge the pair (token[i], token[next])
            int id = str_lookup(vocab[tokens[i]] + vocab[tokens[next]], vocab);
            if (id != -1) {
                tokens[i] = id; // merge next token
                merge_found = 1;
            } else {
                // we cannot merge further, proceed with next token
                i++;
                tokens[i] = tokens[next];
            }
        }
        tokens.resize(i+1);
    }
}

// ----------------------------------------------------------------------------
// utilities

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

unsigned long long rng_seed;
unsigned int random_u32() {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    rng_seed ^= rng_seed >> 12;
    rng_seed ^= rng_seed << 25;
    rng_seed ^= rng_seed >> 27;
    return (rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32() { // random float32 in [0,1)
    return (random_u32() >> 8) / 16777216.0f;
}

int sample(float* probabilities, int n) {
    // sample index from probabilities, they must sum to 1
    float r = random_f32();
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int argmax(float* v, int n) {
    // return argmax of v in elements 0..n
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}
// ----------------------------------------------------------------------------

void init(MMap& m, Config* config, TransformerWeights* weights, vector<string>* vocab, bool* shared_weights) {

    int32_t x, len;
    float dummy;

    // Read in the config header, values are int32_t on disk
    size_t* ptr = reinterpret_cast<size_t*>(config);
    for (size_t i = 0; i < sizeof(Config)/sizeof(size_t); i++) {
        m >> x;
        *ptr++ = static_cast<size_t>(x);
    }
    // Negative vocab size is hacky way of signaling unshared weights. bit yikes.
    x = static_cast<int32_t>(config->vocab_size);
    *shared_weights = (x > 0);
    config->vocab_size = abs(x);

    // Memory map the Transformer weights into the data pointer
    init_weights(m, config, weights, shared_weights);

    // Read in the tokenizer.bin file
    MMap t("tokenizer.bin");
    t >> x; // ignore max_token_length

    for (size_t i = 0; i < config->vocab_size; i++) {
        t >> dummy >> len; // ignore scores
        char* c = t.get<char>(len);
        vocab->push_back(string(c, len));
    }
}

int main(int argc, char* argv[]) {

    // poor man's C argparse
    char* checkpoint;         // e.g. out/model.bin
    float temperature = 0.9f; // e.g. 1.0, or 0.0
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    char* prompt = NULL;      // prompt string

    // 'checkpoint' is necessary arg
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file> [temperature] [steps] [prompt]\n", argv[0]);
        return 1;
    }
    checkpoint = argv[1];

    if (argc >= 3) {
        // optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
        temperature = atof(argv[2]);
    }
    if (argc >= 4) {
        steps = atoi(argv[3]);
    }
    if (argc >= 5) {
        prompt = argv[4];
    }

    // seed rng with time. if you want deterministic behavior use temperature 0.0
    rng_seed = (unsigned int)time(NULL);

    // read in the model.bin file
    Config config;
    TransformerWeights weights;
    vector<string> vocab;
    bool shared_weights;

    // Memory map the checkpoint file, will be unmapped on obj d'tor
    MMap map(checkpoint);
    init(map, &config, &weights, &vocab, &shared_weights);

    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len)
        steps = config.seq_len;

    // process the prompt, if any
    vector<int> prompt_tokens;
    if (prompt)
        bpe_encode(prompt, vocab, prompt_tokens);

    // create and init the application RunState
    RunState state;
    malloc_run_state(&state, &config);

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    int pos = 0;     // position in the sequence
    printf("<s>\n"); // explicit print the initial BOS token for stylistic symmetry reasons

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
                for (int q=0; q<config.vocab_size; q++) { state.logits[q] /= temperature; }
                // apply softmax to the logits to get the probabilities for next token
                softmax(state.logits, config.vocab_size);
                // we sample from this distribution to get the next token
                next = sample(state.logits, config.vocab_size);
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

    // report achieved tok/s
    long end = time_in_ms();
    printf("\nachieved tok/s: %f\n", (steps-1) / (double)(end-start)*1000);

    // memory and file handles cleanup
    free_run_state(&state);

    return 0;
}
