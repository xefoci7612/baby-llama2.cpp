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

#include <algorithm> // std::sort
#include <iostream>
#include <vector>
#include <sys/stat.h>

using namespace std;

// ----------------------------------------------------------------------------
// Memory mapping facility to load model and tokenizer files

class MMap {

    size_t size;
    void* data;
    char* cur; // pointer arithmetic on void* is not standard

public:
    MMap(const char* file) {
        struct stat fileInfo;
        int fd = open(file, O_RDONLY);
        if (fd == -1 || stat(file, &fileInfo) == -1) {
            fprintf(stderr, "Couldn't open file %s\n", file);
            exit(EXIT_FAILURE);
        }
        data = mmap(NULL, fileInfo.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            fprintf(stderr, "mmap failed!\n");
            exit(EXIT_FAILURE);
        }
        cur = static_cast<char*>(data);
        size = fileInfo.st_size;
        close(fd); // we can close the file once mapped
    }
    ~MMap() { munmap(data, size); }

    template<typename T = float>
    T* get_ptr(int len) {
        T* ptr = reinterpret_cast<T*>(cur);
        cur += len * sizeof(T);
        if (cur > (char*)(data) + size) {
            fprintf(stderr, "Mapping after file's end!\n");
            exit(EXIT_FAILURE);
        }
        return ptr;
    }

    template<typename T>
    MMap& operator>> (T& rhs) {
        rhs = *this->get_ptr<T>(1);
        return *this;
    }
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
    float* token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim)
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
    // freq_cis_real (seq_len, head_size / 2) we don't use it
    // freq_cis_imag (seq_len, head_size / 2) we don't use it
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;

    void init(MMap&, Config* p, bool shared_weights);
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
    float* logits; // output logits (vocab_size)
    // Key and Value cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis; // (seq_len, head_size)

    // Poor's man memory management
    vector<void*> allocs;
    size_t allocated_size = 0;

    float* alloc(size_t n) {
        // We calloc instead of malloc to keep valgrind happy
        void* d = calloc(n, sizeof(float));
        if (!d) {
            fprintf(stderr, "Cannot allocate run state!\n");
            exit(EXIT_FAILURE);
        }
        allocated_size += n * sizeof(float);
        allocs.push_back(d);
        return static_cast<float*>(d);
    }

    RunState(const Config&);
   ~RunState() { for (void* d : allocs) free(d); }
};

RunState::RunState(const Config& p) {

    int head_size = p.dim / p.n_heads;

    x   = alloc(p.dim);
    xb  = alloc(p.dim);
    xb2 = alloc(p.dim);
    hb  = alloc(p.hidden_dim);
    hb2 = alloc(p.hidden_dim);
    q   = alloc(p.dim);
    k   = alloc(p.dim);
    v   = alloc(p.dim);
    att = alloc(p.n_heads * p.seq_len);
    logits      = alloc(p.vocab_size);
    key_cache   = alloc(size_t(p.n_layers) * p.seq_len * p.dim);
    value_cache = alloc(size_t(p.n_layers) * p.seq_len * p.dim);
    freq_cis    = alloc(p.seq_len * head_size);

    // Compute freq_cis table, don't load from model.bin
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

void TransformerWeights::init(MMap& m, Config* p, bool shared_weights) {

    int head_size = p->dim / p->n_heads;

    token_embedding_table = m.get_ptr(p->vocab_size * p->dim);
    rms_att_weight        = m.get_ptr(p->n_layers * p->dim);
    wq                    = m.get_ptr(p->n_layers * p->dim * p->dim);
    wk                    = m.get_ptr(p->n_layers * p->dim * p->dim);
    wv                    = m.get_ptr(p->n_layers * p->dim * p->dim);
    wo                    = m.get_ptr(p->n_layers * p->dim * p->dim);
    rms_ffn_weight        = m.get_ptr(p->n_layers * p->dim);
    w1                    = m.get_ptr(p->n_layers * p->dim * p->hidden_dim);
    w2                    = m.get_ptr(p->n_layers * p->dim * p->hidden_dim);
    w3                    = m.get_ptr(p->n_layers * p->dim * p->hidden_dim);
    rms_final_weight      = m.get_ptr(p->dim);
 /* freq_cis_real */        m.get_ptr(p->seq_len * head_size / 2);
 /* freq_cis_imag */        m.get_ptr(p->seq_len * head_size / 2);

    wcls = shared_weights ? token_embedding_table : m.get_ptr(p->vocab_size * p->dim);
}

void init_from_mmap(MMap& m, Config* config, TransformerWeights* weights, vector<string>* vocab) {

    int32_t x, len;
    float score;
    bool shared_weights;

    // Read in the config header, all Config fields are int32_t,
    // so define a union with an array to iterate over them
    static const int N = sizeof(Config) / sizeof(int32_t);
    union U { struct Config; int32_t vec[N]; };
    int32_t* it = reinterpret_cast<U*>(config)->vec;
    for (int i = 0; i < N; i++)
        m >> *it++;

    // Negative vocab size is hacky way of signaling unshared weights. bit yikes.
    shared_weights = (config->vocab_size > 0);
    config->vocab_size = abs(config->vocab_size);

    // Memory map the Transformer weights into the data pointer
    weights->init(m, config, shared_weights);

    // Read in tokenizer.bin
    MMap t("tokenizer.bin");
    t >> x; // ignore max_token_length

    for (int i = 0; i < config->vocab_size; i++) {
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
// sampling can be done in a few ways: greedy argmax, sampling, top-p or top-k sampling

int argmax(float* prob, int n) {
    // return the index that has the highest probability
    return std::distance(prob, max_element(prob, prob + n));
}

// top-p sampling (or "nucleus sampling") samples from the smallest set of
// tokens that exceed probability topp. This way we never sample tokens that
// have very low probabilities and are less likely to go "off the rails".
//
// top-k sampling samples from the K set of tokens with the highest probabities.
// If enabled, top-k will be performed before top-p.
//
// if topp and topk <= 0 simply sample from the predicted probability distribution
int sample(float* prob, int topk, float topp, int size) {

    static const float Threshold = 1e-5;

    auto greater_prob = [&prob](int i1, int i2) { return prob[i1] > prob[i2]; };
    vector<int> v;
    v.reserve(size);
    float cumulative_prob = 1.0f;

    // Skip probabilities below threshold to speed up sorting
    for (int i = 0; i < size; i++)
        if (prob[i] >= Threshold)
            v.push_back(i);

    if (topk > 0 && topk < (int)v.size()) {
        // move to front the topk indices with highest probability and resize
        nth_element(v.begin(), v.begin() + topk, v.end(), greater_prob);
        v.resize(topk);
        float sum = 0.0;
        // normalize probabilities so that sum is 1
        for (int i : v) { sum += prob[i]; }
        for (int i : v) { prob[i] /= sum; }
    }

    if (topp > 0) {
        // sort v in descending order of indexed probabilities
        sort(v.begin(), v.end(), greater_prob);

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

void accum(float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
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

void softmax(float* x, int size, float temperature = 1.0) {
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
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));

    // pluck out the "pos" row of freq_cis
    float* fr = s->freq_cis + pos * head_size;

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm (Root Mean Square normalization)
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // qkv matmuls for this position (V X M = V)
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * dim, dim, dim);
        matmul(s->v, s->xb, w->wv + l * dim * dim, dim, dim);

        // RoPE relative positional encoding: complex-valued
        // rotate q and k by freq_cis in each head.
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < dim; h += 2) {
            int k = h % head_size;
            complexmul(s->q+h, s->q+h+1, fr[k], fr[k+1]);
            complexmul(s->k+h, s->k+h+1, fr[k], fr[k+1]);
        }

        // save key,value at this time step (pos) to our kv cache
        size_t loff = size_t(l) * p->seq_len * dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * dim;
        float* value_cache_row = s->value_cache + loff + pos * dim;
        memcpy(key_cache_row, s->k, dim * sizeof(float));
        memcpy(value_cache_row, s->v, dim * sizeof(float));

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
                // scale down attention score before softmax
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
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
        }

        // elementwise multiply with w3(x)
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * s->hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

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

    auto it = find(vocab.begin(), vocab.end(), str);
    return it != vocab.end() ? it - vocab.begin() : -1;
}

void bpe_encode(vector<int>* tokens_ptr, const string& text, const vector<string>& vocab) {

    vector<int>& tokens = *tokens_ptr; // syntactic sugar
    string str;

    // First encode every individual character in the input string
    for (size_t i = 0, start = 0; i < text.size(); start = i) {
        // In UTF-8 character any byte but the first has format 10xxxxxx
        while ((text[++i] & 0xc0) == 0x80) {}
        str = text.substr(start, i - start);
        int id = str_lookup(str, vocab);
        if (id == -1) {
            cerr << "Character <" << str << "> not in vocab" << endl;
            exit(EXIT_FAILURE);
        }
        tokens.push_back(id);
    }

    // Merge consecutive tokens until there are no more new merges
    while (true) {
        size_t i = 0;
        for (size_t next = i+1; next < tokens.size(); next++) {
            // check if we can merge the pair (token[i], token[next])
            str = vocab[tokens[i]] + vocab[tokens[next]];
            int id = str_lookup(str, vocab);
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

long run_model(int* steps, float temperature, float topp, int topk, const vector<int>& prompt_tokens,
               RunState& state, Config& config, TransformerWeights& weights, const vector<string>& vocab) {

    long start = 0; // used to time our code, only initialized after first iteration
    int next;       // will store the next token in the sequence
    int token = 1;  // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    int pos = 0;    // position in the sequence

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
                // if topk > 0 we perform top-k sampling, sampling among the top k tokens,
                // if topp > 0 we (also) perform top-p (nucleus) sampling, clamping the least likely
                // tokens to zero. Othewise sample from the predicted probability distribution.
                next = sample(state.logits, topk, topp, config.vocab_size);
            }
        }

        pos++; // increment before a possible break due to BOS

        // data-dependent terminating condition: the BOS (1) token delimits sequences
        if (next == 1)
            break;

        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        string next_str = vocab[next];
        if (token == 1 && next_str[0] == ' ')
            next_str.erase(0, 1);

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

void print_model_info(const Config& config, const RunState& state, vector<string>& vocab) {
    cerr << "\nModel parameters:\n"
         << "\n    Vocab size " << vocab.size()
         << "\n    Dimension " << config.dim
         << "\n    Hidden dim " << config.dim
         << "\n    Num heads " << config.n_heads
         << "\n    Head size (dim / num heads) " << config.dim / config.n_heads
         << "\n    Num layers " << config.n_layers
         << "\n    Max context (tokens) " << config.seq_len
         << "\n    Allocated memory (MB) " << state.allocated_size / (1024 * 1024)
         << "\n" << endl;
}

void error_usage() {
    cerr << "Usage:   run <checkpoint> [options]\n"
            "Example: run model.bin -n 256 -i \"Once upon a time\"\n"
            "Options:\n"
            "  -t <float>  temperature, default 1.0\n"
            "  -p <float>  p value in top-p (nucleus) sampling. default 0.9, 0 = off\n"
            "  -k <int>    k value in top-k sampling. disabled by default, 0 = off\n"
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
    int topk = 0;             // top-k in Top-K sampling, disabled by default
    RNG::seed = 0;            // seed rng with time by default
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    string prompt;            // prompt string

    // 'checkpoint' is necessary, optional arguments and their values
    // come in pairs, so argc must be even.
    if (argc % 2 != 0)
        error_usage();

    checkpoint = argv[1];

    // Read in any optional argument
    if (argc > 2) {
        vector<string> opt(argv + 2, argv + argc);
        for (size_t i = 0; i < opt.size(); i += 2) {
            if      (opt[i] == "-t") temperature = stof(opt[i+1]);
            else if (opt[i] == "-p") topp = stof(opt[i+1]);
            else if (opt[i] == "-k") topk = stoi(opt[i+1]);
            else if (opt[i] == "-s") RNG::seed = stoi(opt[i+1]);
            else if (opt[i] == "-n") steps = stoi(opt[i+1]);
            else if (opt[i] == "-i") prompt = opt[i+1];
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
    init_from_mmap(mmap, &config, &weights, &vocab);

    // Create and init the application RunState
    RunState state(config);

    print_model_info(config, state, vocab);

    // Process the prompt, if any
    vector<int> prompt_tokens;
    if (!prompt.empty())
        bpe_encode(&prompt_tokens, prompt, vocab);

    // Right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len)
        steps = config.seq_len;

    // Run the model for the given number of steps or until BOS token
    long elapsed = run_model(&steps, temperature, topp, topk, prompt_tokens, state, config, weights, vocab);

    // report achieved tok/s (steps-1 because the timer starts after first iteration)
    if (steps > 1)
        fprintf(stderr, "\nachieved tok/s: %f\n", (steps-1) / (double)(elapsed)*1000);

    return 0;
}
