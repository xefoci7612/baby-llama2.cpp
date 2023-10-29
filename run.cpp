/* Inference for Llama-2 Transformer model in C/C++ */

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
#include <string>
#include <vector>
#include <sys/stat.h>

#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

using namespace std;

static const int BOS = 1; // sentencepiece BOS token
static const int EOS = 2; // sentencepiece EOS token

// ----------------------------------------------------------------------------
// Base struct for int8_t grouped quantization

static const int GS = 64; // group number

struct QuantizedTensor {
    int8_t* q; // quantized values
    float* s;  // scaling factors, they are GS times less than values
};

// pointer arithmetic on QuantizedTensor
QuantizedTensor operator+(const QuantizedTensor& qt, size_t n) {
    return QuantizedTensor{ qt.q + n, qt.s + (n / GS) };
}

// ----------------------------------------------------------------------------
// Format user prompt according to the model's chat schema
//
// ./run out/tl-chatq.bin -z tok_tl-chat.bin -s 1 -p 0.7 -n 250 -t 0.9 -i "Explain huggingface" -m chat

class Chat {
    // chat schemas for LLama 2 and Tinyllama
    const string templates[2][2] = {{ "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]",
                                      "%s[INST] %s [/INST]" },
                                    { "<|im_start|>system\n%s\nuser\n%s<|im_end|>\n<|im_start|>assistant\n",
                                      "%s<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" }};
    bool enabled;
    int schema;
public:
    Chat(bool b, int s) : enabled(b), schema(s) {}
    operator bool() const { return enabled; }
    string format(const string& user, const string& system) const {
        string str = templates[schema][system.empty()];
        return str.replace(str.find("%s"), 2, system)
                  .replace(str.find("%s"), 2, user);
    }
};

// ----------------------------------------------------------------------------
// Memory mapping facility to load and read model and tokenizer files

class MMap {

    size_t size;
    void* data;
    char* cur; // pointer arithmetic on void* is not standard
    char* eof;

public:
    MMap(const string& file_path) {
        struct stat fileInfo;
        const char* file = file_path.c_str();
        int fd = open(file, O_RDONLY);
        if (fd == -1 || stat(file, &fileInfo) == -1) {
            cerr << "Couldn't open file " << file_path << endl;
            exit(EXIT_FAILURE);
        }
        data = mmap(NULL, fileInfo.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            cerr << "mmap failed!" << endl;
            exit(EXIT_FAILURE);
        }
        cur = static_cast<char*>(data);
        size = fileInfo.st_size;
        eof = cur + size;
        close(fd); // we can close the file after mapping
    }
   ~MMap() { munmap(data, size); }

    // return a pointer to current position and update read pointer
    template<typename T>
    T* next(size_t n = 1) {
        T* ptr = reinterpret_cast<T*>(cur);
        cur += n * sizeof(T);
        if (cur > eof) {
            cerr << "Mapping after end of file!" << endl;
            exit(EXIT_FAILURE);
        }
        return ptr;
    }
};

// ----------------------------------------------------------------------------
// Memory management to alloc/release stuff

struct Memory {
    ~Memory() { for(void* p : m) free(p); }

    template<typename T>
    T* alloc(size_t n) {
        // we calloc instead of malloc to keep valgrind happy
        void* p = calloc(n, sizeof(T));
        if (!p) {
            cerr << "Cannot allocate memory!" << endl;
            exit(EXIT_FAILURE);
        }
        m.push_back(p);
        return static_cast<T*>(p);
    }

    vector<void*> m;
};

// ----------------------------------------------------------------------------
// A simple dynamic multi-dimensional array M(x,y) -> &M[x][y], M(x) -> M[x][...]
// used for both floats and quantized tensors

template <typename T> struct TArray;

// Array is for standard n-dimensional array of float weights
typedef TArray<float*> Array;

// QArray is for quantized tensor
typedef TArray<QuantizedTensor> QArray;

// QMatrix is for matrix of quantized tensors, grouped per layer
typedef TArray<QArray*> QMatrix;

// Little helper to multiply args with a fold expressions
template<typename... Args>
size_t argmul(Args... args) { return (size_t(1) * ... * args); }

template <typename T>
struct TArray {

    // implicit decay to base pointer (float*) as a native C array
    operator T() const { return base; }

    // compute address given indices
    template<size_t I> size_t addr() const { return 0; }

    template<size_t I, typename... Args>
    size_t addr(size_t x, Args... args) const {
        return x * d[I] + addr<I+1>(args...);
    }

    // return a pointer to the indexed item
    template<typename... Args>
    T operator()(Args... args) const {
        return base + addr<1>(args...);
    }

    // set array dimensions and shape
    template<size_t I> void set_shape() { d[I] = 1; }

    template<size_t I, typename... Args>
    void set_shape(size_t a, Args... args) {
        d[I] = a * argmul(args...);
        set_shape<I+1>(args...);
    }

    size_t size() const { return d[0]; }

    T base = T();
    size_t d[8] = {};
};

// ----------------------------------------------------------------------------
// Quantization functions

void quantize(QArray& qarray, float* x) {
    QuantizedTensor qx = qarray;
    int n = qarray.size();
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    int group;
    #pragma omp parallel for private(group)
    for (group = 0; group < num_groups; group++) {

        float* g_start = x + group * GS;
        float* g_end = g_start + GS;

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (float* p = g_start; p < g_end; p++) {
            float val = fabs(*p);
            if (val > wmax)
                wmax = val;
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx.s[group] = scale;

        // calculate and write the quantized values
        int8_t* q = qx.q + group * GS;
        for (float* p = g_start; p < g_end; p++)
            *q++ = (int8_t) round(*p / scale); // scale, round and clamp
    }
}

void dequantize(QArray& qarray, float* x) {
    QuantizedTensor qx = qarray;
    int n = qarray.size();
    for (int i = 0; i < n; i++)
        x[i] = qx.q[i] * qx.s[i / GS];
}

// ----------------------------------------------------------------------------
// Transformer model

struct Transformer {
    int dim;         // transformer dimension
    int hidden_dim;  // for ffn layers
    int n_layers;    // number of layers
    int n_heads;     // number of query heads
    int n_kv_heads;  // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;  // vocabulary size, usually 256 (byte-level)
    int seq_len;     // max sequence length
    bool shared_cls; // if true output classifier uses token_embedding_table
    int group_size;  // must be == GS

    // token embedding table
    QArray q_token_embedding_table; // quantized table on disk (vocab_size, dim)
    Array token_embedding_table;    // dequantized version (vocab_size, dim)

    // weights for rmsnorms
    Array rms_att_weight;   // (layer, dim)
    Array rms_ffn_weight;   // (layer, dim)
    Array rms_final_weight; // (dim,)

    // weights for matmuls. note dim == n_heads * head_size
    QMatrix q_wq; // (layer, dim, n_heads, head_size)
    QMatrix q_wk; // (layer, dim, n_kv_heads, head_size)
    QMatrix q_wv; // (layer, dim, n_kv_heads, head_size)
    QMatrix q_wo; // (layer, n_heads, head_size, dim)

    // weights for ffn
    QMatrix q_w1; // (layer, hidden_dim, dim)
    QMatrix q_w2; // (layer, dim, hidden_dim)
    QMatrix q_w3; // (layer, hidden_dim, dim)

    // (optional) classifier weights for the logits, on the last layer
    QArray q_wcls, *q_wcls_ptr;

    // current wave of activations
    Array x;         // activation at current time stamp (dim,)
    Array xb;        // same, but inside a residual branch (dim,)
    Array hb;        // buffer for hidden dimension in the ffn (hidden_dim,)
    Array hb2;       // buffer for hidden dimension in the ffn (hidden_dim,)
    QArray q_x;      // quantized x (dim,)
    QArray q_hb;     // quantized hb (hidden_dim,)
    Array query;     // query (n_heads, head_size)
    Array attention; // buffer for scores/attention values (n_heads, seq_len)
    Array logits;    // output logits (vocab_size)

    // key and value cache
    Array key_cache;   // (layer, seq_len, n_kv_heads, head_size)
    Array value_cache; // (layer, seq_len, n_kv_heads, head_size)
    Array freq_cis;    // (seq_len, head_size);

    MMap mmap;  // memory map object of the checkpoint file
    Memory mem; // memory allocation object

    enum { Alloc = true, Map = false };

    // allocate or map
    template<bool A, typename T>
    T* set(size_t n) { return A ? mem.alloc<T>(n) : mmap.next<T>(n); }

    // allocate or map an Array
    template<bool A, typename... Args>
    void set(Array& a, Args... args) {
        a.base = set<A, float>(argmul(args...));
        a.set_shape<0>(args...);
    }

    // allocate or map a QArray
    template<bool A, typename... Args>
    void set(QArray& a, Args... args) {
        // scaling factors are GS time less than values
        a.base.q = set<A, int8_t>(argmul(args...));
        a.base.s = set<A, float>(argmul(args...) / GS);
        a.set_shape<0>(args...);
    }

    // in case of a QMatrix, QuantizedTensors are serialized
    // per layer, so we allocate a QArray of QArray
    template<bool A, typename... Args>
    void set(QMatrix& m, size_t n_layers, Args... args) {
        m.base = set<Alloc, QArray>(n_layers);
        m.set_shape<0>(n_layers);
        for (size_t i = 0; i < n_layers; i++)
            set<A>(*m(i), args...);
    }

    Transformer(const string& model_file);
    float* forward(int token, int pos);
};

Transformer::Transformer(const string& model_file) : mmap(model_file) {
    // read a quantizied .bin file (version 2)

    // read header
    size_t header_size = 256;
    char* weights_ptr = mmap.next<char>(0) + header_size;

    uint32_t magic  = *mmap.next<uint32_t>();
    int32_t version = *mmap.next<int32_t>();

    if (magic != 0x616b3432 || version != 2) {
        cerr << "File format not supported!" << endl;
        exit(EXIT_FAILURE);
    }

    // read in config parameters
    dim        = *mmap.next<int32_t>();
    hidden_dim = *mmap.next<int32_t>();
    n_layers   = *mmap.next<int32_t>();
    n_heads    = *mmap.next<int32_t>();
    n_kv_heads = *mmap.next<int32_t>();
    vocab_size = *mmap.next<int32_t>();
    seq_len    = *mmap.next<int32_t>();
    shared_cls = *mmap.next<uint8_t>();
    group_size = *mmap.next<int32_t>();

    if (group_size != GS) {
        cerr << "Group size is " << group_size << ", expected " << GS << endl;
        exit(EXIT_FAILURE);
    }

    // skip padding at header tail
    size_t padding_size = weights_ptr - mmap.next<char>(0);
    mmap.next<char>(padding_size);

    // memory map the Transformer weights into the data pointer
    int head_size = dim / n_heads;

    // first are the parameters that are kept in fp32
    set<Map>(rms_att_weight, n_layers, dim);
    set<Map>(rms_ffn_weight, n_layers, dim);
    set<Map>(rms_final_weight, dim);

    // now read all the quantized weights
    set<Map>(q_token_embedding_table, vocab_size, dim);
    set<Map>(q_wq, n_layers, dim, n_heads, head_size);
    set<Map>(q_wk, n_layers, dim, n_kv_heads, head_size);
    set<Map>(q_wv, n_layers, dim, n_kv_heads, head_size);
    set<Map>(q_wo, n_layers, n_heads, head_size, dim);
    set<Map>(q_w1, n_layers, hidden_dim, dim);
    set<Map>(q_w2, n_layers, dim, hidden_dim);
    set<Map>(q_w3, n_layers, hidden_dim, dim);

    // map also the classifier if not shared with token embeddings
    q_wcls_ptr = shared_cls ? &q_token_embedding_table
                            : (set<Map>(q_wcls, vocab_size, dim), &q_wcls);

    // allocate the run-state buffers
    set<Alloc>(x, dim);
    set<Alloc>(xb, n_heads, head_size); // dim == n_heads * head_size
    set<Alloc>(hb, hidden_dim);
    set<Alloc>(hb2, hidden_dim);
    set<Alloc>(q_x, dim);
    set<Alloc>(q_hb, hidden_dim);
    set<Alloc>(query, n_heads, head_size);
    set<Alloc>(attention, n_heads, seq_len);
    set<Alloc>(logits, vocab_size);
    set<Alloc>(key_cache, n_layers, seq_len, n_kv_heads, head_size);
    set<Alloc>(value_cache, n_layers, seq_len, n_kv_heads, head_size);
    set<Alloc>(freq_cis, seq_len, head_size);
    set<Alloc>(token_embedding_table, vocab_size, dim);

    // compute the fp32 version of token embedding table
    dequantize(q_token_embedding_table, token_embedding_table);

    // compute frequency tables for RoPE
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

int argmax(float* x, int n) {
    // return the index with the highest value
    return std::distance(x, max_element(x, x + n));
}

void rmsnorm(float* o, Array& xa, float* w) {
    float* x = xa;
    int n = xa.size();

    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < n; j++)
        ss += x[j] * x[j];

    ss /= n;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < n; j++)
        o[j] = w[j] * (ss * x[j]);
}

void softmax(float* x, int n, float scale) {
    // find max value (for numerical stability)
    int id = argmax(x, n);
    float max_val = x[id];

    // apply scale/temperature, exp and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf((x[i] - max_val) / scale);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < n; i++)
        x[i] /= sum;
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

void matmul(float* o, const QArray& xa, const QArray* wp) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
    QuantizedTensor x = xa;
    QuantizedTensor w = *wp;
    int n = xa.size();
    int d = wp->size() / n;
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x.q[j + k]) * ((int32_t) w.q[in + j + k]);
            }
            val += ((float) ival) * w.s[(in + j) / GS] * x.s[j / GS];
            ival = 0;
        }
        o[i] = val;
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
    int head_size = dim / n_heads;
    int kv_dim = n_kv_heads * head_size;
    int kv_mul = dim / kv_dim; // integer multiplier of the kv sharing in multiquery

    // copy the token embedding into x
    memcpy(x, token_embedding_table(token), dim * sizeof(float));

    // pluck out the "pos" row of freq_cis
    float* freq = freq_cis(pos);

    // forward all the layers
    for (int l = 0; l < n_layers; l++) {

        // attention rmsnorm (Root Mean Square Normalization)
        rmsnorm(xb, x, rms_att_weight(l));

        // qkv matmuls for this position (V X M = V)
        // and save key,value at this time step (pos) to our kv cache
        quantize(q_x, xb);
        matmul(query, q_x, q_wq(l));
        matmul(key_cache(l, pos), q_x, q_wk(l));
        matmul(value_cache(l, pos), q_x, q_wv(l));

        // RoPE relative positional encoding: complex-valued
        // rotate query and key by freq in each head
        int h;
        float* key = key_cache(l, pos);
        #pragma omp parallel for private(h)
        for (h = 0; h < dim; h += 2) {
            int k = (h % head_size);
            complexmul(query+h, query+h+1, freq[k], freq[k+1]);
            if (h < kv_dim)
                complexmul(key+h, key+h+1, freq[k], freq[k+1]);
        }

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
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            // attention scores are first scaled down by sqrtf(head_size)
            softmax(att, pos + 1, sqrtf(head_size));

            // weighted sum of the values, store back into xb
            float* xb_h = xb(h);
            memset(xb_h, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = value_cache(l, t, h / kv_mul);
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb_h[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        quantize(q_x, xb);
        matmul(xb, q_x, q_wo(l));

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += xb[i];
        }

        // ffn rmsnorm
        rmsnorm(xb, x, rms_ffn_weight(l));

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(q_x, xb);
        matmul(hb, q_x, q_w1(l));
        matmul(hb2, q_x, q_w3(l));

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            hb[i] = val * hb2[i];
        }

        // final matmul to get the output of the ffn
        quantize(q_hb, hb);
        matmul(xb, q_hb, q_w2(l));

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, rms_final_weight);

    // classifier into logits
    quantize(q_x, x);
    matmul(logits, q_x, q_wcls_ptr);
    return logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

struct Tokenizer {

    Tokenizer(const string& tokenizer_path, int vocab_size);
    int str_lookup(const string& str);
    void encode(vector<int>* tokens_ptr, const string& text, bool bos, bool eos);
    string decode(int token, int prev_token);

    MMap mmap;

    vector<string> vocab;
    vector<float> vocab_scores;
    map<string, int> sorted_vocab;
};

Tokenizer::Tokenizer(const string& tokenizer_path, int vocab_size) : mmap(tokenizer_path) {

    // read in the tokenizer .bin file
    mmap.next<float>(); // ignore max_token_length
    for (int i = 0; i < vocab_size; i++) {
        float score = *mmap.next<float>();
        int32_t len = *mmap.next<int32_t>();
        char* c = mmap.next<char>(len);
        vocab.push_back(string(c, len));
        vocab_scores.push_back(score);
        sorted_vocab[vocab[i]] = i;
    }
}

int Tokenizer::str_lookup(const string& str) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    auto it = sorted_vocab.find(str);
    return it != sorted_vocab.end() ? it->second : -1;
}

void Tokenizer::encode(vector<int>* tokens_ptr, const string& text, bool bos, bool eos) {

    // encode the string text (input) into tokens[] vector
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    vector<int>& tokens = *tokens_ptr; // syntactic sugar
    string str;

    // add optional BOS token, if desired
    if (bos)
        tokens.push_back(BOS);

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (!text.empty())
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

    // save scores of every consecutive pair of tokens, then proceed merging
    // form the highest scored one, down to the lowest
    typedef pair<float, int> PairScore;
    PairScore none = make_pair(-1e10f, -1);
    vector<PairScore> sv(tokens.size()); // vector of (score, token) pairs
    int start = 0;
    int end = sv.size() - 1;

    while (true) {

        // find all possible merges between two consecutive tokens in [start, end)
        for (int i = start; i < end; i++) {
            str = vocab[tokens[i]] + vocab[tokens[i+1]];
            int id = str_lookup(str);
            sv[i] = (id != -1 ? make_pair(vocab_scores[id], id) : none);
        }

        // pick the best one with the highest score, ignore last element
        auto it = max_element(sv.begin(), sv.end() - 1);
        if (*it == none)
            break; // we couldn't find any more pairs to merge, so we're done

        // merge the consecutive pair (best_idx, best_idx+1) into a new token
        // delete token at position best_idx+1, shift the entire sequence back 1
        int best_idx = std::distance(sv.begin(), it);
        tokens[best_idx] = it->second;
        tokens.erase(tokens.begin() + best_idx + 1);
        sv.erase(it+1);

        // update scores at previous and current position
        start = std::max(best_idx - 1, 0);
        end = std::min(best_idx + 1, static_cast<int>(tokens.size() - 1));
    }

    // add optional EOS token, if desired
    if (eos)
        tokens.push_back(EOS);
}

string Tokenizer::decode(int token, int prev_token) {

    // following BOS token, sentencepiece decoder strips any leading whitespace (see PR #89)
    string token_str = vocab[token];
    if (prev_token == BOS && token_str[0] == ' ')
        token_str.erase(token_str.begin());

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
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

struct Sampler {

    Sampler(int n, float temp, float tp, unsigned long long rng_seed) {
        vocab_size = n;
        temperature = temp;
        topp = tp;
        rng_state = rng_seed;
    }

    int sample_topp(float* prob, float coin);
    int sample(float* logits);

    int vocab_size;
    float temperature;
    float topp;
    unsigned long long rng_state;
};

unsigned int random_u32(unsigned long long* state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1DULL) >> 32;
}

float random_f32(unsigned long long* state) {
    // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int Sampler::sample_topp(float* prob, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()
    vector<int> v;
    v.reserve(vocab_size);

    if (topp <= 0 || topp > 1) {
        // sample from the predicted probability distribution
        topp = 1.0f;
    }

    // values smaller than (1 - topp) / (vocab_size - 1) cannot be part of the
    // result so for efficiency we crop these out as candidates before sorting
    float cutoff = (1.0f - topp) / (vocab_size - 1);
    for (int i = 0; i < vocab_size; i++) {
        if (prob[i] >= cutoff)
            v.push_back(i);
    }

    if (topp < 1) {
        // sort indices in descending order of probabilities
        sort(v.begin(), v.end(), [prob](int a, int b) { return prob[a] > prob[b]; });

        // truncate the list where cumulative probability exceeds topp
        float cumulative_prob = 0.0f;
        for (size_t i = 0; i < v.size(); i++) {
            cumulative_prob += prob[v[i]];
            if (cumulative_prob > topp) {
                v.resize(i+1);
                break; // we've exceeded topp by including this last item
            }
        }
        coin *= cumulative_prob; // rescale coin into [0, cumulative_prob)
    }

    // sample index from probabilities (they must sum to 1!)
    for (size_t i = 0; i < v.size(); i++) {
        coin -= prob[v[i]];
        if (coin < 0)
            return v[i];
    }

    return v.back(); // in case of rounding errors
}

int Sampler::sample(float* logits) {

    // sample the token given the logits and some hyperparameters
    if (temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        return argmax(logits, vocab_size);
    }
    // apply softmax with temperature to the logits to get the probabilities for next token
    softmax(logits, vocab_size, temperature);

    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&rng_state);

    // if topp > 0 we perform top-p (nucleus) sampling, clamping the least likely
    // tokens to zero. Othewise sample from the predicted probability distribution.
    return sample_topp(logits, coin);
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation/chat loop

void generate(Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
              string prompt, string system_prompt, const Chat& chat, size_t steps) {

    vector<int> prompt_tokens;
    long start = 0; // used to time our code, only initialized after first iteration
    int token;      // will store the current/prev token in the sequence
    int next;       // will store the next token in the sequence
    size_t pos = 0; // position in the sequence
    bool bos, eos;  // poor man's named arguments
    bool is_user_turn = true; // when in chat mode, user starts

    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (is_user_turn) {

            if (chat) {
                // get the user prompt
                if (prompt.empty()) {
                    cout << "User: ";
                    if (!getline(cin, prompt))
                        return; // broken pipe, exit
                }
                // render user/system prompts into the Llama 2 Chat schema
                prompt = chat.format(prompt, system_prompt);
                cout << "Assistant: ";
            }

            // encode the (string) prompt into tokens sequence
            tokenizer.encode(&prompt_tokens, prompt, bos = true, eos = false);
            if (prompt_tokens.size() < 1) {
                cerr << "Something is wrong, expected at least 1 prompt token" << endl;
                exit(EXIT_FAILURE);
            }
            token = prompt_tokens[0]; // kick off with the first token in the prompt
            prompt_tokens.erase(prompt_tokens.begin()); // first token is already consumed
            is_user_turn = false;
            prompt = system_prompt = ""; // mark as consumed
        }

        // forward the transformer to get logits for the next token
        float* logits = transformer.forward(token, pos);

        // if we are still processing the input prompt, force the next prompt token
        // otherwise sample the next token from the logits
        next = pos < prompt_tokens.size() ? prompt_tokens[pos] : sampler.sample(logits);

        pos++; // increment before a possible early exit due to a BOS token

        // data-dependent terminating condition: the BOS token delimits sequences
        if (next == BOS && !chat)
            break;

        // decode the token into a string and print it
        // when in chat mode print only the Assistant response
        if (!chat || (pos >= prompt_tokens.size() && next != EOS))
            cout << tokenizer.decode(next, token) << std::flush;

        // EOS token ends the Assistant turn
        if (chat && next == EOS) {
            is_user_turn = true;
            cout << endl;
        }

        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0)
            start = time_in_ms();
    }

    // Generate: report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1 && !chat) {
        long elapsed = time_in_ms() - start;
        cerr << "\n\nachieved tok/s: " << (pos-1) / (double)(elapsed)*1000 << endl;
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
            "  -i <string> input prompt\n"
            "  -z <string> optional path to custom tokenizer\n"
            "  -m <string> mode: generate|chat, default: generate\n"
            "  -c <string> chat schema: llama|tinyllama, default: llama\n"
            "  -y <string> (optional) system prompt in chat mode\n" << endl;
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
    unsigned long long rng_seed = 0; // seed rng with time by default
    string mode = "generate"; // generate|chat
    string schema = "llama";  // chat schema llama|tinyllama
    string system_prompt;     // the (optional) system prompt to use in chat mode

    // 'checkpoint' is necessary, optional arguments and their values
    // come in pairs, so argc must be even.
    if (argc % 2 != 0)
        error_usage();

    checkpoint_path = string(argv[1]);

    // read in the args
    if (argc > 2) {
        vector<string> opt(argv + 2, argv + argc);
        for (size_t i = 0; i < opt.size(); i += 2) {
            if      (opt[i] == "-t") temperature = stof(opt[i+1]);
            else if (opt[i] == "-p") topp = stof(opt[i+1]);
            else if (opt[i] == "-s") rng_seed = stoi(opt[i+1]);
            else if (opt[i] == "-n") steps = stoi(opt[i+1]);
            else if (opt[i] == "-i") prompt = opt[i+1];
            else if (opt[i] == "-z") tokenizer_path = opt[i+1];
            else if (opt[i] == "-m") mode = opt[i+1];
            else if (opt[i] == "-c") schema = opt[i+1];
            else if (opt[i] == "-y") system_prompt = opt[i+1];
            else
                error_usage();
        }
    }

    // build the Transformer via the model .bin file
    Transformer transformer(checkpoint_path);
    print_model_info(transformer);

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps <= 0 || steps > transformer.seq_len) steps = transformer.seq_len; // ovrerride to ~max length
    if (mode != "generate" && mode != "chat") {
        cerr << "unknown mode: " << mode << endl;
        error_usage();
    }

    // try to get a system prompt
    Chat chat(mode == "chat", schema == "tinyllama");
    if (chat && system_prompt.empty()) {
        // system prompt was not passed in, attempt to get it from stdin
        cout << "Enter system prompt (optional): ";
        if (!getline(cin, system_prompt))
            return 0; // broken pipe, exit
    }

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer(tokenizer_path, transformer.vocab_size);

    // build the Sampler
    Sampler sampler(transformer.vocab_size, temperature, topp, rng_seed);

    // run!
    generate(transformer, tokenizer, sampler, prompt, system_prompt, chat, steps);

    return 0;
}
