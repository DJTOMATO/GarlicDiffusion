// OpenAI CLIP BPE tokenizer (SDXL CLIP-L & OpenCLIP-bigG).
#include "clip_tok.h"

#include <ctype.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#define O_BINARY 0
#endif

static uint64_t fnv1a(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t hash_u64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

static void init_byte_unicode(ClipTok* t) {
    int mapped[256] = {0};
    int cp[256];
    for (int b = '!'; b <= '~'; b++) { mapped[b] = 1; cp[b] = b; }
    for (int b = 0xA1; b <= 0xAC; b++) { mapped[b] = 1; cp[b] = b; }
    for (int b = 0xAE; b <= 0xFF; b++) { mapped[b] = 1; cp[b] = b; }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!mapped[b]) cp[b] = 256 + n++;
        int c = cp[b];
        if (c < 0x80) {
            t->byte_utf8[b][0] = (char)c;
            t->byte_utf8[b][1] = 0;
            t->byte_ulen[b] = 1;
        } else {
            t->byte_utf8[b][0] = (char)(0xC0 | (c >> 6));
            t->byte_utf8[b][1] = (char)(0x80 | (c & 0x3F));
            t->byte_utf8[b][2] = 0;
            t->byte_ulen[b] = 2;
        }
    }
}

static int32_t vocab_lookup(const ClipTok* t, const char* s, size_t len) {
    uint32_t idx = (uint32_t)fnv1a(s, len) & t->vht_mask;
    while (t->vht_id[idx] >= 0) {
        int32_t id = t->vht_id[idx];
        if (t->vlen[id] == len && memcmp(t->vocab[id], s, len) == 0) return id;
        idx = (idx + 1) & t->vht_mask;
    }
    return -1;
}

static int32_t merge_lookup(const ClipTok* t, uint32_t a, uint32_t b, uint32_t* out_id) {
    uint64_t key = ((uint64_t)a << 32) | b | 0x8000000000000000ull;
    uint32_t idx = (uint32_t)hash_u64(key) & t->mht_mask;
    while (t->mht_key[idx]) {
        if (t->mht_key[idx] == key) {
            uint32_t r = t->mht_rank[idx];
            if (out_id) *out_id = t->merge_out[r];
            return (int32_t)r;
        }
        idx = (idx + 1) & t->mht_mask;
    }
    return -1;
}

int clip_tok_load(ClipTok* t, const char* path) {
    memset(t, 0, sizeof(*t));
    init_byte_unicode(t);
    int fd = open(path, O_RDONLY | O_BINARY);
    if (fd < 0) return -1;
    off_t sz = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    if (sz < 12) { close(fd); return -1; }
    uint8_t* d = (uint8_t*)malloc((size_t)sz);
    if (!d) { close(fd); return -1; }
    size_t done = 0;
    while (done < (size_t)sz) {
        int r = (int)read(fd, d + done, (unsigned)((size_t)sz - done));
        if (r <= 0) break;
        done += (size_t)r;
    }
    close(fd);
    if (done != (size_t)sz || memcmp(d, "CTOK", 4) != 0) { free(d); return -1; }

    memcpy(&t->n_vocab, d + 4, 4);
    memcpy(&t->n_merges, d + 8, 4);

    uint32_t vht_sz = 1;
    while (vht_sz < t->n_vocab * 2) vht_sz <<= 1;
    t->vht_mask = vht_sz - 1;
    t->vht_id = (int32_t*)malloc(vht_sz * sizeof(int32_t));
    memset(t->vht_id, 0xFF, vht_sz * sizeof(int32_t));

    t->vocab = (char**)calloc(t->n_vocab, sizeof(char*));
    t->vlen = (uint16_t*)calloc(t->n_vocab, sizeof(uint16_t));

    size_t pos = 12;
    for (uint32_t i = 0; i < t->n_vocab; i++) {
        uint16_t len;
        memcpy(&len, d + pos, 2);
        pos += 2;
        t->vocab[i] = (char*)malloc(len + 1);
        memcpy(t->vocab[i], d + pos, len);
        t->vocab[i][len] = 0;
        t->vlen[i] = len;
        pos += len;
        uint32_t h = (uint32_t)fnv1a(t->vocab[i], len) & t->vht_mask;
        while (t->vht_id[h] >= 0) h = (h + 1) & t->vht_mask;
        t->vht_id[h] = (int32_t)i;
    }

    uint32_t mht_sz = 1;
    while (mht_sz < t->n_merges * 2) mht_sz <<= 1;
    t->mht_mask = mht_sz - 1;
    t->mht_key = (uint64_t*)calloc(mht_sz, sizeof(uint64_t));
    t->mht_rank = (uint32_t*)calloc(mht_sz, sizeof(uint32_t));
    t->merge_out = (uint32_t*)calloc(t->n_merges, sizeof(uint32_t));

    for (uint32_t i = 0; i < t->n_merges; i++) {
        uint32_t a, b, ab;
        memcpy(&a, d + pos, 4);
        memcpy(&b, d + pos + 4, 4);
        memcpy(&ab, d + pos + 8, 4);
        pos += 12;
        t->merge_out[i] = ab;
        uint64_t key = ((uint64_t)a << 32) | b | 0x8000000000000000ull;
        uint32_t h = (uint32_t)hash_u64(key) & t->mht_mask;
        while (t->mht_key[h]) h = (h + 1) & t->mht_mask;
        t->mht_key[h] = key;
        t->mht_rank[h] = i;
    }

    free(d);
    return 0;
}

void clip_tok_free(ClipTok* t) {
    if (t->vocab) {
        for (uint32_t i = 0; i < t->n_vocab; i++) free(t->vocab[i]);
        free(t->vocab);
    }
    free(t->vlen);
    free(t->vht_id);
    free(t->mht_key);
    free(t->mht_rank);
    free(t->merge_out);
    memset(t, 0, sizeof(*t));
}

static void encode_word(const ClipTok* t, const uint8_t* w, size_t wlen,
                        uint32_t* dst, uint32_t* n_dst, uint32_t max_dst) {
    if (wlen == 0 || *n_dst >= max_dst) return;
    uint32_t syms[256];
    uint32_t ns = 0;
    for (size_t i = 0; i < wlen && ns < 256; i++) {
        char buf[16];
        uint8_t b = w[i];
        memcpy(buf, t->byte_utf8[b], t->byte_ulen[b]);
        size_t blen = t->byte_ulen[b];
        if (i + 1 == wlen) {
            memcpy(buf + blen, "</w>", 4);
            blen += 4;
        }
        int32_t id = vocab_lookup(t, buf, blen);
        if (id >= 0) syms[ns++] = (uint32_t)id;
    }
    // Greedy BPE merge loop
    while (ns > 1) {
        int32_t best_rank = 0x7FFFFFFF;
        uint32_t best_pos = 0, best_out = 0;
        for (uint32_t i = 0; i + 1 < ns; i++) {
            uint32_t out_id = 0;
            int32_t r = merge_lookup(t, syms[i], syms[i + 1], &out_id);
            if (r >= 0 && r < best_rank) {
                best_rank = r;
                best_pos = i;
                best_out = out_id;
            }
        }
        if (best_rank == 0x7FFFFFFF) break;
        syms[best_pos] = best_out;
        for (uint32_t i = best_pos + 1; i + 1 < ns; i++) syms[i] = syms[i + 1];
        ns--;
    }
    for (uint32_t i = 0; i < ns && *n_dst < max_dst; i++) dst[(*n_dst)++] = syms[i];
}

// CLIP pre-tokenization: lowercase + contractions + letters + digits + non-space punctuation runs
uint32_t clip_tok_encode(const ClipTok* t, const char* text, uint32_t out_ids[CLIP_MAX_LEN], uint32_t pad_id) {
    uint32_t n = 0;
    out_ids[n++] = CLIP_BOS;
    const uint8_t* p = (const uint8_t*)(text ? text : "");
    uint8_t word[256];

    while (*p && n < CLIP_MAX_LEN - 1) {
        while (*p && isspace(*p)) p++;
        if (!*p) break;

        size_t wl = 0;
        // Check English contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if (*p == '\'') {
            char c1 = (char)tolower(p[1]);
            char c2 = p[1] ? (char)tolower(p[2]) : 0;
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                word[0] = '\''; word[1] = (uint8_t)c1; word[2] = (uint8_t)c2; wl = 3; p += 3;
            } else if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                word[0] = '\''; word[1] = (uint8_t)c1; wl = 2; p += 2;
            }
        }
        if (wl == 0) {
            if (isalpha(*p) || (*p >= 0x80)) {
                while (*p && (isalpha(*p) || *p >= 0x80) && wl < 250) {
                    word[wl++] = (*p < 0x80) ? (uint8_t)tolower(*p) : *p;
                    p++;
                }
            } else if (isdigit(*p)) {
                // CLIP regex splits digits one-by-one (\p{N})
                word[wl++] = *p++;
            } else {
                while (*p && !isspace(*p) && !isalnum(*p) && *p < 0x80 && wl < 250) {
                    if (*p == '\'' && wl > 0) break;
                    word[wl++] = *p++;
                }
            }
        }
        encode_word(t, word, wl, out_ids, &n, CLIP_MAX_LEN - 1);
    }

    uint32_t eos_pos = n;
    out_ids[n++] = CLIP_EOS;
    while (n < CLIP_MAX_LEN) out_ids[n++] = pad_id;
    return eos_pos;
}
