#ifndef CLIP_TOK_H
#define CLIP_TOK_H

#include <stdint.h>

#define CLIP_MAX_LEN 77
#define CLIP_BOS 49406u
#define CLIP_EOS 49407u

typedef struct {
    uint32_t n_vocab;
    uint32_t n_merges;
    char** vocab;
    uint16_t* vlen;
    // Open-addressed hash table: token string -> id
    uint32_t vht_mask;
    int32_t* vht_id;
    // Open-addressed hash table: (a, b) -> merge index (rank)
    uint32_t mht_mask;
    uint64_t* mht_key;
    uint32_t* mht_rank;
    uint32_t* merge_out;
    // Byte -> CLIP unicode UTF-8 string
    char byte_utf8[256][4];
    uint8_t byte_ulen[256];
} ClipTok;

int clip_tok_load(ClipTok* t, const char* path);
void clip_tok_free(ClipTok* t);

// Encodes UTF-8 prompt into 77 token IDs:
//   pad_id = 49407 for CLIP-L, pad_id = 0 for OpenCLIP-bigG.
// Returns index of EOS token (0..76) for pooled embedding lookup.
uint32_t clip_tok_encode(const ClipTok* t, const char* text, uint32_t out_ids[CLIP_MAX_LEN], uint32_t pad_id);

#endif
