#ifndef PNG_WRITE_H
#define PNG_WRITE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t png_crc32(const uint8_t* data, size_t len) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        c ^= data[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(c & 1u)));
    }
    return c ^ 0xFFFFFFFFu;
}

static void png_put32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

static size_t png_chunk(uint8_t* dst, const char type[4], const uint8_t* data, uint32_t len) {
    png_put32(dst, len);
    memcpy(dst + 4, type, 4);
    if (len && data) memcpy(dst + 8, data, len);
    uint32_t crc = png_crc32(dst + 4, 4 + len);
    png_put32(dst + 8 + len, crc);
    return 12 + len;
}

// Encode RGB888 [h][w][3] to an uncompressed valid PNG buffer (caller frees).
// If parameters_text is non-NULL/non-empty, injects a standard A1111/WebUI PNG tEXt "parameters" chunk!
static inline uint8_t* png_encode_rgb_meta(const uint8_t* rgb, uint32_t w, uint32_t h,
                                           const char* parameters_text, size_t* out_size) {
    size_t row_bytes = (size_t)w * 3 + 1;
    size_t raw_len = row_bytes * h;
    uint8_t* raw = (uint8_t*)malloc(raw_len);
    if (!raw) return NULL;

    uint32_t s1 = 1, s2 = 0;
    for (uint32_t y = 0; y < h; y++) {
        uint8_t* row = raw + y * row_bytes;
        row[0] = 0; // filter None
        memcpy(row + 1, rgb + (size_t)y * w * 3, (size_t)w * 3);
        for (size_t i = 0; i < row_bytes; i++) {
            s1 += row[i];
            if (s1 >= 65521u) s1 -= 65521u;
            s2 += s1;
            if (s2 >= 65521u) s2 -= 65521u;
        }
    }

    // Stored zlib stream: 2-byte header + ceil(raw_len / 65535) * 5-byte block headers + raw_len + 4-byte adler32
    size_t nblocks = (raw_len + 65534) / 65535;
    if (nblocks == 0) nblocks = 1;
    size_t zlen = 2 + nblocks * 5 + raw_len + 4;
    uint8_t* zbuf = (uint8_t*)malloc(zlen);
    if (!zbuf) { free(raw); return NULL; }

    size_t zp = 0;
    zbuf[zp++] = 0x78; // CMF
    zbuf[zp++] = 0x01; // FLG (no compression)
    size_t rem = raw_len, off = 0;
    while (rem > 0 || off == 0) {
        uint16_t blen = (rem > 65535) ? 65535 : (uint16_t)rem;
        int last = (rem <= 65535);
        zbuf[zp++] = last ? 0x01 : 0x00;
        zbuf[zp++] = (uint8_t)(blen & 0xFF);
        zbuf[zp++] = (uint8_t)(blen >> 8);
        uint16_t nlen = (uint16_t)(~blen);
        zbuf[zp++] = (uint8_t)(nlen & 0xFF);
        zbuf[zp++] = (uint8_t)(nlen >> 8);
        if (blen) {
            memcpy(zbuf + zp, raw + off, blen);
            zp += blen;
            off += blen;
            rem -= blen;
        }
        if (last) break;
    }
    png_put32(zbuf + zp, (s2 << 16) | s1);
    zp += 4;
    free(raw);

    // Optional A1111 PNG tEXt chunk: "parameters\0<text>"
    size_t meta_len = (parameters_text && parameters_text[0]) ? strlen(parameters_text) : 0;
    size_t text_chunk_data = meta_len ? (11 + meta_len) : 0;
    size_t text_chunk_total = text_chunk_data ? (12 + text_chunk_data) : 0;

    // Assemble PNG
    size_t max_png = 8 + (12 + 13) + text_chunk_total + (12 + zp) + 12;
    uint8_t* png = (uint8_t*)malloc(max_png);
    if (!png) { free(zbuf); return NULL; }
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    memcpy(png, sig, 8);
    size_t pos = 8;

    uint8_t ihdr[13];
    png_put32(ihdr, w);
    png_put32(ihdr + 4, h);
    ihdr[8] = 8;  // bit depth
    ihdr[9] = 2;  // RGB
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    pos += png_chunk(png + pos, "IHDR", ihdr, 13);
    if (text_chunk_data) {
        uint8_t* tdata = (uint8_t*)malloc(text_chunk_data);
        if (tdata) {
            memcpy(tdata, "parameters\0", 11);
            memcpy(tdata + 11, parameters_text, meta_len);
            pos += png_chunk(png + pos, "tEXt", tdata, (uint32_t)text_chunk_data);
            free(tdata);
        }
    }
    pos += png_chunk(png + pos, "IDAT", zbuf, (uint32_t)zp);
    pos += png_chunk(png + pos, "IEND", NULL, 0);
    free(zbuf);
    *out_size = pos;
    return png;
}

static inline uint8_t* png_encode_rgb(const uint8_t* rgb, uint32_t w, uint32_t h, size_t* out_size) {
    return png_encode_rgb_meta(rgb, w, h, NULL, out_size);
}

static inline uint32_t png_get32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

// Fast IHDR probe: extract width/height from any valid PNG header without decoding pixels.
static inline int png_probe_dims(const uint8_t* png, size_t len, uint32_t* w, uint32_t* h) {
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (!png || len < 33 || memcmp(png, sig, 8) != 0 || memcmp(png + 12, "IHDR", 4) != 0) return -1;
    *w = png_get32(png + 16);
    *h = png_get32(png + 20);
    return (*w > 0 && *h > 0 && *w <= 8192 && *h <= 8192) ? 0 : -1;
}

// Extract A1111 "parameters" tEXt chunk from a PNG buffer (returns bytes copied, or 0 if not present).
static inline size_t png_extract_parameters(const uint8_t* png, size_t len, char* out, size_t out_max) {
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (!png || len < 33 || memcmp(png, sig, 8) != 0 || !out || !out_max) return 0;
    out[0] = 0;
    size_t pos = 8;
    while (pos + 12 <= len) {
        uint32_t clen = png_get32(png + pos);
        if (pos + 12 + clen > len) break;
        if (memcmp(png + pos + 4, "tEXt", 4) == 0 && clen > 11 &&
            memcmp(png + pos + 8, "parameters\0", 11) == 0) {
            size_t n = clen - 11;
            if (n >= out_max) n = out_max - 1;
            memcpy(out, png + pos + 19, n);
            out[n] = 0;
            return n;
        }
        pos += 12 + clen;
    }
    return 0;
}

// Decode stored-DEFLATE 8-bit RGB PNG (format written by png_encode_rgb) back to RGB888 (caller frees).
static inline uint8_t* png_decode_rgb(const uint8_t* png, size_t len, uint32_t* out_w, uint32_t* out_h) {
    uint32_t w = 0, h = 0;
    if (png_probe_dims(png, len, &w, &h) != 0) return NULL;
    if (png[24] != 8 || png[25] != 2) return NULL; // 8-bit RGB only

    // Collect concatenated IDAT payload
    size_t idat_tot = 0, pos = 8;
    while (pos + 12 <= len) {
        uint32_t clen = png_get32(png + pos);
        if (pos + 12 + clen > len) break;
        if (memcmp(png + pos + 4, "IDAT", 4) == 0) idat_tot += clen;
        pos += 12 + clen;
    }
    if (idat_tot < 11) return NULL;
    uint8_t* zbuf = (uint8_t*)malloc(idat_tot);
    if (!zbuf) return NULL;
    size_t zp = 0;
    pos = 8;
    while (pos + 12 <= len) {
        uint32_t clen = png_get32(png + pos);
        if (pos + 12 + clen > len) break;
        if (memcmp(png + pos + 4, "IDAT", 4) == 0) {
            memcpy(zbuf + zp, png + pos + 8, clen);
            zp += clen;
        }
        pos += 12 + clen;
    }

    size_t row_bytes = (size_t)w * 3 + 1;
    size_t raw_len = row_bytes * h;
    uint8_t* raw = (uint8_t*)malloc(raw_len);
    if (!raw) { free(zbuf); return NULL; }

    size_t zi = 2, ro = 0; // skip 2-byte zlib header
    int ok = 1;
    while (zi + 5 <= zp && ro < raw_len) {
        uint8_t bhdr = zbuf[zi++];
        if ((bhdr & 0x06u) != 0) { ok = 0; break; } // only stored blocks (BTYPE=00)
        uint16_t blen = (uint16_t)zbuf[zi] | ((uint16_t)zbuf[zi + 1] << 8);
        zi += 4; // skip LEN + NLEN
        if (zi + blen > zp || ro + blen > raw_len) { ok = 0; break; }
        memcpy(raw + ro, zbuf + zi, blen);
        zi += blen;
        ro += blen;
        if (bhdr & 0x01u) break;
    }
    free(zbuf);
    if (!ok || ro != raw_len) { free(raw); return NULL; }

    uint8_t* rgb = (uint8_t*)malloc((size_t)w * h * 3);
    if (!rgb) { free(raw); return NULL; }
    for (uint32_t y = 0; y < h; y++) {
        memcpy(rgb + (size_t)y * w * 3, raw + (size_t)y * row_bytes + 1, (size_t)w * 3);
    }
    free(raw);
    *out_w = w;
    *out_h = h;
    return rgb;
}

// High-quality area/bilinear RGB888 resizer (caller frees).
static inline uint8_t* rgb_resize(const uint8_t* src, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh) {
    if (!src || sw == 0 || sh == 0 || dw == 0 || dh == 0) return NULL;
    uint8_t* dst = (uint8_t*)malloc((size_t)dw * dh * 3);
    if (!dst) return NULL;
    float sx = (float)sw / (float)dw;
    float sy = (float)sh / (float)dh;
    float rx = sx > 1.0f ? sx * 0.5f : 1.0f;
    float ry = sy > 1.0f ? sy * 0.5f : 1.0f;
    for (uint32_t oy = 0; oy < dh; oy++) {
        float fy = ((float)oy + 0.5f) * sy - 0.5f;
        int y0 = (int)(fy - ry); if (y0 < 0) y0 = 0;
        int y1 = (int)(fy + ry + 1.0f); if (y1 >= (int)sh) y1 = (int)sh - 1;
        for (uint32_t ox = 0; ox < dw; ox++) {
            float fx = ((float)ox + 0.5f) * sx - 0.5f;
            int x0 = (int)(fx - rx); if (x0 < 0) x0 = 0;
            int x1 = (int)(fx + rx + 1.0f); if (x1 >= (int)sw) x1 = (int)sw - 1;
            float r = 0, g = 0, b = 0, ws = 0;
            for (int iy = y0; iy <= y1; iy++) {
                float dy = (float)iy - fy; if (dy < 0) dy = -dy;
                float ky = 1.0f - dy / ry;
                if (ky <= 0.0f) continue;
                const uint8_t* sr = src + (size_t)iy * sw * 3;
                for (int ix = x0; ix <= x1; ix++) {
                    float dx = (float)ix - fx; if (dx < 0) dx = -dx;
                    float kx = 1.0f - dx / rx;
                    if (kx <= 0.0f) continue;
                    float k = kx * ky;
                    r += sr[ix * 3 + 0] * k;
                    g += sr[ix * 3 + 1] * k;
                    b += sr[ix * 3 + 2] * k;
                    ws += k;
                }
            }
            if (ws <= 0.0f) {
                int iy = (int)(fy + 0.5f), ix = (int)(fx + 0.5f);
                if (iy < 0) iy = 0; if (iy >= (int)sh) iy = (int)sh - 1;
                if (ix < 0) ix = 0; if (ix >= (int)sw) ix = (int)sw - 1;
                memcpy(dst + ((size_t)oy * dw + ox) * 3, src + ((size_t)iy * sw + ix) * 3, 3);
            } else {
                float inv = 1.0f / ws;
                uint8_t* dp = dst + ((size_t)oy * dw + ox) * 3;
                float vr = r * inv + 0.5f, vg = g * inv + 0.5f, vb = b * inv + 0.5f;
                dp[0] = vr < 0 ? 0 : (vr > 255 ? 255 : (uint8_t)vr);
                dp[1] = vg < 0 ? 0 : (vg > 255 ? 255 : (uint8_t)vg);
                dp[2] = vb < 0 ? 0 : (vb > 255 ? 255 : (uint8_t)vb);
            }
        }
    }
    return dst;
}

#endif
