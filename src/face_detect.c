// Garlic Diffusion: Lightweight Pure-C CPU Anime Face Detector & Feathered Face Crop/Blender
// Zero external dependencies (no OpenCV/ONNX/PyTorch); runs in <40ms on PS4 Jaguar CPU.
#include "face_detect.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Classify anime skin pixel (covers pale/porcelain anime skin, warm sunlit skin, and shaded peach skin)
static inline int is_anime_skin(uint8_t r, uint8_t g, uint8_t b) {
    if (r < 115 || g < 85 || b < 70) return 0;
    int maxv = (r > g) ? (r > b ? r : b) : (g > b ? g : b);
    int minv = (r < g) ? (r < b ? r : b) : (g < b ? g : b);
    if (maxv - minv > 105) return 0; // exclude vivid saturated clothing/hair
    // Anime skin: R >= G >= B-12, slightly warm tint, not pure gray/white background
    if ((int)r + 6 < (int)g) return 0;
    if ((int)r < (int)b + 4) return 0;
    if (r > 248 && g > 246 && b > 244) return 0; // pure white background
    float cb = 128.0f - 0.168736f * r - 0.331264f * g + 0.5f * b;
    float cr = 128.0f + 0.5f * r - 0.418688f * g - 0.081312f * b;
    return (cb >= 78.0f && cb <= 132.0f && cr >= 127.0f && cr <= 174.0f);
}

static inline float box_sum(const float* ii, uint32_t stride, int x0, int y0, int x1, int y1) {
    return ii[(size_t)y1 * stride + x1] - ii[(size_t)y0 * stride + x1]
         - ii[(size_t)y1 * stride + x0] + ii[(size_t)y0 * stride + x0];
}

static float box_iou(const FaceBox* a, const FaceBox* b) {
    int x0 = a->x > b->x ? a->x : b->x;
    int y0 = a->y > b->y ? a->y : b->y;
    int x1 = (a->x + a->w) < (b->x + b->w) ? (a->x + a->w) : (b->x + b->w);
    int y1 = (a->y + a->h) < (b->y + b->h) ? (a->y + a->h) : (b->y + b->h);
    if (x1 <= x0 || y1 <= y0) return 0.0f;
    float inter = (float)(x1 - x0) * (float)(y1 - y0);
    float uni = (float)a->w * a->h + (float)b->w * b->h - inter;
    return uni > 0.0f ? (inter / uni) : 0.0f;
}

int face_detect_anime(const uint8_t* rgb, uint32_t img_w, uint32_t img_h,
                      FaceBox* out_boxes, int max_boxes) {
    if (!rgb || !out_boxes || max_boxes <= 0 || img_w < 64 || img_h < 64) return 0;

    // Downscale image to max edge 240px for fast O(1) integral-image multi-scale scanning
    const uint32_t MAX_D = 240u;
    uint32_t dw = img_w, dh = img_h;
    if (img_w >= img_h && img_w > MAX_D) {
        dw = MAX_D;
        dh = (uint32_t)(((uint64_t)img_h * MAX_D + img_w / 2u) / img_w);
    } else if (img_h > img_w && img_h > MAX_D) {
        dh = MAX_D;
        dw = (uint32_t)(((uint64_t)img_w * MAX_D + img_h / 2u) / img_h);
    }
    if (dw < 32u) dw = 32u;
    if (dh < 32u) dh = 32u;

    uint32_t st = dw + 1u;
    size_t ii_n = (size_t)(dh + 1u) * st;
    float* ii_skin = (float*)calloc(ii_n, sizeof(float));
    float* ii_luma = (float*)calloc(ii_n, sizeof(float));
    float* ii_edge = (float*)calloc(ii_n, sizeof(float));
    uint8_t* luma  = (uint8_t*)malloc((size_t)dw * dh);
    if (!ii_skin || !ii_luma || !ii_edge || !luma) {
        free(ii_skin); free(ii_luma); free(ii_edge); free(luma);
        return 0;
    }

    for (uint32_t y = 0; y < dh; y++) {
        uint32_t sy = (uint32_t)(((uint64_t)y * img_h) / dh);
        for (uint32_t x = 0; x < dw; x++) {
            uint32_t sx = (uint32_t)(((uint64_t)x * img_w) / dw);
            const uint8_t* p = rgb + ((size_t)sy * img_w + sx) * 3u;
            uint8_t r = p[0], g = p[1], b = p[2];
            uint8_t lum = (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
            luma[(size_t)y * dw + x] = lum;
            float sk = is_anime_skin(r, g, b) ? 1.0f : 0.0f;
            float ed = 0.0f;
            if (x > 0 && y > 0) {
                int gx = abs((int)lum - (int)luma[(size_t)y * dw + (x - 1u)]);
                int gy = abs((int)lum - (int)luma[(size_t)(y - 1u) * dw + x]);
                ed = (float)(gx + gy) / 255.0f;
            }
            size_t idx = (size_t)(y + 1u) * st + (x + 1u);
            ii_skin[idx] = sk + ii_skin[idx - 1] + ii_skin[idx - st] - ii_skin[idx - st - 1];
            ii_luma[idx] = (float)lum / 255.0f + ii_luma[idx - 1] + ii_luma[idx - st] - ii_luma[idx - st - 1];
            ii_edge[idx] = ed + ii_edge[idx - 1] + ii_edge[idx - st] - ii_edge[idx - st - 1];
        }
    }

    FaceBox cands[64];
    int n_cands = 0;
    int min_s = (int)(dw < dh ? dw : dh) / 10;
    int max_s = (int)(dw < dh ? dw : dh) * 58 / 100;
    if (min_s < 16) min_s = 16;

    for (int sz = min_s; sz <= max_s; sz = (int)(sz * 1.18f) + 1) {
        int step = sz / 8;
        if (step < 2) step = 2;
        for (int y = 0; y + sz <= (int)dh; y += step) {
            for (int x = 0; x + sz <= (int)dw; x += step) {
                float area = (float)(sz * sz);
                // Core face skin ratio (center 70% width, lower 65% height)
                int cx0 = x + sz * 15 / 100, cx1 = x + sz * 85 / 100;
                int cy0 = y + sz * 30 / 100, cy1 = y + sz * 92 / 100;
                float carea = (float)((cx1 - cx0) * (cy1 - cy0));
                if (carea < 16.0f) continue;
                float skin_core = box_sum(ii_skin, st, cx0, cy0, cx1, cy1) / carea;
                if (skin_core < 0.34f || skin_core > 0.98f) continue;

                // Left & right eye zones (y: 35%..58%, x: 16%..44% and 56%..84%)
                int ey0 = y + sz * 34 / 100, ey1 = y + sz * 58 / 100;
                int lx0 = x + sz * 16 / 100, lx1 = x + sz * 44 / 100;
                int rx0 = x + sz * 56 / 100, rx1 = x + sz * 84 / 100;
                int nx0 = x + sz * 44 / 100, nx1 = x + sz * 56 / 100;
                float ea = (float)((lx1 - lx0) * (ey1 - ey0));
                float na = (float)((nx1 - nx0) * (ey1 - ey0));
                if (ea < 4.0f || na < 2.0f) continue;

                float l_edge = box_sum(ii_edge, st, lx0, ey0, lx1, ey1) / ea;
                float r_edge = box_sum(ii_edge, st, rx0, ey0, rx1, ey1) / ea;
                float l_lum  = box_sum(ii_luma, st, lx0, ey0, lx1, ey1) / ea;
                float r_lum  = box_sum(ii_luma, st, rx0, ey0, rx1, ey1) / ea;
                float n_lum  = box_sum(ii_luma, st, nx0, ey0, nx1, ey1) / na;

                // Cheek/chin skin zone (y: 62%..88%) should be brighter/smoother than eye band
                int ch0 = y + sz * 62 / 100, ch1 = y + sz * 88 / 100;
                float cha = (float)((cx1 - cx0) * (ch1 - ch0));
                float ch_lum  = cha > 1.0f ? (box_sum(ii_luma, st, cx0, ch0, cx1, ch1) / cha) : 0.5f;
                float ch_skin = cha > 1.0f ? (box_sum(ii_skin, st, cx0, ch0, cx1, ch1) / cha) : 0.0f;

                // Hair crown above forehead (y: 0%..25%) tends to have lower skin ratio than face core
                int hy1 = y + sz * 25 / 100;
                float ha = (float)(sz * (hy1 - y));
                float top_skin = ha > 1.0f ? (box_sum(ii_skin, st, x, y, x + sz, hy1) / ha) : 0.5f;

                float eye_contrast = (ch_lum - 0.5f * (l_lum + r_lum)) + 0.5f * (n_lum - 0.5f * (l_lum + r_lum));
                float eye_sym = 1.0f - fabsf(l_lum - r_lum) * 2.2f - fabsf(l_edge - r_edge) * 1.4f;
                if (eye_sym < 0.0f) eye_sym = 0.0f;
                float eye_detail = fminf(l_edge, r_edge);
                if (eye_detail < 0.055f) continue;

                // Slight vertical prior: anime faces are usually in the upper 65% of full/upper-body shots
                float cy_norm = ((float)y + 0.5f * sz) / (float)dh;
                float pos_prior = (cy_norm < 0.58f) ? 1.15f : (cy_norm < 0.78f ? 0.92f : 0.70f);
                float crown_bonus = fmaxf(0.0f, skin_core - top_skin);

                float score = (skin_core * 1.35f + ch_skin * 0.65f + eye_detail * 2.6f
                               + fmaxf(0.0f, eye_contrast) * 2.4f + crown_bonus * 0.95f)
                              * (0.55f + 0.45f * eye_sym) * pos_prior;
                // Mild preference against tiny noisy boxes
                score *= (0.85f + 0.15f * sqrtf(area / (float)(dw * dh)));

                if (score < 1.05f) continue;

                FaceBox fb;
                fb.x = (int)((double)x * img_w / dw);
                fb.y = (int)((double)y * img_h / dh);
                fb.w = (int)((double)sz * img_w / dw);
                fb.h = (int)((double)sz * img_h / dh);
                fb.score = score;

                if (n_cands < 64) {
                    cands[n_cands++] = fb;
                } else {
                    int min_i = 0;
                    for (int k = 1; k < 64; k++) if (cands[k].score < cands[min_i].score) min_i = k;
                    if (fb.score > cands[min_i].score) cands[min_i] = fb;
                }
            }
        }
    }

    free(ii_skin); free(ii_luma); free(ii_edge); free(luma);

    // Sort candidates descending by score
    for (int i = 0; i < n_cands; i++) {
        for (int j = i + 1; j < n_cands; j++) {
            if (cands[j].score > cands[i].score) {
                FaceBox tmp = cands[i]; cands[i] = cands[j]; cands[j] = tmp;
            }
        }
    }

    // Non-maximum suppression
    int out_n = 0;
    for (int i = 0; i < n_cands && out_n < max_boxes; i++) {
        int overlap = 0;
        for (int j = 0; j < out_n; j++) {
            if (box_iou(&cands[i], &out_boxes[j]) > 0.25f) { overlap = 1; break; }
        }
        if (!overlap) out_boxes[out_n++] = cands[i];
    }
    return out_n;
}

void face_box_expand_square(const FaceBox* in, uint32_t img_w, uint32_t img_h,
                            float pad_scale, FaceBox* out) {
    if (!in || !out) return;
    if (pad_scale < 1.1f) pad_scale = 1.1f;
    float cx = (float)in->x + 0.5f * (float)in->w;
    float cy = (float)in->y + 0.5f * (float)in->h;
    int side = (int)(fmaxf((float)in->w, (float)in->h) * pad_scale + 0.5f);
    int max_side = (int)(img_w < img_h ? img_w : img_h);
    if (side > max_side) side = max_side;
    if (side < 64) side = 64;
    int x0 = (int)(cx - 0.5f * side);
    int y0 = (int)(cy - 0.5f * side);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 + side > (int)img_w) x0 = (int)img_w - side;
    if (y0 + side > (int)img_h) y0 = (int)img_h - side;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    out->x = x0;
    out->y = y0;
    out->w = side;
    out->h = side;
    out->score = in->score;
}

uint8_t* face_crop_resize(const uint8_t* src_rgb, uint32_t img_w, uint32_t img_h,
                          const FaceBox* box, uint32_t dst_w, uint32_t dst_h) {
    if (!src_rgb || !box || !dst_w || !dst_h || box->w <= 0 || box->h <= 0) return NULL;
    uint8_t* out = (uint8_t*)malloc((size_t)dst_w * dst_h * 3u);
    if (!out) return NULL;
    for (uint32_t y = 0; y < dst_h; y++) {
        float fy = (float)box->y + ((float)y + 0.5f) * (float)box->h / (float)dst_h - 0.5f;
        int y0 = (int)floorf(fy), y1 = y0 + 1;
        float wy = fy - (float)y0;
        if (y0 < 0) y0 = 0;
        if (y1 < 0) y1 = 0;
        if (y0 >= (int)img_h) y0 = (int)img_h - 1;
        if (y1 >= (int)img_h) y1 = (int)img_h - 1;
        for (uint32_t x = 0; x < dst_w; x++) {
            float fx = (float)box->x + ((float)x + 0.5f) * (float)box->w / (float)dst_w - 0.5f;
            int x0 = (int)floorf(fx), x1 = x0 + 1;
            float wx = fx - (float)x0;
            if (x0 < 0) x0 = 0;
            if (x1 < 0) x1 = 0;
            if (x0 >= (int)img_w) x0 = (int)img_w - 1;
            if (x1 >= (int)img_w) x1 = (int)img_w - 1;
            const uint8_t* p00 = src_rgb + ((size_t)y0 * img_w + x0) * 3u;
            const uint8_t* p10 = src_rgb + ((size_t)y0 * img_w + x1) * 3u;
            const uint8_t* p01 = src_rgb + ((size_t)y1 * img_w + x0) * 3u;
            const uint8_t* p11 = src_rgb + ((size_t)y1 * img_w + x1) * 3u;
            uint8_t* dp = out + ((size_t)y * dst_w + x) * 3u;
            for (int c = 0; c < 3; c++) {
                float t0 = p00[c] * (1.0f - wx) + p10[c] * wx;
                float t1 = p01[c] * (1.0f - wx) + p11[c] * wx;
                float v = t0 * (1.0f - wy) + t1 * wy;
                int iv = (int)(v + 0.5f);
                dp[c] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
            }
        }
    }
    return out;
}

void face_blend_back(uint8_t* dst_rgb, uint32_t img_w, uint32_t img_h,
                     const uint8_t* patch_rgb, uint32_t patch_w, uint32_t patch_h,
                     const FaceBox* box, float feather_frac) {
    if (!dst_rgb || !patch_rgb || !box || box->w <= 0 || box->h <= 0 || !patch_w || !patch_h) return;
    if (feather_frac < 0.04f) feather_frac = 0.04f;
    if (feather_frac > 0.35f) feather_frac = 0.35f;
    float fw = fmaxf(4.0f, (float)box->w * feather_frac);
    float fh = fmaxf(4.0f, (float)box->h * feather_frac);

    for (int by = 0; by < box->h; by++) {
        int dy = box->y + by;
        if (dy < 0 || dy >= (int)img_h) continue;
        float fy = ((float)by + 0.5f) * (float)patch_h / (float)box->h - 0.5f;
        int y0 = (int)floorf(fy), y1 = y0 + 1;
        float wy = fy - (float)y0;
        if (y0 < 0) y0 = 0;
        if (y1 < 0) y1 = 0;
        if (y0 >= (int)patch_h) y0 = (int)patch_h - 1;
        if (y1 >= (int)patch_h) y1 = (int)patch_h - 1;

        float my = 1.0f;
        if ((float)by < fh) my = (float)by / fh;
        else if ((float)(box->h - 1 - by) < fh) my = (float)(box->h - 1 - by) / fh;
        // Smoothstep feathering
        my = my * my * (3.0f - 2.0f * my);

        for (int bx = 0; bx < box->w; bx++) {
            int dx = box->x + bx;
            if (dx < 0 || dx >= (int)img_w) continue;
            float fx = ((float)bx + 0.5f) * (float)patch_w / (float)box->w - 0.5f;
            int x0 = (int)floorf(fx), x1 = x0 + 1;
            float wx = fx - (float)x0;
            if (x0 < 0) x0 = 0;
            if (x1 < 0) x1 = 0;
            if (x0 >= (int)patch_w) x0 = (int)patch_w - 1;
            if (x1 >= (int)patch_w) x1 = (int)patch_w - 1;

            float mx = 1.0f;
            if ((float)bx < fw) mx = (float)bx / fw;
            else if ((float)(box->w - 1 - bx) < fw) mx = (float)(box->w - 1 - bx) / fw;
            mx = mx * mx * (3.0f - 2.0f * mx);

            float alpha = mx * my;
            const uint8_t* p00 = patch_rgb + ((size_t)y0 * patch_w + x0) * 3u;
            const uint8_t* p10 = patch_rgb + ((size_t)y0 * patch_w + x1) * 3u;
            const uint8_t* p01 = patch_rgb + ((size_t)y1 * patch_w + x0) * 3u;
            const uint8_t* p11 = patch_rgb + ((size_t)y1 * patch_w + x1) * 3u;
            uint8_t* dp = dst_rgb + ((size_t)dy * img_w + dx) * 3u;
            for (int c = 0; c < 3; c++) {
                float t0 = p00[c] * (1.0f - wx) + p10[c] * wx;
                float t1 = p01[c] * (1.0f - wx) + p11[c] * wx;
                float pv = t0 * (1.0f - wy) + t1 * wy;
                float bv = (float)dp[c] * (1.0f - alpha) + pv * alpha;
                int iv = (int)(bv + 0.5f);
                dp[c] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
            }
        }
    }
}
