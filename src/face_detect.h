#ifndef FACE_DETECT_H
#define FACE_DETECT_H

#include <stdint.h>

typedef struct {
    int x, y, w, h;
    float score;
} FaceBox;

// Pure-C CPU Anime Face Detector (zero external dependencies; runs in ~15-40ms on PS4 Jaguar CPU).
// Uses multi-scale integral-image skin-region + bilateral eye-darkness + hair-crown contrast scoring
// tailored specifically for anime characters (pale/warm/cool anime skin + twin dark eye bands).
// Writes up to `max_boxes` sorted by confidence descending; returns number of faces detected (0..max_boxes).
int face_detect_anime(const uint8_t* rgb, uint32_t img_w, uint32_t img_h,
                      FaceBox* out_boxes, int max_boxes);

// Expand a detected face box with context padding (e.g. 1.75x), square-align it, and clamp to image bounds.
void face_box_expand_square(const FaceBox* in, uint32_t img_w, uint32_t img_h,
                            float pad_scale, FaceBox* out);

// Crop `box` from `src_rgb` and bilinear-resize to `dst_w x dst_h` (newly allocated buffer).
uint8_t* face_crop_resize(const uint8_t* src_rgb, uint32_t img_w, uint32_t img_h,
                          const FaceBox* box, uint32_t dst_w, uint32_t dst_h);

// Resize refined face `patch_rgb` (`patch_w x patch_h`) back to `box` and feather-blend into `dst_rgb` in-place.
void face_blend_back(uint8_t* dst_rgb, uint32_t img_w, uint32_t img_h,
                     const uint8_t* patch_rgb, uint32_t patch_w, uint32_t patch_h,
                     const FaceBox* box, float feather_frac);

#endif
