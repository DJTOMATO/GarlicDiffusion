#ifndef HTTP_SRV_H
#define HTTP_SRV_H

#include <stdint.h>
#include <stddef.h>

#define DIFF_HTTP_PORT 8081
#define DIFF_MAX_MODELS 16

typedef struct {
    char prompt[1024];
    char negative[512];
    uint32_t width;
    uint32_t height;
    uint32_t steps;
    float cfg;
    uint64_t seed;
    int eco_mode;   // 0=Turbo(36CU), 1=Balanced(25CU), 2=Quiet(20CU), 3=Silent(14CU)
    int sampler;    // 0=Euler a, 1=Euler, 2=LCM
    int upscale;    // 0=Off, 1=1080p Anime (RealESRGAN), 2=2x Anime HD
    int pending;
    int cancel_req;
    char switch_model[64];
    int switch_req;
    int rescan_req;
    int gal_op;     // 0=none, 1=upscale 1080p, 2=upscale 2x, 3=resize
    char gal_id[64];
    uint32_t gal_w, gal_h;
    int presets_dirty;
} DiffJob;

typedef struct {
    int gpu_ready;  // 1 once GPU + kernels are initialized (model switching allowed even if ready==0)
    int ready;      // 1 once a complete model is loaded (generation allowed)
    int busy;
    int neo;
    int is_sd15;
    int has_upscaler;
    int temp_c;
    int temp_peak_c;
    int eco_mode;
    uint32_t gpu_used_mb;
    uint32_t gpu_total_mb;
    float progress;
    char stage[128];
    char model[64];
    char models[DIFF_MAX_MODELS][64];
    int num_models;
    // Active model auto-preset metadata
    uint32_t preset_w, preset_h, preset_steps;
    float preset_cfg;
    int preset_sampler, preset_upscale;
    char preset_desc[96];
    char preset_prompt[512];
    char preset_negative[384];
    char error[256];
    uint32_t width, height, steps;
    float cfg;
    uint64_t seed;
    double last_sec;
    double unet_gpu_sec;
    double vae_sec;
    double upscale_sec;
    double cooldown_sec;
    double eff_tflops;
    double bench_gemm_tflops;
    double bench_attn_ms;
    double bench_mode_tf[4];
    uint8_t* png_data;
    size_t png_size;
    uint32_t image_gen_id;
    char last_gallery_id[64];
} DiffStatus;

int http_srv_start(DiffJob* shared_job, DiffStatus* shared_status);
const char* http_srv_address(void);
void http_srv_lock(void);
void http_srv_unlock(void);

#endif
