#ifndef HTTP_SRV_H
#define HTTP_SRV_H

#include <stdint.h>
#include <stddef.h>

#define DIFF_HTTP_PORT 8081
#define DIFF_MAX_MODELS 32
#define DIFF_MAX_BATCH_GAL 128
#define DIFF_MAX_BATCH_GEN 100

typedef struct {
    char prompt[1024];
    char negative[512];
    char lora[80];
    float lora_scale;
    uint32_t width;
    uint32_t height;
    uint32_t steps;
    float cfg;
    uint64_t seed;
    int eco_mode;     // 0=Turbo(36CU), 1=Balanced(25CU), 2=Quiet(20CU), 3=Silent(14CU)
    int api_eco_mode; // Power mode for A1111/OpenAI API & Discord bot calls: -1=Follow Studio, 0..3 (default 3=Silent)
    int sampler;    // 0=Euler a, 1=Euler, 2=LCM, 3=DPM++ 2M Karras
    int upscale;    // 0=Off, 1=1080p Anime (RealESRGAN), 2=2x Anime HD, 3=4K UHD (up to 3840x2160)
    uint32_t batch_count;     // 1..100 sequential images with fresh random seeds
    uint32_t batch_rand_mask; // RAND_POSE | RAND_EXPR | RAND_SCENE | ... bitmask
    char batch_folder[64];    // optional gallery folder for generated images
    int nsfw_allow;           // 0 = SFW guard ON (default), 1 = NSFW allowed (unrestricted)
    int auto_quality;         // 1 = Auto-Quality & Hand/Eye Shield ON (default)
    int auto_face_fix;        // 1 = Automatic ADetailer Face Fix (img2img crop) on full-body/SD1.5 shots (default ON)
    int api_face_fix;         // 0 = OFF by default for SDXL/Illustrious API calls unless requested (ADetailer/auto_face_fix), 1 = ON
    int ui_mode;              // -1 = Unset (show 2-choice welcome), 0 = Easy Mode, 1 = Advanced Mode
    int shutdown_after;       // 1 = cleanly power off PS4 after batch finishes
    int pending;
    int cancel_req;
    char switch_model[64];
    int switch_req;
    int rescan_req;
    int gal_op;     // 0=none, 1=upscale 1080p, 2=upscale 2x, 3=upscale 4K, 4=resize, 5=fix_face
    char gal_id[64];
    char gal_ids[DIFF_MAX_BATCH_GAL][64];
    int gal_ids_n;
    uint32_t gal_w, gal_h;
    float gal_denoise;
    int presets_dirty;
    int styles_dirty;
    int chars_dirty;
    int loras_dirty;
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
    int api_eco_mode;
    uint32_t gpu_used_mb;
    uint32_t gpu_total_mb;
    float progress;
    char stage[128];
    char model[64];
    char models[DIFF_MAX_MODELS][64];
    int num_models;
    // Active model auto-preset metadata (Best Quality + Speed Mode)
    uint32_t preset_w, preset_h, preset_steps;
    float preset_cfg;
    int preset_sampler, preset_upscale;
    char preset_desc[96];
    char preset_prompt[512];
    char preset_negative[384];
    uint32_t speed_w, speed_h, speed_steps;
    float speed_cfg;
    int speed_sampler, speed_upscale;
    char speed_desc[96];
    char error[256];
    uint32_t width, height, steps;
    float cfg;
    uint64_t seed;
    double last_sec;
    double unet_gpu_sec;
    double vae_sec;
    double upscale_sec;
    double face_fix_sec;
    double cooldown_sec;
    double eff_tflops;
    double bench_gemm_tflops;
    double bench_attn_ms;
    double bench_mode_tf[4];
    uint8_t* png_data;
    size_t png_size;
    uint32_t image_gen_id;
    uint8_t* preview_png;
    size_t preview_png_size;
    uint32_t preview_gen_id;
    uint32_t batch_cur;
    uint32_t batch_total;
    int shutdown_after;
    char last_gallery_id[64];
    char active_lora[128];
    uint32_t gallery_rev;
    float eta_sec;
    float elapsed_sec;
    // Live / in-progress Studio & Job settings snapshot (for reload & multi-device sync)
    uint32_t settings_seq;
    char cur_prompt[1024];
    char cur_negative[512];
    char cur_lora[80];
    float cur_lora_scale;
    uint32_t cur_w, cur_h, cur_steps, cur_batch, cur_rand_mask;
    float cur_cfg;
    uint64_t cur_seed;
    int cur_sampler, cur_upscale, cur_eco;
    int nsfw_allow, auto_quality, auto_face_fix, api_face_fix, ui_mode;
    char cur_folder[64];
} DiffStatus;

int http_srv_start(DiffJob* shared_job, DiffStatus* shared_status);
const char* http_srv_address(void);
void http_srv_lock(void);
void http_srv_unlock(void);

#endif
