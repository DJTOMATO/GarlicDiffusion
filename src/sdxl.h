#ifndef SDXL_H
#define SDXL_H

#include <stdint.h>
#include <stddef.h>
#include "clip_tok.h"

typedef void (*SdxlProgressFn)(void* user, float frac, const char* msg);
typedef int (*SdxlCancelFn)(void* user);
typedef void (*SdxlPreviewFn)(void* user, const uint8_t* rgb, uint32_t w, uint32_t h);

typedef struct {
    char name[96];
    uint64_t off;
    uint64_t size;
    uint64_t scales;
    uint32_t rows;
    uint32_t cols;
    uint32_t fmt;   // 0 = F32, 1 = F16, 2 = Q8_0
    uint32_t _pad;
} SdxlTensor;

typedef struct {
    uint32_t n;
    uint64_t data_size;
    float vae_scale;
    SdxlTensor* t;
    uint64_t gpu_base;
} SdxlPack;

typedef struct {
    char name[32];
    uint64_t addr;
    uint32_t rsrc1, rsrc2, user_sgprs, kernarg_sgpr;
} SdxlKernel;

typedef struct {
    double clip_l_ms;
    double clip_g_ms;
    double unet_total_ms;
    double unet_gpu_ms;
    double vae_ms;
    double vae_gpu_ms;
    double upscale_ms;
    double cooldown_ms;
    double total_sec;
    double total_gflops;
    double effective_tflops;
    double bench_gemm_tflops;
    double bench_attn_ms;
    double bench_mode_tf[4]; // 0=Turbo, 1=Balanced, 2=Quiet, 3=Ultra-Cool
    int vae_tiles;
    int temp_start_c;
    int temp_peak_c;
    int temp_end_c;
    uint32_t width, height, steps;
    float cfg;
} SdxlStats;

typedef struct {
    int neo;
    uint64_t tsc_freq;
    uint8_t* mem;
    int64_t mem_phys_off;
    uint64_t mem_size;
    uint64_t next;
    uint64_t scratch_bytes;
    uint64_t weights_bytes;

    // Command buffer (PS4 GPU mode)
    uint32_t* dcb;
    uint32_t dcb_cap, ndw;
    uint8_t* kargs;
    uint32_t kargs_cap, kargs_used;
    volatile uint32_t* label;
    volatile uint64_t* stamps;
    uint32_t init_state[256];
    int dispatch_count;
    double gpu_acc_sec;
    double flops_acc;

    // Thermal & cancel hooks: 0=Turbo(36CU), 1=Balanced(~26CU), 2=Quiet(~18CU), 3=Ultra-Cool(~14CU)
    int eco_mode;
    SdxlCancelFn cancel_fn;
    void* cancel_user;

    // Kernels
    SdxlKernel k[28];
    int nk;
    const SdxlKernel *k_add, *k_copy, *k_scale_add, *k_gemm_f32;
    const SdxlKernel *k_gemm_q8_64, *k_gemm_q8_128x64, *k_gemm_q8_128;
    int best_gemm; // 0 = 64x64, 1 = 128x64, 2 = 128x128
    const SdxlKernel *k_im2col, *k_gn, *k_ln, *k_add_temb, *k_concat;
    const SdxlKernel *k_up2x, *k_geglu, *k_act, *k_clip_emb, *k_attn64, *k_attn40, *k_attn80, *k_attn160, *k_attn512, *k_prelu;
    const SdxlKernel *k_lora_q8, *k_lora_f32;

    // Architecture flag: 0 = SDXL (CLIP-L + OpenCLIP-G + 2.6B UNet), 1 = SD 1.5 (CLIP-L + 860M UNet)
    int is_sd15;

    // Model packs + built-in RealESRGAN Anime v3 upscaler (2.4 MiB)
    SdxlPack clip_l, clip_g, unet, vae, upscaler;
    int has_upscaler;
    ClipTok tok;

    // Scratch buffers in GPU memory (compact on Base PS4 so SDXL fits in ~4.0 GiB!)
    uint64_t buf_main_bytes;
    uint64_t buf_im2col_bytes;
    uint64_t buf_skips_bytes;
    uint32_t vae_tile;
    uint64_t buf_a, buf_b, buf_c;
    uint64_t buf_q, buf_k, buf_v;
    uint64_t buf_im2col;
    uint64_t skip_pool;
    uint64_t skip_ptr[12];
    uint32_t skip_c[12];
    int n_skips;

    // Conditioning & small buffers
    uint64_t ctx_cond, ctx_uncond;
    uint32_t cur_ctx_len;
    uint64_t pool_cond, pool_uncond;
    uint64_t tmp_cl, tmp_cg;
    uint64_t tok_gpu;
    uint64_t t_raw, temb, temb_silu, tproj, add_in, aug_emb;
    uint64_t lat_in, eps_u, eps_c;

    double last_gen_sec;
    // Live runtime telemetry for real-time ETA calculation
    uint32_t rt_phase;        // 0=idle, 1=clip, 2=unet, 3=vae, 4=upscale, 5=load
    uint32_t rt_step_done;
    uint32_t rt_steps_total;
    uint32_t rt_tiles_done;
    uint32_t rt_tiles_total;
    double   rt_step_avg_sec;
    double   rt_tile_avg_sec;

    SdxlStats stats;
    SdxlPreviewFn preview_fn;
    void* preview_user;
    char model_name[64];
    char model_dir[256];
    char applied_lora_spec[384];
    char error[256];
} SdxlEngine;

typedef struct {
    const char* prompt;
    const char* negative;
    uint32_t width;
    uint32_t height;
    uint32_t steps;
    float cfg_scale;
    uint64_t seed;
    int eco_mode;        // 0 = Turbo (36CU), 1 = Balanced (~25CU), 2 = Quiet (~20CU), 3 = Silent (~14CU)
    int sampler;         // 0 = Euler a, 1 = Euler, 2 = LCM, 3 = DPM++ 2M Karras
} SdxlParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t steps;
    float cfg;
    int sampler;         // 0 = Euler a, 1 = Euler, 2 = LCM, 3 = DPM++ 2M Karras
    int upscale;         // 0 = Off, 1 = 1080p Anime (RealESRGAN), 2 = 2x Anime HD
    char description[96];
    char prompt[512];
    char negative[384];
} ModelPreset;

const char* sdxl_eco_name(int eco_mode);
const char* sdxl_eco_name_ex(int eco_mode, int neo);
void sdxl_load_preset(const char* model_dir, const char* model_name, int is_sd15, ModelPreset* out);
void sdxl_load_speed_preset(const char* model_name, int is_sd15, ModelPreset* out);
void sdxl_clean_prompt_dupes(const char* src, char* dst, size_t dst_max);
void sdxl_sanitize_nsfw_prompts(int nsfw_allow, char* prompt, size_t pr_max, char* negative, size_t ng_max);
void sdxl_expand_wildcards(const char* src, uint64_t seed, int nsfw_allow, uint32_t rand_mask,
                           char* dst, size_t dst_max);
void sdxl_apply_quality_shield(const char* model_name, int is_sd15, int auto_quality, int nsfw_allow,
                               char* prompt, size_t pr_max, char* negative, size_t ng_max);
int sdxl_read_temp_c(int* cpu_c, int* soc_c);
int sdxl_check_pack(const char* path, uint64_t* actual_out, uint64_t* expected_out);
int sdxl_init(SdxlEngine* e, const char* assets_dir, const char* model_dir,
              SdxlProgressFn progress, void* user);
// Returns 0 on success, -2 if pre-flight failed (previous loaded model kept intact), -1 if load failed after unloading.
int sdxl_reload_model(SdxlEngine* e, const char* model_dir,
                      SdxlProgressFn progress, void* user);
void sdxl_free(SdxlEngine* e);

// Synchronize active Kohya/Diffusers .safetensors LoRAs in `loras_dir` with `lora_spec`
// (comma-separated "file_or_stem:weight", or "" for none). Reloads clean base UNet/CLIP weights
// automatically when changing/clearing previously fused LoRAs; 0-cost no-op when unchanged.
int sdxl_sync_loras(SdxlEngine* e, const char* loras_dir, const char* lora_spec,
                    SdxlProgressFn progress, void* user);

// Extract `<lora:name:weight>` tags from `raw_prompt` into `clean_prompt` (stripped for CLIP)
// and merge with optional `ui_lora` + `ui_strength` into canonical `out_spec` ("name:0.80,...").
void sdxl_extract_loras(const char* raw_prompt, const char* ui_lora, float ui_strength,
                        char* clean_prompt, size_t clean_max, char* out_spec, size_t spec_max);

int sdxl_generate(SdxlEngine* e, const SdxlParams* p, uint8_t* rgb_out,
                  SdxlProgressFn progress, SdxlCancelFn cancel, void* user);
int sdxl_img2img(SdxlEngine* e, const SdxlParams* p, const uint8_t* init_rgb, float denoise,
                 uint8_t* rgb_out, SdxlProgressFn progress, SdxlCancelFn cancel, void* user);
int sdxl_has_vae_encoder(const SdxlEngine* e);
int sdxl_upscale_rgb(SdxlEngine* e, const uint8_t* src_rgb, uint32_t src_w, uint32_t src_h,
                     uint8_t** dst_rgb_out, uint32_t* dst_w_out, uint32_t* dst_h_out,
                     int upscale_mode, int eco_mode,
                     SdxlProgressFn progress, SdxlCancelFn cancel, void* user);

#endif
