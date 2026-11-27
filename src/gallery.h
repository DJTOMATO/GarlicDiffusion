#ifndef GALLERY_H
#define GALLERY_H

#include <stddef.h>
#include <stdint.h>

#define GALLERY_MAX_ITEMS 2048
#define GALLERY_TV_MAX 512
#define GALLERY_MAX_FOLDERS 64
#define PRESETS_MAX_ITEMS 256
#define STYLES_MAX_ITEMS 64
#define CHARS_MAX_ITEMS 400

typedef struct {
    char name[64];
    int count;
} GalleryFolder;

typedef struct {
    char id[64];
    char name[64];
    char prompt[512];
    char negative[384];
} StyleItem;

typedef struct {
    char id[64];
    char name[64];
    char series[48];
    char prompt[512];
    char negative[384];
} CharacterItem;

typedef struct {
    char id[64];
    char model[64];
    char prompt[1024];
    char negative[512];
    char sampler[16];
    uint32_t width;
    uint32_t height;
    uint32_t steps;
    float cfg;
    uint64_t seed;
    int eco;
    int upscaled;
    double sec;
    double unet_sec;
    double vae_sec;
    double upscale_sec;
    double face_fix_sec;
    double cooldown_sec;
    double eff_tflops;
    size_t bytes;
    char lora[128];
    char folder[64];
    int starred;
} GalleryItem;

typedef struct {
    char id[64];
    char name[64];
    char model[64];
    char prompt[1024];
    char negative[512];
    char sampler[16];
    uint32_t width;
    uint32_t height;
    uint32_t steps;
    float cfg;
    int upscale;
    int eco;
    int nsfw;
    int is_speed;
} UserPreset;

void gallery_init(const char* out_dir, const char* presets_dir);
int gallery_valid_id(const char* id);
uint32_t gallery_revision(void);

// Save PNG + metadata sidecar (+ optional fast RGB thumbnail); writes assigned id into out_id (optional).
int gallery_save(const uint8_t* png, size_t png_size, const GalleryItem* meta, char* out_id, size_t out_id_max);
int gallery_save_ex(const uint8_t* png, size_t png_size, const uint8_t* rgb_opt, uint32_t rgb_w, uint32_t rgb_h,
                    const GalleryItem* meta, char* out_id, size_t out_id_max);

// Bounded RAM-first image staging buffer (tmpfs-style) with background disk flush & safe fallback
enum {
    GALLERY_STAGE_IDLE = 0,
    GALLERY_STAGE_STAGING = 1,
    GALLERY_STAGE_FLUSHING = 2,
    GALLERY_STAGE_COMPLETE = 3,
    GALLERY_STAGE_ERROR = 4
};

typedef struct {
    int state;              // GALLERY_STAGE_*
    uint32_t queued_count;  // items currently staged/flushing in RAM
    uint32_t flushed_total; // total items persisted to disk since startup
    uint32_t used_bytes;    // current RAM staging bytes used
    uint32_t max_bytes;     // configured hard RAM staging cap
    int fallback_used;      // 1 if last save fell back directly to disk
    char last_id[64];
    char error[128];
} GalleryStageStatus;

void gallery_stage_init(size_t max_bytes);
int gallery_stage_enqueue(const uint8_t* png, size_t png_size, const uint8_t* rgb_opt, uint32_t rgb_w, uint32_t rgb_h,
                          const GalleryItem* meta, char* out_id, size_t out_id_max);
void gallery_stage_flush_all(void);
void gallery_stage_get_status(GalleryStageStatus* out);
int gallery_list(GalleryItem* items, int max_items);
int gallery_list_page(GalleryItem* items, int offset, int limit, const char* query, int* out_total);
int gallery_list_page_ex(GalleryItem* items, int offset, int limit, const char* query,
                         const char* folder_filter, int starred_only, int* out_total);
int gallery_get(const char* id, GalleryItem* out);
uint8_t* gallery_load_png(const char* id, size_t* out_size);
uint8_t* gallery_load_thumb_png(const char* id, size_t* out_size);
int gallery_delete(const char* id);

// Gallery Folders, Starred Favorites & Stored-ZIP Archive Builder
int gallery_clean_folder_name(const char* raw, char* out, size_t max);
int gallery_create_folder(const char* name, char* clean_out, size_t max);
int gallery_delete_folder(const char* name, int delete_images);
int gallery_list_folders(GalleryFolder* out, int max_folders, int* out_unsorted, int* out_starred, int* out_total);
int gallery_set_folder(const char* id, const char* folder);
int gallery_set_folder_batch(const char (*ids)[64], int n_ids, const char* folder);
int gallery_set_starred(const char* id, int starred);
uint8_t* gallery_build_zip(const char (*ids)[64], int n_ids, const char* folder_filter, size_t* out_zip_size);

// User Preset Manager
int presets_list(UserPreset* items, int max_items);
int presets_save(const UserPreset* p, char* out_id, size_t out_id_max);
int presets_delete(const char* id);

// Reusable Art Styles Manager (saved to /data/garlic-diffusion/styles/)
void styles_init(const char* styles_dir);
int styles_list(StyleItem* items, int max_items);
int styles_save(const StyleItem* s, char* out_id, size_t out_id_max);
int styles_delete(const char* id);

// Character Presets Manager (saved to /data/garlic-diffusion/characters/)
void characters_init(const char* chars_dir);
int characters_list(CharacterItem* items, int max_items);
int characters_save(const CharacterItem* c, char* out_id, size_t out_id_max);
int characters_delete(const char* id);

// Per-model default preset.cfg writer
int model_preset_save(const char* models_root, const char* model_name,
                      int is_sd15, uint32_t width, uint32_t height, uint32_t steps,
                      float cfg, const char* sampler, int upscale,
                      const char* description, const char* prompt, const char* negative);

#endif
