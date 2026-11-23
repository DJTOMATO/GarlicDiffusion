#ifndef GALLERY_H
#define GALLERY_H

#include <stddef.h>
#include <stdint.h>

#define GALLERY_MAX_ITEMS 128
#define PRESETS_MAX_ITEMS 96
#define STYLES_MAX_ITEMS 64
#define CHARS_MAX_ITEMS 96

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
    size_t bytes;
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
} UserPreset;

void gallery_init(const char* out_dir, const char* presets_dir);
int gallery_valid_id(const char* id);

// Save PNG + metadata sidecar; writes assigned id into out_id (optional).
int gallery_save(const uint8_t* png, size_t png_size, const GalleryItem* meta, char* out_id, size_t out_id_max);
int gallery_list(GalleryItem* items, int max_items);
int gallery_get(const char* id, GalleryItem* out);
uint8_t* gallery_load_png(const char* id, size_t* out_size);
int gallery_delete(const char* id);

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
