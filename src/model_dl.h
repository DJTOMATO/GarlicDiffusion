#ifndef MODEL_DL_H
#define MODEL_DL_H

#include <stddef.h>
#include <stdint.h>

#define MODEL_DL_MAX_CATALOG 24
#define MODEL_DL_MAX_INSTALLED 24

typedef struct {
    char id[64];
    char title[80];
    char arch[12];      // "sd15" or "sdxl"
    uint32_t size_mb;
    char desc[144];
    char files[256];    // comma-separated relative filenames (e.g. "unet.psd1,clip_l.psd1,vae.psd1,preset.cfg")
    char repo[128];     // optional override HF repo; empty = use configured default repo
} CatalogModel;

typedef struct {
    char name[64];
    char arch[12];      // "sd15" or "sdxl"
    uint32_t size_mb;
    int complete;       // 1 if unet.psd1 (and local files) pass PSD1 completeness check
    char detail[96];
} InstalledModel;

typedef struct {
    int active;
    int cancel_req;
    int finished;       // increments each time a download finishes successfully
    char model_id[64];
    char repo[128];
    char cur_file[64];
    int file_idx;
    int file_total;
    uint64_t bytes_done;
    uint64_t bytes_total;
    double speed_mbs;
    char stage[160];
    char error[256];
} ModelDlStatus;

void model_dl_init(const char* models_root, const char* cfg_path);
void model_dl_get_repo(char* out, size_t max);
int model_dl_set_repo(const char* repo);

// Inspect installed models on disk (size, arch, completeness)
int model_dl_list_installed(InstalledModel* out, int max_items);
int model_dl_delete(const char* model_id, const char* active_model);

// Catalog (built-in curated list, optionally refreshed from HF manifest.json)
int model_dl_get_catalog(CatalogModel* out, int max_items);
int model_dl_refresh_catalog(void);

// Start background download of a catalog entry OR custom repo/folder/files
int model_dl_start(const char* model_id, const char* repo_override, const char* files_override);
void model_dl_cancel(void);
void model_dl_get_status(ModelDlStatus* out);

#endif
