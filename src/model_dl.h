#ifndef MODEL_DL_H
#define MODEL_DL_H

#include <stddef.h>
#include <stdint.h>

#define MODEL_DL_MAX_CATALOG 32
#define MODEL_DL_MAX_INSTALLED 32
#define LORA_MAX_CATALOG 48
#define LORA_MAX_INSTALLED 64

#define GITHUB_MODELS_URL  "https://raw.githubusercontent.com/DJTOMATO/GarlicDiffusion/main/catalog/models.json"
#define GITHUB_LORAS_URL   "https://raw.githubusercontent.com/DJTOMATO/GarlicDiffusion/main/catalog/loras.json"
#define GITHUB_VERSION_URL "https://raw.githubusercontent.com/DJTOMATO/GarlicDiffusion/main/catalog/version.json"
#define DEFAULT_OTA_PKG_URL "https://github.com/DJTOMATO/GarlicDiffusion/releases/latest/download/Garlic-Diffusion.pkg"
#define OTA_PKG_DEST       "/data/pkg/Garlic-Diffusion.pkg"
#define GD_APP_VERSION     "1.16"

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
    int queue_len;      // pending queued LoRA downloads waiting behind current file
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

// ---- LoRA Manager & GitHub-Synced Catalog (/data/garlic-diffusion/loras/*.safetensors) ----------
typedef struct {
    char name[64];          // stem without .safetensors
    char filename[80];      // exact filename in /data/garlic-diffusion/loras/
    char arch[12];          // "sd15", "sdxl", or "any"
    uint32_t size_mb;
    uint32_t rank;
    int has_te;
} InstalledLora;

typedef struct {
    char id[64];
    char title[80];
    char arch[12];          // "sd15" or "sdxl"
    uint32_t size_mb;
    float default_weight;
    char trigger[96];
    char desc[160];
    char filename[80];
    char url[256];
} CatalogLora;

void lora_dl_init(const char* loras_root);
int lora_list_installed(InstalledLora* out, int max_items);
int lora_delete(const char* name_or_file);
int lora_get_catalog(CatalogLora* out, int max_items);
int lora_refresh_catalog(void);
int lora_dl_start(const char* id_or_name, const char* url_override, const char* filename_override);
// Queue all uninstalled catalog LoRAs matching `arch_filter` ("sdxl", "sd15", or ""/"all" for all).
// Returns number of LoRAs queued/started (>=0), or <0 on error.
int lora_dl_queue_all(const char* arch_filter);
int lora_dl_is_queued(const char* id_or_filename);

// ---- GitHub OTA Update Checker & /data/pkg Downloader -------------------------------------------
typedef struct {
    int checked;
    int update_available;
    int pkg_ready;
    char current_ver[16];
    char remote_ver[16];
    char pkg_url[256];
    char changelog[256];
    uint32_t size_mb;
} AppUpdateStatus;

int app_version_is_newer(const char* cur_ver, const char* rem_ver);
int app_update_check_now(void);
void app_update_get_status(AppUpdateStatus* out);
int app_update_download_pkg(void);
int app_update_prepare_and_exit(char* out_msg, size_t max_msg);

#endif
