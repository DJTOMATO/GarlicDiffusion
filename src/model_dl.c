// Garlic Diffusion: In-App Model Manager & Hugging Face HTTPS Downloader
// Downloads converted PSD1 models straight to /data/garlic-diffusion/models/ using PS4 libSceHttp + libSceSsl.

#include "model_dl.h"
#include "gallery.h"
#include "sdxl.h"

#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <direct.h>
#define mkdir_p(p) _mkdir(p)
#define rmdir_p(p) _rmdir(p)
#else
#include <unistd.h>
#include <orbis/AppInstUtil.h>
#include <orbis/Http.h>
#include <orbis/Net.h>
#include <orbis/Ssl.h>
#include <orbis/Sysmodule.h>
#include <orbis/SystemService.h>
#include <orbis/libkernel.h>
#define mkdir_p(p) mkdir(p, 0777)
#define rmdir_p(p) rmdir(p)
void glog(const char* fmt, ...);
#endif

#ifdef SDXL_CPU
static void glog(const char* fmt, ...) { (void)fmt; }
#endif

static char g_models_root[256] = "/data/garlic-diffusion/models";
static char g_loras_root[256]  = "/data/garlic-diffusion/loras";
static char g_repo_cfg[256] = "/data/garlic-diffusion/repo.cfg";
static char g_models_cache_json[280] = "/data/garlic-diffusion/models_catalog.json";
static char g_loras_cache_json[280]  = "/data/garlic-diffusion/loras_catalog.json";
static char g_hf_repo[128] = "DJTomato/ps4-diffusion-models"; // Default public Hugging Face model catalog

static CatalogModel g_catalog[MODEL_DL_MAX_CATALOG];
static int g_catalog_count = 0;
static CatalogLora g_lora_catalog[LORA_MAX_CATALOG];
static int g_lora_catalog_count = 0;
static ModelDlStatus g_dl;
static AppUpdateStatus g_upd;

static int check_ota_pkg_ready_unlocked(void) {
#ifdef _WIN32
    int fd = open(OTA_PKG_DEST, O_RDONLY | O_BINARY);
    int64_t fsz = (fd >= 0) ? _filelengthi64(fd) : -1;
#else
    int fd = open(OTA_PKG_DEST, O_RDONLY);
    off_t fsz = (fd >= 0) ? lseek(fd, 0, SEEK_END) : -1;
#endif
    if (fd >= 0) close(fd);
    return (fsz > (1 << 20)) ? 1 : 0;
}

#ifndef SDXL_CPU
static OrbisPthreadMutex g_dl_mu;
static int g_mu_ok = 0;
static int g_http_ready = 0;
static int g_net_pool = -1, g_ssl_ctx = -1, g_http_ctx = -1;
static void dl_lock(void) { if (g_mu_ok) scePthreadMutexLock(&g_dl_mu); }
static void dl_unlock(void) { if (g_mu_ok) scePthreadMutexUnlock(&g_dl_mu); }
#else
static void dl_lock(void) {}
static void dl_unlock(void) {}
#endif

static void ensure_dir_dl(const char* path) {
    char tmp[320];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char c = *p; *p = 0;
            mkdir_p(tmp);
            *p = c;
        }
    }
    mkdir_p(tmp);
}

static void seed_builtin_catalog(void) {
    static const CatalogModel BUILTIN[] = {
        { "vae_enc_sdxl", "VAE Encoder Add-on (SDXL Fix Face / img2img)", "sdxl", 130,
          "Recommended add-on · Enables full AI img2img Face Fix for all SDXL / Illustrious models",
          "vae_enc_sdxl.psd1", "" },
        { "vae_enc_sd15", "VAE Encoder Add-on (SD 1.5 Fix Face / img2img)", "sd15", 130,
          "Recommended add-on · Enables full AI img2img Face Fix for all SD 1.5 models",
          "vae_enc_sd15.psd1", "" },
        { "WAI_Illustrious_v17_dmd2", "WAI-Illustrious v17 DMD2 (Fast SDXL)", "sdxl", 3716,
          "Latest WAI v17 anime · 6 steps CFG 1.4 LCM (~50s Pro) · Standalone SDXL",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "waiNSFWIllustrious_v140_dmd2", "WAI-Illustrious v14 DMD2 (Fast SDXL)", "sdxl", 3716,
          "Sharp glowing anime · 4 steps CFG 1.0 LCM · Standalone SDXL",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "Nova_Anime_XL_v18_dmd2", "Nova Anime XL v18 DMD2 (Fast SDXL)", "sdxl", 3716,
          "Vibrant 2D/2.5D Illustrious anime · Jewel eyes & clean hands · 6 steps (~50s Pro)",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "Hassaku_XL_v31_dmd2", "Hassaku XL Illustrious v3.1 DMD2 (Fast SDXL)", "sdxl", 3716,
          "Crisp broadcast anime · Clean fingers & expressions · 6 steps (~50s Pro)",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "Illustrious_Pencil_v32_dmd2", "Illustrious Pencil XL v3.2 DMD2 (Fast SDXL)", "sdxl", 3716,
          "Fine-line artbook illustration · Sharp hand & pose anatomy · 6 steps (~50s Pro)",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "Plant_Milk_Walnut_dmd2", "Plant Milk Walnut DMD2 (Fast SDXL)", "sdxl", 3716,
          "Painterly gouache & dark-fantasy anime · Expressive eyes · 6 steps (~50s Pro)",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "Prefect_Illustrious_XL_v2_dmd2", "Prefect Illustrious XL v2 DMD2 (Fast SDXL)", "sdxl", 3716,
          "Razor-sharp modern Illustrious anime · Crisp eyes, anatomy & cel lighting · 6 steps (~50s Pro)",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "Obsession_Illustrious_v4_dmd2", "Obsession Illustrious v4 DMD2 (Fast SDXL)", "sdxl", 3716,
          "Lush glossy anime lighting & expressive jewel eyes · Illustrious DMD2 · 6 steps (~50s Pro)",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "waiNSFWIllustrious_v140_8step", "WAI-Illustrious v14 8-Step (Hyper-SD)", "sdxl", 3716,
          "Euler CFG 3.5 · Negative prompt active · Standalone 8-step SDXL",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "waiNSFWIllustrious_v140", "WAI-Illustrious v14 Full (20-Step SDXL)", "sdxl", 3716,
          "Full 20-step reference SDXL checkpoint · Standalone SDXL",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "MeinaMix_V11_8step", "MeinaMix V11 8-Step (Ultra-Fast SD 1.5)", "sd15", 1220,
          "Classic glossy anime · ~25s + 1080p upscale · Ideal for Base PS4!",
          "unet.psd1,clip_l.psd1,vae.psd1,preset.cfg", "" },
        { "MeinaMix_V11", "MeinaMix V11 (20-Step SD 1.5)", "sd15", 1220,
          "Full quality SD 1.5 anime · 1.5 GiB VRAM · Auto 1080p RealESRGAN",
          "unet.psd1,clip_l.psd1,vae.psd1,preset.cfg", "" },
        { "GhostMix_V2", "GhostMix v2.0 (Luminous Anime SD 1.5)", "sd15", 1220,
          "Detailed luminous anime · Sharp eyes · Auto 1080p RealESRGAN",
          "unet.psd1,clip_l.psd1,vae.psd1,preset.cfg", "" },
        { "Anything_V5", "Anything V5 (Classic 2D Anime SD 1.5)", "sd15", 1031,
          "Crisp classic 2D anime · 1.5 GiB VRAM · Auto 1080p RealESRGAN",
          "unet.psd1,clip_l.psd1,vae.psd1,preset.cfg", "" },
    };
    g_catalog_count = (int)(sizeof(BUILTIN) / sizeof(BUILTIN[0]));
    if (g_catalog_count > MODEL_DL_MAX_CATALOG) g_catalog_count = MODEL_DL_MAX_CATALOG;
    memcpy(g_catalog, BUILTIN, (size_t)g_catalog_count * sizeof(CatalogModel));
}

static void seed_builtin_lora_catalog(void) {
    static const CatalogLora BUILTIN_LORAS[] = {
        { "trickcal_revive_style_v1.0", "Trickcal Revive/트릭컬 리바이브/穿越异世界恶作剧 Style v1.0", "sdxl", 72, 0.70f, "chibi, blush stickers, long hair",
          "SDXL Trickcal Revive/트릭컬 리바이브/穿越异世界恶作剧 Style v1.0 · DPM++ 2M Karras · CFG 6.0",
          "Trickcal_Revive_Style_v1.0.safetensors",
          "https://civitai.com/models/169973/trickcal-revive-style" },
        { "trickcal_re_vive_character_style_v1.0", "Trickcal RE:VIVE Character Style v1.0", "sdxl", 97, 0.70f, "all",
          "SDXL Trickcal RE:VIVE Character Style v1.0 · DPM++ 2M Karras · CFG 6.0",
          "Trickcal_RE_VIVE_Character_Style_v1.0.safetensors",
          "https://civitai.com/models/1216680/trickcal-revive-character-style" },
        { "eyes-for-illustrious", "Perfect Anime Eyes (Illustrious XL)", "sdxl", 218, 0.75f, "",
          "Sharpens iris catchlights, eyelashes & pupil symmetry on Illustrious / Nova / Hassaku / WAI SDXL",
          "eyes-for-illustrious.safetensors",
          "https://huggingface.co/Muapi/eyes-for-illustrious-lora-perfect-anime-eyes/resolve/main/eyes-for-illustrious-lora-perfect-anime-eyes.safetensors" },
        { "detail-tweaker-illustrious", "Detail Tweaker XL (Illustrious)", "sdxl", 218, 0.70f, "",
          "Boosts lineart crispness, hair strands, fabric folds & fine detail (0.5..1.0 = more detail)",
          "detail-tweaker-illustrious.safetensors",
          "https://huggingface.co/Muapi/detail-tweaker-illustrious-by-stable-yogi/resolve/main/detail-tweaker-illustrious-by-stable-yogi.safetensors" },
        { "better-hands-xl", "Better Hands Anatomy Helper (SDXL)", "sdxl", 110, 0.65f, "",
          "Improves finger separation, wrist joints & hand gestures on SDXL / Illustrious checkpoints",
          "better-hands-xl.safetensors",
          "https://huggingface.co/Muapi/better-hands/resolve/main/better-hands.safetensors" },
        { "fine-anime-screencap-xl", "Fine Anime Screencap Style (Illustrious XL)", "sdxl", 165, 0.80f, "anime coloring",
          "Clean TV broadcast anime screencap shading & cinematic color grading",
          "fine-anime-screencap-xl.safetensors",
          "https://huggingface.co/Muapi/fine-anime-screencap-xl-anime-screencap-style-lora-illustrious-and-ponyxl/resolve/main/fine-anime-screencap-xl-anime-screencap-style-lora-illustrious-and-ponyxl.safetensors" },
        { "add-detail-xl", "Add Detail XL (Universal SDXL)", "sdxl", 218, 0.70f, "",
          "Universal SDXL detail enhancer slider (+0.6..1.0 adds micro-detail, -0.5 simplifies)",
          "add-detail-xl.safetensors",
          "https://huggingface.co/LyliaEngine/add-detail-xl/resolve/main/add-detail-xl.safetensors" },
        { "add-detail-sd15", "Detail Tweaker (SD 1.5)", "sd15", 36, 0.75f, "",
          "Classic SD 1.5 detail & sharpness enhancer for MeinaMix, GhostMix & DreamShaper",
          "add_detail_sd15.safetensors",
          "https://huggingface.co/OedoSoldier/detail-tweaker-lora/resolve/main/add_detail.safetensors" },
        { "detail-slider-illustrious", "Detail Slider Mini (Illustrious XL)", "sdxl", 9, 0.70f, "",
          "Ultra-lightweight 8.4 MiB Illustrious detail slider (+0.5..1.2 sharper lineart, -0.5 cleaner)",
          "detail-slider-illustrious.safetensors",
          "https://huggingface.co/Muapi/detail-slider-lora-illustrious-xl/resolve/main/detail-slider-lora-illustrious-xl.safetensors" },
        { "detailer-concept-illustrious", "Detailer Concept (Illustrious XL)", "sdxl", 218, 0.65f, "",
          "High-impact Illustrious XL detailer for eyes, hair strands, fabric folds & lighting depth",
          "detailer-concept-illustrious.safetensors",
          "https://huggingface.co/Muapi/detailer-tool-concept-lora-illustriousxl/resolve/main/detailer-tool-concept-lora-illustriousxl.safetensors" },
        { "add-micro-details-xl", "Add Micro Details (Illustrious / Pony XL)", "sdxl", 218, 0.65f, "",
          "Adds intricate micro-details to hair, lace, jewelry & backgrounds without shifting style",
          "add-micro-details-xl.safetensors",
          "https://huggingface.co/AiAF/Add_Micro_Details_-_Concept_Illustrious__v1.0_Pony_lora/resolve/main/AddMicroDetails_pony_v1.safetensors" },
        { "theobrobine-anime-style-xl", "Theobrobine Anime Style (Illustrious XL)", "sdxl", 218, 0.75f, "",
          "Lush modern anime illustration style with rich shading & expressive highlights",
          "theobrobine-anime-style-xl.safetensors",
          "https://huggingface.co/Muapi/theobrobine-illustrious-anima-style-lora/resolve/main/theobrobine-illustrious-anima-style-lora.safetensors" },
        { "soraoraora-style-xl", "Soraoraora Glossy Style (Illustrious XL)", "sdxl", 229, 0.75f, "",
          "Vibrant glossy Pixiv anime coloring & crisp rim lighting for Illustrious XL",
          "soraoraora-style-xl.safetensors",
          "https://huggingface.co/Muapi/soraoraora-style-lora-for-ponyxl-illustrious/resolve/main/soraoraora-style-lora-for-ponyxl-illustrious.safetensors" },
        { "shaded-retro-anime-xl", "Shaded Retro Anime (SDXL)", "sdxl", 163, 0.75f, "",
          "Classic cel-shaded retro anime look with dramatic shadows & crisp ink lines",
          "shaded-retro-anime-xl.safetensors",
          "https://huggingface.co/nDimensional/Shaded-Retro-Style-Anime-with-Natural-Language-SDXL-LoRA/resolve/main/Shaded-Retro-Anime-000002.safetensors" },
        { "vintage-90s-anime-xl", "Vintage 1990s Anime Cel (SDXL / Illustrious)", "sdxl", 218, 0.75f, "",
          "Authentic 1990s hand-painted anime OVA film aesthetic & soft halation",
          "vintage-90s-anime-xl.safetensors",
          "https://huggingface.co/Muapi/vintage-1990s-anime-sdxl-lora-style-pony/resolve/main/vintage-1990s-anime-sdxl-lora-style-pony.safetensors" },
        { "anime-slider-sdxl", "Anime Style Slider Mini (SDXL)", "sdxl", 9, 0.80f, "",
          "Tiny 8.4 MiB universal SDXL anime slider — boosts 2D anime stylization with 0 VRAM cost",
          "anime-slider-sdxl.safetensors",
          "https://huggingface.co/ntc-ai/SDXL-LoRA-slider.anime/resolve/main/anime.safetensors" },
        { "retro-90s-slider-sdxl", "90s Anime Slider Mini (SDXL)", "sdxl", 9, 0.80f, "",
          "Tiny 8.4 MiB 90s retro anime slider — quick nostalgic cel shading on any SDXL checkpoint",
          "retro-90s-slider-sdxl.safetensors",
          "https://huggingface.co/ntc-ai/SDXL-LoRA-slider.90s-anime/resolve/main/90s%20anime.safetensors" },
        { "pixel-art-illustrious", "Pixel Art Style (Illustrious XL)", "sdxl", 218, 0.80f, "pixel art",
          "Clean 16-bit / 32-bit anime sprite & pixel-art illustration style for Illustrious XL",
          "pixel-art-illustrious.safetensors",
          "https://huggingface.co/Muapi/pixel-art-style-illustrious-by-skormino/resolve/main/pixel-art-style-illustrious-by-skormino.safetensors" },
        { "anime-detailer-xl", "Anime Detailer XL (Universal SDXL)", "sdxl", 23, 0.70f, "",
          "Lightweight 23 MiB anime lineart, eye & shading detailer by Linaqruf — works on Animagine & Illustrious",
          "anime-detailer-xl.safetensors",
          "https://huggingface.co/Linaqruf/anime-detailer-xl-lora/resolve/main/anime-detailer-xl.safetensors" },
        { "ghibli-slider-sdxl", "Studio Ghibli Style Slider (SDXL)", "sdxl", 9, 0.80f, "",
          "Tiny 8.4 MiB Studio Ghibli painterly anime slider for any SDXL / Illustrious checkpoint",
          "ghibli-slider-sdxl.safetensors",
          "https://huggingface.co/ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style/resolve/main/Studio%20Ghibli%20style.safetensors" },
        { "watercolor-slider-sdxl", "Watercolor Anime Slider (SDXL)", "sdxl", 9, 0.80f, "",
          "Tiny 8.4 MiB soft watercolor illustration slider for SDXL / Illustrious",
          "watercolor-slider-sdxl.safetensors",
          "https://huggingface.co/ntc-ai/SDXL-LoRA-slider.watercolor/resolve/main/watercolor.safetensors" },
        { "cinematic-light-slider-sdxl", "Cinematic Lighting Slider (SDXL)", "sdxl", 9, 0.75f, "",
          "Tiny 8.4 MiB dramatic cinematic rim & volumetric lighting slider for SDXL",
          "cinematic-light-slider-sdxl.safetensors",
          "https://huggingface.co/ntc-ai/SDXL-LoRA-slider.cinematic-lighting/resolve/main/cinematic%20lighting.safetensors" },
        { "storybook-anime-sd15", "Storybook Anime Style (SD 1.5)", "sd15", 72, 0.75f, "",
          "Soft hand-painted storybook anime backgrounds & cozy cel-shaded lighting for SD 1.5",
          "storybook-anime-sd15.safetensors",
          "https://huggingface.co/neonforestmist/sd15-storybook-anime-lora/resolve/main/storybook_anime_lora.safetensors" },
        { "animate-scenery-sd15", "Cinematic Anime Scenery (SD 1.5)", "sd15", 72, 0.70f, "",
          "Atmospheric Makoto-style skies, clouds & cinematic anime lighting for SD 1.5",
          "animate-scenery-sd15.safetensors",
          "https://huggingface.co/J-YOON/animate-lora-sd1.5/resolve/main/animate_v1-000005.safetensors" },
    };
    g_lora_catalog_count = (int)(sizeof(BUILTIN_LORAS) / sizeof(BUILTIN_LORAS[0]));
    if (g_lora_catalog_count > LORA_MAX_CATALOG) g_lora_catalog_count = LORA_MAX_CATALOG;
    memcpy(g_lora_catalog, BUILTIN_LORAS, (size_t)g_lora_catalog_count * sizeof(CatalogLora));
}

static int apply_manifest_json(const char* json);
static int apply_lora_manifest_json(const char* json);

static void load_cached_json_file(const char* path, int is_lora) {
#ifdef _WIN32
    int fd = open(path, O_RDONLY | O_BINARY);
#else
    int fd = open(path, O_RDONLY);
#endif
    if (fd < 0) return;
    char* buf = (char*)malloc(65536);
    if (buf) {
        ssize_t n = read(fd, buf, 65535);
        if (n > 2) {
            buf[n] = 0;
            if (is_lora) apply_lora_manifest_json(buf);
            else apply_manifest_json(buf);
        }
        free(buf);
    }
    close(fd);
}

static void save_cached_json_file(const char* path, const char* json) {
    if (!path || !json || !json[0]) return;
    char dir[280];
    snprintf(dir, sizeof(dir), "%s", path);
    char* sl = strrchr(dir, '/');
    if (sl) { *sl = 0; ensure_dir_dl(dir); }
#ifdef _WIN32
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0666);
#else
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
#endif
    if (fd >= 0) {
        size_t l = strlen(json);
        if (l > 0) (void)write(fd, json, (unsigned)l);
        close(fd);
    }
}

#ifndef SDXL_CPU
static void* startup_catalog_sync_fn(void* arg) {
    (void)arg;
    int nm = model_dl_refresh_catalog();
    int nl = lora_refresh_catalog();
    int uok = (app_update_check_now() == 0);
    glog("catalog auto-sync (GitHub + fallback): models=%d, loras=%d, ota_checked=%d",
         nm > 0 ? nm : g_catalog_count, nl > 0 ? nl : g_lora_catalog_count, uok);
    return NULL;
}
#endif

void model_dl_init(const char* models_root, const char* cfg_path) {
#ifndef SDXL_CPU
    if (!g_mu_ok) {
        scePthreadMutexInit(&g_dl_mu, NULL, "gdiff_dl");
        g_mu_ok = 1;
    }
#endif
    if (models_root && models_root[0]) snprintf(g_models_root, sizeof(g_models_root), "%s", models_root);
    if (cfg_path && cfg_path[0]) {
        snprintf(g_repo_cfg, sizeof(g_repo_cfg), "%s", cfg_path);
        char base[256];
        snprintf(base, sizeof(base), "%s", cfg_path);
        char* sl = strrchr(base, '/');
#ifdef _WIN32
        char* bsl = strrchr(base, '\\');
        if (bsl && (!sl || bsl > sl)) sl = bsl;
#endif
        if (sl) {
            *sl = 0;
            snprintf(g_models_cache_json, sizeof(g_models_cache_json), "%s/models_catalog.json", base);
            snprintf(g_loras_cache_json, sizeof(g_loras_cache_json), "%s/loras_catalog.json", base);
        }
    }
    memset(&g_dl, 0, sizeof(g_dl));
    memset(&g_upd, 0, sizeof(g_upd));
    snprintf(g_upd.current_ver, sizeof(g_upd.current_ver), "%s", GD_APP_VERSION);
    snprintf(g_upd.pkg_url, sizeof(g_upd.pkg_url), "%s", DEFAULT_OTA_PKG_URL);
    g_upd.size_mb = 8;
    g_upd.pkg_ready = check_ota_pkg_ready_unlocked();
    seed_builtin_catalog();
    seed_builtin_lora_catalog();
    load_cached_json_file(g_models_cache_json, 0);
    load_cached_json_file(g_loras_cache_json, 1);

    // Load saved Hugging Face repo ID if present
    int fd = open(g_repo_cfg, O_RDONLY);
    if (fd >= 0) {
        char buf[256];
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        close(fd);
        if (n > 0) {
            buf[n] = 0;
            char* p = strstr(buf, "repo=");
            if (p) p += 5; else p = buf;
            while (*p == ' ' || *p == '\t') p++;
            size_t j = 0;
            while (p[j] && p[j] != '\r' && p[j] != '\n' && j + 1 < sizeof(g_hf_repo)) j++;
            memcpy(g_hf_repo, p, j);
            g_hf_repo[j] = 0;
        }
    }
#ifndef SDXL_CPU
    // Non-blocking background sync from GitHub raw catalog (falls back cleanly to local if offline)
    OrbisPthread th;
    OrbisPthreadAttr attr;
    scePthreadAttrInit(&attr);
    scePthreadAttrSetstacksize(&attr, 256 * 1024);
    scePthreadAttrSetdetachstate(&attr, 1);
    if (scePthreadCreate(&th, &attr, startup_catalog_sync_fn, NULL, "gdiff_cat") != 0) {
        // Ignore; built-in local catalog remains active
    }
    scePthreadAttrDestroy(&attr);
#endif
}

void lora_dl_init(const char* loras_root) {
    if (loras_root && loras_root[0]) snprintf(g_loras_root, sizeof(g_loras_root), "%s", loras_root);
    ensure_dir_dl(g_loras_root);
    if (g_lora_catalog_count == 0) seed_builtin_lora_catalog();
}

void model_dl_get_repo(char* out, size_t max) {
    dl_lock();
    snprintf(out, max, "%s", g_hf_repo);
    dl_unlock();
}

int model_dl_set_repo(const char* repo) {
    if (!repo) return -1;
    while (*repo == ' ' || *repo == '\t') repo++;
    // Strip optional https://huggingface.co/ prefix so user can paste either full URL or owner/repo!
    if (strncmp(repo, "https://huggingface.co/", 23) == 0) repo += 23;
    else if (strncmp(repo, "http://huggingface.co/", 22) == 0) repo += 22;
    char clean[128];
    size_t j = 0;
    for (; *repo && j + 1 < sizeof(clean); repo++) {
        if (*repo == '\r' || *repo == '\n' || *repo == ' ') break;
        clean[j++] = *repo;
    }
    while (j > 0 && clean[j - 1] == '/') j--;
    clean[j] = 0;

    dl_lock();
    snprintf(g_hf_repo, sizeof(g_hf_repo), "%s", clean);
    dl_unlock();

    char dir[256];
    snprintf(dir, sizeof(dir), "%s", g_repo_cfg);
    char* sl = strrchr(dir, '/');
    if (sl) { *sl = 0; ensure_dir_dl(dir); }
    int fd = open(g_repo_cfg, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd >= 0) {
        char line[160];
        int n = snprintf(line, sizeof(line), "repo=%s\n", clean);
        if (n > 0) (void)write(fd, line, (unsigned)n);
        close(fd);
    }
    return 0;
}

static int cmp_inst(const void* a, const void* b) {
    return strcmp(((const InstalledModel*)a)->name, ((const InstalledModel*)b)->name);
}

int model_dl_list_installed(InstalledModel* out, int max_items) {
    DIR* d = opendir(g_models_root);
    if (!d) return 0;
    int n = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL && n < max_items) {
        if (ent->d_name[0] == '.') continue;
        char mdir[320];
        snprintf(mdir, sizeof(mdir), "%s/%s", g_models_root, ent->d_name);
        DIR* sub = opendir(mdir);
        if (!sub) continue;

        InstalledModel* im = &out[n];
        memset(im, 0, sizeof(*im));
        snprintf(im->name, sizeof(im->name), "%s", ent->d_name);
        uint64_t total_bytes = 0;
        int has_unet = 0, bad_file = 0;
        char bad_desc[96] = {0};

        struct dirent* fe;
        while ((fe = readdir(sub)) != NULL) {
            if (fe->d_name[0] == '.') continue;
            char fp[384];
            snprintf(fp, sizeof(fp), "%s/%s", mdir, fe->d_name);
#ifdef _WIN32
            int fd = open(fp, O_RDONLY | O_BINARY);
            int64_t fsz = (fd >= 0) ? _filelengthi64(fd) : -1;
#else
            int fd = open(fp, O_RDONLY);
            off_t fsz = (fd >= 0) ? lseek(fd, 0, SEEK_END) : -1;
#endif
            if (fd >= 0) close(fd);
            if (fsz > 0) total_bytes += (uint64_t)fsz;

            size_t fl = strlen(fe->d_name);
            if (fl > 5 && strcmp(fe->d_name + fl - 5, ".psd1") == 0) {
                if (strcmp(fe->d_name, "unet.psd1") == 0) has_unet = 1;
                uint64_t act = 0, exp = 0;
                int ck = sdxl_check_pack(fp, &act, &exp);
                if (ck != 0 && !bad_file) {
                    bad_file = 1;
                    if (ck == -3) snprintf(bad_desc, sizeof(bad_desc), "Incomplete %s (%llu/%llu MiB)",
                                           fe->d_name, (unsigned long long)(act >> 20), (unsigned long long)(exp >> 20));
                    else snprintf(bad_desc, sizeof(bad_desc), "Corrupt %s", fe->d_name);
                }
            }
        }
        closedir(sub);
        if (!has_unet) continue;

        // Detect arch from unet.psd1 size (<1500 MiB = SD 1.5, >=1500 MiB = SDXL)
        char up[384];
        snprintf(up, sizeof(up), "%s/unet.psd1", mdir);
        uint64_t uact = 0, uexp = 0;
        sdxl_check_pack(up, &uact, &uexp);
        snprintf(im->arch, sizeof(im->arch), "%s", (uexp > 0 && uexp < (1500ull << 20)) ? "sd15" : "sdxl");
        im->size_mb = (uint32_t)(total_bytes >> 20);
        im->complete = !bad_file;
        snprintf(im->detail, sizeof(im->detail), "%s", bad_file ? bad_desc : "Complete");
        n++;
    }
    closedir(d);
    if (n > 1) qsort(out, (size_t)n, sizeof(InstalledModel), cmp_inst);
    return n;
}

int model_dl_delete(const char* model_id, const char* active_model) {
    if (!gallery_valid_id(model_id)) return -1;
    if (active_model && strcmp(model_id, active_model) == 0) return -2; // cannot delete active model
    char mdir[320];
    snprintf(mdir, sizeof(mdir), "%s/%s", g_models_root, model_id);
    DIR* d = opendir(mdir);
    if (!d) return -1;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        char fp[384];
        snprintf(fp, sizeof(fp), "%s/%s", mdir, ent->d_name);
        unlink(fp);
    }
    closedir(d);
    return rmdir_p(mdir);
}

int model_dl_get_catalog(CatalogModel* out, int max_items) {
    dl_lock();
    int n = g_catalog_count < max_items ? g_catalog_count : max_items;
    if (n > 0) memcpy(out, g_catalog, (size_t)n * sizeof(CatalogModel));
    dl_unlock();
    return n;
}

void model_dl_get_status(ModelDlStatus* out) {
    dl_lock();
    *out = g_dl;
    dl_unlock();
}

typedef struct {
    char id[64];
    char filename[80];
    char url[384];
} LoraDlTask;

#define LORA_DL_QMAX 64
static LoraDlTask g_lora_q[LORA_DL_QMAX];
static int g_lora_q_count = 0;
static int g_lora_dl_is_lora = 0;

void model_dl_cancel(void) {
    dl_lock();
    g_lora_q_count = 0;
    g_dl.queue_len = 0;
    if (g_dl.active) {
        g_dl.cancel_req = 1;
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Cancelling download...");
    }
    dl_unlock();
}

// Parse a lightweight Hugging Face manifest.json array of model objects:
// [{"id":"...","title":"...","arch":"sdxl","size_mb":3530,"desc":"...","files":"unet.psd1,clip_l.psd1,..."}]
static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static int parse_manifest_str(const char* obj, const char* end, const char* key, char* out, size_t max) {
    char pat[48];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(obj, pat);
    if (!p || p >= end) return 0;
    p += strlen(pat);
    while (p < end && (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (p >= end || *p != '"') return 0;
    p++;
    size_t n = 0;
    while (p < end && *p && *p != '"' && n + 1 < max) {
        if (*p == '\\' && p + 1 < end) {
            p++;
            if (*p == 'u' && p + 4 < end) {
                int h0 = hex_val(p[1]), h1 = hex_val(p[2]), h2 = hex_val(p[3]), h3 = hex_val(p[4]);
                if (h0 >= 0 && h1 >= 0 && h2 >= 0 && h3 >= 0) {
                    unsigned cp = (unsigned)((h0 << 12) | (h1 << 8) | (h2 << 4) | h3);
                    p += 5;
                    if (cp < 0x80u) {
                        out[n++] = (char)cp;
                    } else if (cp < 0x800u && n + 2 < max) {
                        out[n++] = (char)(0xC0u | (cp >> 6));
                        out[n++] = (char)(0x80u | (cp & 0x3Fu));
                    } else if (n + 3 < max) {
                        out[n++] = (char)(0xE0u | (cp >> 12));
                        out[n++] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
                        out[n++] = (char)(0x80u | (cp & 0x3Fu));
                    }
                    continue;
                }
            }
            if (*p == 'n') out[n++] = '\n';
            else if (*p == 't') out[n++] = '\t';
            else out[n++] = *p;
            p++;
        } else {
            out[n++] = *p++;
        }
    }
    out[n] = 0;
    return 1;
}

static uint32_t parse_manifest_u32(const char* obj, const char* end, const char* key, uint32_t def) {
    char pat[48];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(obj, pat);
    if (!p || p >= end) return def;
    p += strlen(pat);
    while (p < end && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    if (p >= end || !isdigit((unsigned char)*p)) return def;
    return (uint32_t)strtoul(p, NULL, 10);
}

static float parse_manifest_f32(const char* obj, const char* end, const char* key, float def) {
    char pat[48];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(obj, pat);
    if (!p || p >= end) return def;
    p += strlen(pat);
    while (p < end && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    if (p >= end || (*p != '-' && !isdigit((unsigned char)*p))) return def;
    return (float)strtod(p, NULL);
}

static int apply_lora_manifest_json(const char* json) {
    CatalogLora parsed[LORA_MAX_CATALOG];
    int n = 0;
    const char* p = json;
    while (p && *p && n < LORA_MAX_CATALOG) {
        const char* ob = strchr(p, '{');
        if (!ob) break;
        const char* cb = strchr(ob + 1, '}');
        if (!cb) break;
        CatalogLora l;
        memset(&l, 0, sizeof(l));
        if (parse_manifest_str(ob, cb, "id", l.id, sizeof(l.id)) && gallery_valid_id(l.id) &&
            parse_manifest_str(ob, cb, "url", l.url, sizeof(l.url)) && l.url[0]) {
            if (!parse_manifest_str(ob, cb, "title", l.title, sizeof(l.title)))
                snprintf(l.title, sizeof(l.title), "%s", l.id);
            if (!parse_manifest_str(ob, cb, "arch", l.arch, sizeof(l.arch)))
                snprintf(l.arch, sizeof(l.arch), "sdxl");
            l.size_mb = parse_manifest_u32(ob, cb, "size_mb", 0);
            l.default_weight = parse_manifest_f32(ob, cb, "default_weight", 0.80f);
            parse_manifest_str(ob, cb, "trigger", l.trigger, sizeof(l.trigger));
            parse_manifest_str(ob, cb, "desc", l.desc, sizeof(l.desc));
            if (!parse_manifest_str(ob, cb, "filename", l.filename, sizeof(l.filename)) || !l.filename[0])
                snprintf(l.filename, sizeof(l.filename), "%s.safetensors", l.id);
            parsed[n++] = l;
        }
        p = cb + 1;
    }
    if (n > 0) {
        dl_lock();
        g_lora_catalog_count = n;
        memcpy(g_lora_catalog, parsed, (size_t)n * sizeof(CatalogLora));
        dl_unlock();
    }
    return n;
}

static int cmp_lora_inst(const void* a, const void* b) {
    return strcmp(((const InstalledLora*)a)->name, ((const InstalledLora*)b)->name);
}

// Fast (<1 ms) .safetensors header probe for /data/garlic-diffusion/loras/*.safetensors
static void probe_lora_safetensors(const char* path, InstalledLora* out) {
    snprintf(out->arch, sizeof(out->arch), "sdxl");
#ifdef _WIN32
    int fd = open(path, O_RDONLY | O_BINARY);
    int64_t fsz = (fd >= 0) ? _filelengthi64(fd) : -1;
#else
    int fd = open(path, O_RDONLY);
    off_t fsz = (fd >= 0) ? lseek(fd, 0, SEEK_END) : -1;
#endif
    if (fd < 0) return;
    if (fsz > 0) out->size_mb = (uint32_t)(((uint64_t)fsz + (1ull << 19)) >> 20);
    if (out->size_mb == 0 && fsz > 0) out->size_mb = 1;
    lseek(fd, 0, SEEK_SET);
    uint64_t hlen = 0;
    if (read(fd, &hlen, 8) == 8 && hlen > 2 && hlen <= (8ull << 20)) {
        size_t rlen = hlen > 655360 ? 655360 : (size_t)hlen;
        char* hdr = (char*)malloc(rlen + 1);
        if (hdr) {
            ssize_t got = read(fd, hdr, (unsigned)rlen);
            if (got > 0) {
                hdr[got] = 0;
                int has_te12 = (strstr(hdr, "lora_te1_") != NULL || strstr(hdr, "lora_te2_") != NULL);
                int has_up3  = (strstr(hdr, "up_blocks_3_") != NULL || strstr(hdr, "up_blocks.3.") != NULL);
                int has_te   = has_te12 || (strstr(hdr, "lora_te_") != NULL);
                out->has_te = has_te;
                if (has_te12) snprintf(out->arch, sizeof(out->arch), "sdxl");
                else if (has_up3 || (strstr(hdr, "lora_te_") != NULL && !has_te12))
                    snprintf(out->arch, sizeof(out->arch), "sd15");
                const char* dm = strstr(hdr, "\"ss_network_dim\"");
                if (dm) {
                    const char* q1 = strchr(dm + 16, '"');
                    if (q1) out->rank = (uint32_t)strtoul(q1 + 1, NULL, 10);
                }
                if (out->rank == 0) {
                    const char* dn = strstr(hdr, ".lora_down.weight\"");
                    if (!dn) dn = strstr(hdr, ".lora.down.weight\"");
                    if (!dn) dn = strstr(hdr, ".lora_A.weight\"");
                    if (dn) {
                        const char* sp = strstr(dn, "\"shape\"");
                        const char* br = sp ? strchr(sp, '[') : NULL;
                        if (br && (size_t)(br - dn) < 160u) {
                            const char* d0 = br + 1;
                            while (*d0 == ' ') d0++;
                            uint32_t rk = (uint32_t)strtoul(d0, NULL, 10);
                            if (rk > 0 && rk <= 512) out->rank = rk;
                        }
                    }
                }
            }
            free(hdr);
        }
    }
    close(fd);
}

int lora_list_installed(InstalledLora* out, int max_items) {
    if (!out || max_items <= 0) return 0;
    ensure_dir_dl(g_loras_root);
    DIR* d = opendir(g_loras_root);
    if (!d) return 0;
    int n = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL && n < max_items) {
        const char* fn = ent->d_name;
        if (fn[0] == '.') continue;
        size_t l = strlen(fn);
        if (l <= 12 || strcmp(fn + l - 12, ".safetensors") != 0 || l >= 80) continue;
        InstalledLora* il = &out[n];
        memset(il, 0, sizeof(*il));
        snprintf(il->filename, sizeof(il->filename), "%s", fn);
        size_t sl = l - 12;
        if (sl >= sizeof(il->name)) sl = sizeof(il->name) - 1;
        memcpy(il->name, fn, sl);
        il->name[sl] = 0;
        char fp[360];
        snprintf(fp, sizeof(fp), "%s/%s", g_loras_root, fn);
        probe_lora_safetensors(fp, il);
        n++;
    }
    closedir(d);
    if (n > 1) qsort(out, (size_t)n, sizeof(InstalledLora), cmp_lora_inst);
    return n;
}

int lora_delete(const char* name_or_file) {
    if (!name_or_file || !name_or_file[0] || strchr(name_or_file, '/') ||
        strchr(name_or_file, '\\') || strstr(name_or_file, "..")) return -1;
    char fp[360];
    size_t l = strlen(name_or_file);
    if (l > 12 && strcmp(name_or_file + l - 12, ".safetensors") == 0)
        snprintf(fp, sizeof(fp), "%s/%s", g_loras_root, name_or_file);
    else
        snprintf(fp, sizeof(fp), "%s/%s.safetensors", g_loras_root, name_or_file);
    return unlink(fp);
}

int lora_get_catalog(CatalogLora* out, int max_items) {
    dl_lock();
    int n = g_lora_catalog_count < max_items ? g_lora_catalog_count : max_items;
    if (n > 0) memcpy(out, g_lora_catalog, (size_t)n * sizeof(CatalogLora));
    dl_unlock();
    return n;
}

static int apply_manifest_json(const char* json) {
    CatalogModel parsed[MODEL_DL_MAX_CATALOG];
    int n = 0;
    const char* p = json;
    while (p && *p && n < MODEL_DL_MAX_CATALOG) {
        const char* ob = strchr(p, '{');
        if (!ob) break;
        const char* cb = strchr(ob + 1, '}');
        if (!cb) break;
        CatalogModel m;
        memset(&m, 0, sizeof(m));
        if (parse_manifest_str(ob, cb, "id", m.id, sizeof(m.id)) && gallery_valid_id(m.id)) {
            if (!parse_manifest_str(ob, cb, "title", m.title, sizeof(m.title)))
                snprintf(m.title, sizeof(m.title), "%s", m.id);
            if (!parse_manifest_str(ob, cb, "arch", m.arch, sizeof(m.arch)))
                snprintf(m.arch, sizeof(m.arch), "sdxl");
            m.size_mb = parse_manifest_u32(ob, cb, "size_mb", 0);
            parse_manifest_str(ob, cb, "desc", m.desc, sizeof(m.desc));
            if (!parse_manifest_str(ob, cb, "files", m.files, sizeof(m.files)))
                snprintf(m.files, sizeof(m.files), "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg");
            parse_manifest_str(ob, cb, "repo", m.repo, sizeof(m.repo));
            parsed[n++] = m;
        }
        p = cb + 1;
    }
    if (n > 0) {
        dl_lock();
        g_catalog_count = n;
        memcpy(g_catalog, parsed, (size_t)n * sizeof(CatalogModel));
        dl_unlock();
    }
    return n;
}

#ifndef SDXL_CPU

static int32_t skip_ssl_cb(int32_t libsslId, uint32_t verifyErr, void* const sslCert[], int32_t certNum, void* userArg) {
    (void)libsslId; (void)verifyErr; (void)sslCert; (void)certNum; (void)userArg;
    return 0; // SCE_OK: accept modern TLS certs even on older PS4 firmware root CA stores
}

static int ensure_http_client(void) {
    if (g_http_ready) return 0;
    sceSysmoduleLoadModuleInternal(ORBIS_SYSMODULE_INTERNAL_NET);
    sceSysmoduleLoadModuleInternal(ORBIS_SYSMODULE_INTERNAL_HTTP);
    sceSysmoduleLoadModuleInternal(ORBIS_SYSMODULE_INTERNAL_SSL);
    sceNetInit();
    g_net_pool = sceNetPoolCreate("gdiff_dl", 32 * 1024, 0);
    if (g_net_pool < 0) { glog("model_dl: sceNetPoolCreate 0x%08x", (unsigned)g_net_pool); return -1; }
    g_ssl_ctx = sceSslInit(256 * 1024);
    if (g_ssl_ctx < 0) { glog("model_dl: sceSslInit 0x%08x", (unsigned)g_ssl_ctx); return -1; }
    g_http_ctx = sceHttpInit(g_net_pool, g_ssl_ctx, 256 * 1024);
    if (g_http_ctx < 0) { glog("model_dl: sceHttpInit 0x%08x", (unsigned)g_http_ctx); return -1; }
    g_http_ready = 1;
    glog("model_dl: PS4 HTTPS client ready");
    return 0;
}

static int find_location_hdr(const char* hdrs, size_t len, char* out, size_t max) {
    const char* p = hdrs;
    const char* end = hdrs + len;
    while (p < end) {
        const char* nl = memchr(p, '\n', (size_t)(end - p));
        const char* lend = nl ? nl : end;
        if ((size_t)(lend - p) > 9 && strncasecmp(p, "location:", 9) == 0) {
            const char* v = p + 9;
            while (v < lend && (*v == ' ' || *v == '\t')) v++;
            size_t n = 0;
            while (v < lend && *v != '\r' && *v != '\n' && n + 1 < max) out[n++] = *v++;
            out[n] = 0;
            return n > 0;
        }
        if (!nl) break;
        p = nl + 1;
    }
    return 0;
}

// Download a single URL over HTTPS (follows 3xx redirects up to 5 hops).
// If dst_path != NULL, streams to dst_path.part and atomically renames to dst_path.
// If dst_buf != NULL, reads into memory buffer (for manifest.json).
static int http_fetch_url(const char* initial_url, const char* dst_path, char* dst_buf, size_t dst_buf_max,
                          int track_progress, char* err_out, size_t err_max) {
    if (ensure_http_client() != 0) {
        snprintf(err_out, err_max, "PS4 HTTPS init failed");
        return -1;
    }
    char url[1024];
    snprintf(url, sizeof(url), "%s", initial_url);

    for (int hop = 0; hop < 6; hop++) {
        int tpl = sceHttpCreateTemplate(g_http_ctx, "GarlicDiffusion/1.7 (PS4)", ORBIS_HTTP_VERSION_1_1, 1);
        if (tpl < 0) { snprintf(err_out, err_max, "http template err 0x%08x", (unsigned)tpl); return -1; }
        sceHttpsDisableOption(tpl, 0x3Du);
        sceHttpsSetSslCallback(tpl, skip_ssl_cb, NULL);
        sceHttpSetConnectTimeOut(tpl, 15000000u);
        sceHttpSetResolveTimeOut(tpl, 10000000u);
        sceHttpSetSendTimeOut(tpl, 15000000u);

        int conn = sceHttpCreateConnectionWithURL(tpl, url, true);
        int req = (conn >= 0) ? sceHttpCreateRequestWithURL(conn, ORBIS_METHOD_GET, url, 0) : -1;
        int ret = (req >= 0) ? sceHttpSendRequest(req, NULL, 0) : -1;
        int32_t code = 0;
        if (ret >= 0) ret = sceHttpGetStatusCode(req, &code);

        if (ret < 0) {
            snprintf(err_out, err_max, "HTTP request failed (0x%08x)", (unsigned)ret);
            if (req >= 0) sceHttpDeleteRequest(req);
            if (conn >= 0) sceHttpDeleteConnection(conn);
            sceHttpDeleteTemplate(tpl);
            return -1;
        }
        if (code == 301 || code == 302 || code == 303 || code == 307 || code == 308) {
            char* hdrs = NULL;
            size_t hlen = 0;
            char next_url[1024] = {0};
            if (sceHttpGetAllResponseHeaders(req, &hdrs, &hlen) >= 0 && hdrs && hlen > 0) {
                find_location_hdr(hdrs, hlen, next_url, sizeof(next_url));
            }
            sceHttpDeleteRequest(req);
            sceHttpDeleteConnection(conn);
            sceHttpDeleteTemplate(tpl);
            if (!next_url[0]) { snprintf(err_out, err_max, "HTTP %d without Location", code); return -1; }
            if (next_url[0] == '/') {
                // Relative redirect on same host
                char origin[256] = {0};
                const char* s1 = strstr(url, "://");
                const char* slash = s1 ? strchr(s1 + 3, '/') : NULL;
                size_t olen = slash ? (size_t)(slash - url) : strlen(url);
                if (olen >= sizeof(origin)) olen = sizeof(origin) - 1;
                memcpy(origin, url, olen);
                snprintf(url, sizeof(url), "%s%s", origin, next_url);
            } else {
                snprintf(url, sizeof(url), "%s", next_url);
            }
            continue;
        }
        if (code != 200) {
            snprintf(err_out, err_max, "HTTP %d for %s", code, url);
            sceHttpDeleteRequest(req);
            sceHttpDeleteConnection(conn);
            sceHttpDeleteTemplate(tpl);
            return (code == 404) ? -404 : -1;
        }

        int32_t cl_type = 0;
        size_t cl_len = 0;
        if (sceHttpGetResponseContentLength(req, &cl_type, &cl_len) < 0 || cl_type != ORBIS_HTTP_CONTENTLEN_EXIST)
            cl_len = 0;

        if (track_progress) {
            dl_lock();
            g_dl.bytes_done = 0;
            g_dl.bytes_total = (uint64_t)cl_len;
            g_dl.speed_mbs = 0.0;
            dl_unlock();
        }

        int rc = 0;
        if (dst_path) {
            char part_path[384];
            snprintf(part_path, sizeof(part_path), "%s.part", dst_path);
            int fd = open(part_path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
            uint8_t* buf = (fd >= 0) ? (uint8_t*)malloc(128 * 1024) : NULL;
            if (fd < 0 || !buf) {
                if (fd >= 0) close(fd);
                free(buf);
                snprintf(err_out, err_max, "cannot create %s", part_path);
                rc = -1;
            } else {
                uint64_t done = 0;
                uint64_t t0 = sceKernelGetProcessTime();
                for (;;) {
                    dl_lock();
                    int can = g_dl.cancel_req;
                    dl_unlock();
                    if (can) { rc = 1; break; }

                    int r = sceHttpReadData(req, buf, 128 * 1024);
                    if (r < 0) { snprintf(err_out, err_max, "read error 0x%08x", (unsigned)r); rc = -1; break; }
                    if (r == 0) break;
                    ssize_t w = write(fd, buf, (size_t)r);
                    if (w != r) { snprintf(err_out, err_max, "disk write failed (PS4 storage full?)"); rc = -1; break; }
                    done += (uint64_t)r;
                    if (track_progress) {
                        double dt = (double)(sceKernelGetProcessTime() - t0) / 1e6;
                        dl_lock();
                        g_dl.bytes_done = done;
                        if (dt > 0.25) g_dl.speed_mbs = ((double)done / 1048576.0) / dt;
                        dl_unlock();
                    }
                }
                close(fd);
                free(buf);
                if (rc == 0) {
                    if (cl_len > 0 && done != (uint64_t)cl_len) {
                        unlink(part_path);
                        snprintf(err_out, err_max, "truncated download (%llu / %llu)",
                                 (unsigned long long)done, (unsigned long long)cl_len);
                        rc = -1;
                    } else {
                        unlink(dst_path);
                        rename(part_path, dst_path);
                    }
                } else {
                    unlink(part_path);
                }
            }
        } else if (dst_buf && dst_buf_max > 1) {
            size_t done = 0;
            while (done + 1 < dst_buf_max) {
                int r = sceHttpReadData(req, dst_buf + done, (uint32_t)(dst_buf_max - 1 - done));
                if (r <= 0) break;
                done += (size_t)r;
            }
            dst_buf[done] = 0;
        }

        sceHttpDeleteRequest(req);
        sceHttpDeleteConnection(conn);
        sceHttpDeleteTemplate(tpl);
        return rc;
    }
    snprintf(err_out, err_max, "too many HTTP redirects");
    return -1;
}

typedef struct {
    char model_id[64];
    char repo[128];
    char files[256];
} DlTask;

static void* dl_thread_fn(void* arg) {
    DlTask* t = (DlTask*)arg;
    char mdir[320];
    snprintf(mdir, sizeof(mdir), "%s/%s", g_models_root, t->model_id);
    ensure_dir_dl(mdir);

    // Split comma-separated file list
    char fnames[8][64];
    int nf = 0;
    const char* p = t->files;
    while (*p && nf < 8) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        size_t j = 0;
        while (*p && *p != ',' && *p != ' ' && j + 1 < 64) fnames[nf][j++] = *p++;
        fnames[nf][j] = 0;
        if (fnames[nf][0]) nf++;
    }

    dl_lock();
    g_dl.file_total = nf;
    dl_unlock();
    glog("model_dl: starting %s from %s (%d files)", t->model_id, t->repo, nf);

    int ok = 1;
    char err[256] = {0};
    for (int i = 0; i < nf; i++) {
        dl_lock();
        int can = g_dl.cancel_req;
        g_dl.file_idx = i + 1;
        snprintf(g_dl.cur_file, sizeof(g_dl.cur_file), "%s", fnames[i]);
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Downloading %s (%d/%d)...", fnames[i], i + 1, nf);
        g_dl.bytes_done = 0;
        g_dl.bytes_total = 0;
        dl_unlock();
        if (can) { snprintf(err, sizeof(err), "Download cancelled"); ok = 0; break; }

        char dst[384];
        snprintf(dst, sizeof(dst), "%s/%s", mdir, fnames[i]);
        // Skip already-complete .psd1 files so interrupted downloads resume cleanly!
        size_t fl = strlen(fnames[i]);
        if (fl > 5 && strcmp(fnames[i] + fl - 5, ".psd1") == 0 && sdxl_check_pack(dst, NULL, NULL) == 0) {
            glog("model_dl: skipping complete %s/%s", t->model_id, fnames[i]);
            continue;
        }

        char url[512];
        if (strncmp(t->repo, "http://", 7) == 0 || strncmp(t->repo, "https://", 8) == 0) {
            snprintf(url, sizeof(url), "%s/%s/%s", t->repo, t->model_id, fnames[i]);
        } else {
            snprintf(url, sizeof(url), "https://huggingface.co/%s/resolve/main/%s/%s",
                     t->repo, t->model_id, fnames[i]);
        }
        int rc = http_fetch_url(url, dst, NULL, 0, 1, err, sizeof(err));
        if (rc == -404 && strcmp(fnames[i], "unet.psd1") != 0) {
            // Optional shared file (e.g. vae.psd1 / clip_l.psd1 / clip_g.psd1 / preset.cfg) not in subfolder -> OK!
            glog("model_dl: optional file %s/%s not on server (404), will share from sibling", t->model_id, fnames[i]);
            err[0] = 0;
            continue;
        }
        if (rc == 1) { snprintf(err, sizeof(err), "Download cancelled"); ok = 0; break; }
        if (rc != 0) { ok = 0; break; }
        glog("model_dl: saved %s/%s", t->model_id, fnames[i]);
    }

    dl_lock();
    g_dl.active = 0;
    if (ok) {
        g_dl.finished++;
        g_dl.error[0] = 0;
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Installed %s! Ready to load.", t->model_id);
        glog("model_dl: completed %s!", t->model_id);
    } else {
        snprintf(g_dl.error, sizeof(g_dl.error), "%s", err[0] ? err : "Download failed");
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Download stopped");
        glog("model_dl: failed %s: %s", t->model_id, g_dl.error);
    }
    dl_unlock();
    free(t);
    return NULL;
}

int model_dl_refresh_catalog(void) {
    char* buf = (char*)malloc(65536);
    if (!buf) return -1;
    char err[160] = {0};
    int n = 0;

    // 1. Primary: GitHub raw catalog (DJTOMATO/GarlicDiffusion/main/catalog/models.json)
    buf[0] = 0;
    if (http_fetch_url(GITHUB_MODELS_URL, NULL, buf, 65536, 0, err, sizeof(err)) == 0) {
        n = apply_manifest_json(buf);
        if (n > 0) {
            save_cached_json_file(g_models_cache_json, buf);
            glog("model_dl: synced %d models from GitHub catalog", n);
        }
    } else {
        glog("model_dl: GitHub models catalog unavailable (%s), checking HF/local fallback", err);
    }

    // 2. Custom Hugging Face repo override (or secondary source if GitHub catalog not pushed yet)
    char repo[128];
    model_dl_get_repo(repo, sizeof(repo));
    if (repo[0] && (n == 0 || strcmp(repo, "DJTomato/ps4-diffusion-models") != 0)) {
        char url[512];
        if (strncmp(repo, "http://", 7) == 0 || strncmp(repo, "https://", 8) == 0)
            snprintf(url, sizeof(url), "%s/manifest.json", repo);
        else
            snprintf(url, sizeof(url), "https://huggingface.co/%s/resolve/main/manifest.json", repo);
        buf[0] = 0;
        if (http_fetch_url(url, NULL, buf, 65536, 0, err, sizeof(err)) == 0) {
            int nhf = apply_manifest_json(buf);
            if (nhf > 0) {
                n = nhf;
                save_cached_json_file(g_models_cache_json, buf);
            }
        }
    }
    free(buf);
    return n > 0 ? n : g_catalog_count;
}

int lora_refresh_catalog(void) {
    char* buf = (char*)malloc(65536);
    if (!buf) return -1;
    char err[160] = {0};
    int n = 0;
    buf[0] = 0;
    if (http_fetch_url(GITHUB_LORAS_URL, NULL, buf, 65536, 0, err, sizeof(err)) == 0) {
        n = apply_lora_manifest_json(buf);
        if (n > 0) {
            save_cached_json_file(g_loras_cache_json, buf);
            glog("lora_dl: synced %d LoRAs from GitHub catalog", n);
        }
    } else {
        glog("lora_dl: GitHub LoRAs catalog unavailable (%s), keeping local fallback (%d LoRAs)", err, g_lora_catalog_count);
    }
    free(buf);
    return n > 0 ? n : g_lora_catalog_count;
}

static int resolve_lora_task_locked(const char* id_or_name, const char* url_override,
                                    const char* filename_override, LoraDlTask* t) {
    memset(t, 0, sizeof(*t));
    const CatalogLora* cl = NULL;
    if (id_or_name && id_or_name[0]) {
        for (int i = 0; i < g_lora_catalog_count; i++) {
            if (strcmp(g_lora_catalog[i].id, id_or_name) == 0) { cl = &g_lora_catalog[i]; break; }
        }
    }
    snprintf(t->id, sizeof(t->id), "%s", (id_or_name && id_or_name[0]) ? id_or_name : "custom_lora");
    if (url_override && url_override[0]) {
        if (strncmp(url_override, "http://", 7) == 0 || strncmp(url_override, "https://", 8) == 0) {
            snprintf(t->url, sizeof(t->url), "%s", url_override);
        } else {
            const char* s1 = strchr(url_override, '/');
            const char* s2 = s1 ? strchr(s1 + 1, '/') : NULL;
            if (s2) {
                size_t rl = (size_t)(s2 - url_override);
                snprintf(t->url, sizeof(t->url), "https://huggingface.co/%.*s/resolve/main/%s",
                         (int)rl, url_override, s2 + 1);
            }
        }
    } else if (cl) {
        snprintf(t->url, sizeof(t->url), "%s", cl->url);
    }

    if (filename_override && filename_override[0]) {
        size_t fl = strlen(filename_override);
        if (fl > 12 && strcmp(filename_override + fl - 12, ".safetensors") == 0)
            snprintf(t->filename, sizeof(t->filename), "%s", filename_override);
        else
            snprintf(t->filename, sizeof(t->filename), "%s.safetensors", filename_override);
    } else if (cl && cl->filename[0]) {
        snprintf(t->filename, sizeof(t->filename), "%s", cl->filename);
    } else if (t->url[0]) {
        const char* sl = strrchr(t->url, '/');
        const char* base = sl ? sl + 1 : t->url;
        size_t bl = 0;
        while (base[bl] && base[bl] != '?' && bl + 1 < sizeof(t->filename)) {
            t->filename[bl] = base[bl];
            bl++;
        }
        t->filename[bl] = 0;
        if (bl <= 12 || strcmp(t->filename + bl - 12, ".safetensors") != 0)
            snprintf(t->filename, sizeof(t->filename), "%s.safetensors", t->id);
    }

    if (!t->url[0] || !t->filename[0] || strchr(t->filename, '/') || strchr(t->filename, '\\') || strstr(t->filename, "..")) {
        return -3;
    }
    return 0;
}

int lora_dl_is_queued(const char* id_or_filename) {
    if (!id_or_filename || !id_or_filename[0]) return 0;
    int found = 0;
    dl_lock();
    if (g_dl.active && g_lora_dl_is_lora &&
        (strcmp(g_dl.cur_file, id_or_filename) == 0 || strcmp(g_dl.model_id, id_or_filename) == 0)) {
        found = 1;
    }
    for (int i = 0; !found && i < g_lora_q_count; i++) {
        if (strcmp(g_lora_q[i].id, id_or_filename) == 0 || strcmp(g_lora_q[i].filename, id_or_filename) == 0) {
            found = 1;
        }
    }
    dl_unlock();
    return found;
}

static void* lora_dl_thread_fn(void* arg) {
    LoraDlTask cur = *(LoraDlTask*)arg;
    free(arg);
    ensure_dir_dl(g_loras_root);
    int done_ok = 0, total_batch = 1;

    for (;;) {
        char dst[384], err[256] = {0};
        snprintf(dst, sizeof(dst), "%s/%s", g_loras_root, cur.filename);

        dl_lock();
        g_dl.file_idx = done_ok + 1;
        total_batch = done_ok + 1 + g_lora_q_count;
        g_dl.file_total = total_batch;
        g_dl.queue_len = g_lora_q_count;
        g_dl.bytes_done = 0;
        g_dl.bytes_total = 0;
        g_dl.speed_mbs = 0.0;
        snprintf(g_dl.model_id, sizeof(g_dl.model_id), "%s", cur.filename);
        snprintf(g_dl.cur_file, sizeof(g_dl.cur_file), "%s", cur.filename);
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Downloading LoRA %s...", cur.filename);
        dl_unlock();

        int rc = http_fetch_url(cur.url, dst, NULL, 0, 1, err, sizeof(err));

        dl_lock();
        if (rc == 0) {
            done_ok++;
            g_dl.finished++;
            g_dl.error[0] = 0;
            glog("lora_dl: completed %s -> %s", cur.id, dst);
        } else if (rc == 1 || g_dl.cancel_req) {
            g_lora_q_count = 0;
            g_dl.queue_len = 0;
            g_dl.active = 0;
            g_lora_dl_is_lora = 0;
            snprintf(g_dl.error, sizeof(g_dl.error), "Download cancelled");
            snprintf(g_dl.stage, sizeof(g_dl.stage), "Download stopped");
            dl_unlock();
            return NULL;
        } else {
            snprintf(g_dl.error, sizeof(g_dl.error), "%s", err[0] ? err : "LoRA download failed");
            glog("lora_dl: failed %s: %s", cur.id, g_dl.error);
        }

        if (g_lora_q_count > 0 && !g_dl.cancel_req) {
            cur = g_lora_q[0];
            memmove(&g_lora_q[0], &g_lora_q[1], (size_t)(g_lora_q_count - 1) * sizeof(LoraDlTask));
            g_lora_q_count--;
            g_dl.queue_len = g_lora_q_count;
            dl_unlock();
            continue;
        }

        g_dl.active = 0;
        g_lora_dl_is_lora = 0;
        g_dl.queue_len = 0;
        if (done_ok > 1) {
            snprintf(g_dl.stage, sizeof(g_dl.stage), "Installed %d LoRAs! Ready in Studio.", done_ok);
        } else if (rc == 0) {
            snprintf(g_dl.stage, sizeof(g_dl.stage), "Installed LoRA %s! Ready in Studio.", cur.filename);
        } else {
            snprintf(g_dl.stage, sizeof(g_dl.stage), "Download stopped");
        }
        dl_unlock();
        return NULL;
    }
}

int lora_dl_start(const char* id_or_name, const char* url_override, const char* filename_override) {
    dl_lock();
    LoraDlTask resolved;
    if (resolve_lora_task_locked(id_or_name, url_override, filename_override, &resolved) != 0) {
        snprintf(g_dl.error, sizeof(g_dl.error), "Invalid LoRA URL or filename");
        dl_unlock();
        return -3;
    }

    if (g_dl.active) {
        if (!g_lora_dl_is_lora) { dl_unlock(); return -2; }
        // Avoid duplicates already active or queued
        if (strcmp(g_dl.cur_file, resolved.filename) == 0) { dl_unlock(); return 1; }
        for (int i = 0; i < g_lora_q_count; i++) {
            if (strcmp(g_lora_q[i].filename, resolved.filename) == 0) { dl_unlock(); return 1; }
        }
        if (g_lora_q_count >= LORA_DL_QMAX) {
            snprintf(g_dl.error, sizeof(g_dl.error), "LoRA download queue is full");
            dl_unlock();
            return -2;
        }
        g_lora_q[g_lora_q_count++] = resolved;
        g_dl.queue_len = g_lora_q_count;
        g_dl.file_total = g_dl.file_idx + g_lora_q_count;
        dl_unlock();
        return 1; // Queued behind active LoRA download
    }

    LoraDlTask* t = (LoraDlTask*)malloc(sizeof(LoraDlTask));
    if (!t) { dl_unlock(); return -1; }
    *t = resolved;

    g_dl.active = 1;
    g_lora_dl_is_lora = 1;
    g_dl.cancel_req = 0;
    g_dl.error[0] = 0;
    g_dl.bytes_done = 0;
    g_dl.bytes_total = 0;
    g_dl.speed_mbs = 0.0;
    g_dl.file_idx = 1;
    g_dl.file_total = 1 + g_lora_q_count;
    g_dl.queue_len = g_lora_q_count;
    snprintf(g_dl.model_id, sizeof(g_dl.model_id), "%s", t->filename);
    snprintf(g_dl.repo, sizeof(g_dl.repo), "LoRA HTTPS");
    snprintf(g_dl.stage, sizeof(g_dl.stage), "Connecting to download LoRA...");
    dl_unlock();

    OrbisPthread th;
    OrbisPthreadAttr attr;
    scePthreadAttrInit(&attr);
    scePthreadAttrSetstacksize(&attr, 256 * 1024);
    scePthreadAttrSetdetachstate(&attr, 1);
    if (scePthreadCreate(&th, &attr, lora_dl_thread_fn, t, "gdiff_ldl") != 0) {
        scePthreadAttrDestroy(&attr);
        free(t);
        dl_lock(); g_dl.active = 0; g_lora_dl_is_lora = 0;
        snprintf(g_dl.error, sizeof(g_dl.error), "failed to start LoRA download thread");
        dl_unlock();
        return -1;
    }
    scePthreadAttrDestroy(&attr);
    return 0;
}

int lora_dl_queue_all(const char* arch_filter) {
    InstalledLora* inst = (InstalledLora*)malloc(sizeof(InstalledLora) * LORA_MAX_INSTALLED);
    CatalogLora* cat = (CatalogLora*)malloc(sizeof(CatalogLora) * LORA_MAX_CATALOG);
    if (!inst || !cat) { free(inst); free(cat); return -1; }
    int ni = lora_list_installed(inst, LORA_MAX_INSTALLED);
    int nc = lora_get_catalog(cat, LORA_MAX_CATALOG);
    int queued = 0;
    int all_arch = (!arch_filter || !arch_filter[0] || strcmp(arch_filter, "all") == 0);
    for (int i = 0; i < nc; i++) {
        if (!all_arch && strcmp(cat[i].arch, arch_filter) != 0) continue;
        int already = 0;
        for (int j = 0; j < ni; j++) {
            if (strcmp(inst[j].filename, cat[i].filename) == 0 || strcmp(inst[j].name, cat[i].id) == 0) {
                already = 1; break;
            }
        }
        if (already) continue;
        int rc = lora_dl_start(cat[i].id, cat[i].url, cat[i].filename);
        if (rc == 0 || rc == 1) queued++;
        else if (rc == -2) { queued = -2; break; }
    }
    free(inst);
    free(cat);
    return queued;
}

int vae_enc_is_installed(int is_sd15) {
    const char* fn = is_sd15 ? "vae_enc_sd15.psd1" : "vae_enc_sdxl.psd1";
    char p1[320], p2[320];
    snprintf(p1, sizeof(p1), "/data/garlic-diffusion/%s", fn);
    snprintf(p2, sizeof(p2), "%s/%s", g_models_root, fn);
    return (sdxl_check_pack(p1, NULL, NULL) == 0 || sdxl_check_pack(p2, NULL, NULL) == 0) ? 1 : 0;
}

typedef struct {
    int is_sd15;
    char filename[48];
    char url[384];
} VaeEncDlTask;

static void* vae_enc_dl_thread_fn(void* arg) {
    VaeEncDlTask* t = (VaeEncDlTask*)arg;
    ensure_dir_dl("/data/garlic-diffusion");
    char dst[320], err[256] = {0};
    snprintf(dst, sizeof(dst), "/data/garlic-diffusion/%s", t->filename);

    dl_lock();
    g_dl.file_idx = 1;
    g_dl.file_total = 1;
    snprintf(g_dl.cur_file, sizeof(g_dl.cur_file), "%s", t->filename);
    snprintf(g_dl.stage, sizeof(g_dl.stage), "Downloading %s (Fix Face / img2img)...", t->filename);
    dl_unlock();

    int rc = http_fetch_url(t->url, dst, NULL, 0, 1, err, sizeof(err));

    dl_lock();
    g_dl.active = 0;
    if (rc == 0) {
        g_dl.finished++;
        g_dl.error[0] = 0;
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Installed %s! Full AI Fix Face is now active.", t->filename);
        glog("vae_enc_dl: completed %s -> %s", t->filename, dst);
    } else if (rc == 1) {
        snprintf(g_dl.error, sizeof(g_dl.error), "Download cancelled");
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Download stopped");
    } else {
        snprintf(g_dl.error, sizeof(g_dl.error), "%s", err[0] ? err : "VAE encoder download failed");
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Download stopped");
        glog("vae_enc_dl: failed %s: %s", t->filename, g_dl.error);
    }
    dl_unlock();
    free(t);
    return NULL;
}

int vae_enc_dl_start(int is_sd15) {
    dl_lock();
    if (g_dl.active) { dl_unlock(); return -2; }
    VaeEncDlTask* t = (VaeEncDlTask*)calloc(1, sizeof(VaeEncDlTask));
    if (!t) { dl_unlock(); return -1; }
    t->is_sd15 = is_sd15;
    snprintf(t->filename, sizeof(t->filename), "%s", is_sd15 ? "vae_enc_sd15.psd1" : "vae_enc_sdxl.psd1");
    const char* repo = g_hf_repo[0] ? g_hf_repo : "DJTomato/ps4-diffusion-models";
    snprintf(t->url, sizeof(t->url), "https://huggingface.co/%s/resolve/main/%s", repo, t->filename);

    g_dl.active = 1;
    g_lora_dl_is_lora = 0;
    g_dl.cancel_req = 0;
    g_dl.error[0] = 0;
    g_dl.bytes_done = 0;
    g_dl.bytes_total = 0;
    g_dl.speed_mbs = 0.0;
    g_dl.file_idx = 1;
    g_dl.file_total = 1;
    g_dl.queue_len = 0;
    snprintf(g_dl.model_id, sizeof(g_dl.model_id), "%s", is_sd15 ? "vae_enc_sd15" : "vae_enc_sdxl");
    snprintf(g_dl.repo, sizeof(g_dl.repo), "%s", repo);
    snprintf(g_dl.cur_file, sizeof(g_dl.cur_file), "%s", t->filename);
    snprintf(g_dl.stage, sizeof(g_dl.stage), "Connecting to download %s...", t->filename);
    dl_unlock();

    OrbisPthread th;
    OrbisPthreadAttr attr;
    scePthreadAttrInit(&attr);
    scePthreadAttrSetstacksize(&attr, 256 * 1024);
    scePthreadAttrSetdetachstate(&attr, 1);
    if (scePthreadCreate(&th, &attr, vae_enc_dl_thread_fn, t, "gdiff_vdl") != 0) {
        scePthreadAttrDestroy(&attr);
        free(t);
        dl_lock(); g_dl.active = 0; snprintf(g_dl.error, sizeof(g_dl.error), "failed to start VAE encoder download"); dl_unlock();
        return -1;
    }
    scePthreadAttrDestroy(&attr);
    return 0;
}

int vae_enc_ensure_sync(int is_sd15) {
    if (vae_enc_is_installed(is_sd15)) return 0;
    ensure_dir_dl("/data/garlic-diffusion");
    const char* fn = is_sd15 ? "vae_enc_sd15.psd1" : "vae_enc_sdxl.psd1";
    const char* repo = g_hf_repo[0] ? g_hf_repo : "DJTomato/ps4-diffusion-models";
    char url[384], dst[320], err[256] = {0};
    snprintf(url, sizeof(url), "https://huggingface.co/%s/resolve/main/%s", repo, fn);
    snprintf(dst, sizeof(dst), "/data/garlic-diffusion/%s", fn);
    int rc = http_fetch_url(url, dst, NULL, 0, 0, err, sizeof(err));
    if (rc == 0 && sdxl_check_pack(dst, NULL, NULL) == 0) {
        glog("vae_enc_ensure_sync: auto-downloaded %s", dst);
        return 0;
    }
    unlink(dst);
    glog("vae_enc_ensure_sync: %s unavailable (%s), falling back to Real-ESRGAN face restore", fn, err);
    return -1;
}

int model_dl_start(const char* model_id, const char* repo_override, const char* files_override) {
    if (!gallery_valid_id(model_id)) return -1;
    if (strcmp(model_id, "vae_enc_sdxl") == 0) return vae_enc_dl_start(0);
    if (strcmp(model_id, "vae_enc_sd15") == 0) return vae_enc_dl_start(1);
    dl_lock();
    if (g_dl.active) { dl_unlock(); return -2; }

    DlTask* t = (DlTask*)calloc(1, sizeof(DlTask));
    if (!t) { dl_unlock(); return -1; }
    snprintf(t->model_id, sizeof(t->model_id), "%s", model_id);

    // Resolve repo + file list from override, catalog, or default HF repo
    const CatalogModel* cm = NULL;
    for (int i = 0; i < g_catalog_count; i++) {
        if (strcmp(g_catalog[i].id, model_id) == 0) { cm = &g_catalog[i]; break; }
    }
    if (repo_override && repo_override[0]) snprintf(t->repo, sizeof(t->repo), "%s", repo_override);
    else if (cm && cm->repo[0]) snprintf(t->repo, sizeof(t->repo), "%s", cm->repo);
    else snprintf(t->repo, sizeof(t->repo), "%s", g_hf_repo);

    if (files_override && files_override[0]) snprintf(t->files, sizeof(t->files), "%s", files_override);
    else if (cm && cm->files[0]) snprintf(t->files, sizeof(t->files), "%s", cm->files);
    else snprintf(t->files, sizeof(t->files), "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg");

    if (!t->repo[0]) {
        free(t);
        snprintf(g_dl.error, sizeof(g_dl.error), "Set your Hugging Face repo first (e.g. username/garlic-diffusion-models)");
        dl_unlock();
        return -3;
    }

    g_dl.active = 1;
    g_dl.cancel_req = 0;
    g_dl.error[0] = 0;
    g_dl.bytes_done = 0;
    g_dl.bytes_total = 0;
    g_dl.speed_mbs = 0.0;
    g_dl.file_idx = 0;
    g_dl.file_total = 0;
    snprintf(g_dl.model_id, sizeof(g_dl.model_id), "%s", t->model_id);
    snprintf(g_dl.repo, sizeof(g_dl.repo), "%s", t->repo);
    snprintf(g_dl.stage, sizeof(g_dl.stage), "Connecting to Hugging Face...");
    dl_unlock();

    OrbisPthread th;
    OrbisPthreadAttr attr;
    scePthreadAttrInit(&attr);
    scePthreadAttrSetstacksize(&attr, 256 * 1024);
    scePthreadAttrSetdetachstate(&attr, 1);
    if (scePthreadCreate(&th, &attr, dl_thread_fn, t, "gdiff_dl") != 0) {
        scePthreadAttrDestroy(&attr);
        free(t);
        dl_lock(); g_dl.active = 0; snprintf(g_dl.error, sizeof(g_dl.error), "failed to start download thread"); dl_unlock();
        return -1;
    }
    scePthreadAttrDestroy(&attr);
    return 0;
}

#else // Host CPU test stubs
int model_dl_refresh_catalog(void) {
    return apply_manifest_json("[{\"id\":\"test_model\",\"title\":\"Test\",\"arch\":\"sd15\",\"size_mb\":1200,\"desc\":\"ok\",\"files\":\"unet.psd1\"}]");
}
int model_dl_start(const char* model_id, const char* repo_override, const char* files_override) {
    (void)model_id; (void)repo_override; (void)files_override;
    return -1;
}
int lora_dl_queue_all(const char* arch_filter) {
    (void)arch_filter;
    return 0;
}
int lora_dl_is_queued(const char* id_or_filename) {
    (void)id_or_filename;
    return 0;
}
int vae_enc_is_installed(int is_sd15) {
    (void)is_sd15;
    return 0;
}
int vae_enc_dl_start(int is_sd15) {
    (void)is_sd15;
    return -1;
}
int vae_enc_ensure_sync(int is_sd15) {
    (void)is_sd15;
    return -1;
}
int lora_refresh_catalog(void) {
    return apply_lora_manifest_json("[{\"id\":\"test_lora\",\"title\":\"Test LoRA\",\"arch\":\"sd15\",\"size_mb\":18,\"default_weight\":0.8,\"desc\":\"ok\",\"filename\":\"test_lora.safetensors\",\"url\":\"https://example.com/test_lora.safetensors\"}]");
}
int lora_dl_start(const char* id_or_name, const char* url_override, const char* filename_override) {
    (void)id_or_name; (void)url_override; (void)filename_override;
    return -1;
}
#endif

// ---- GitHub OTA Update Checker, /data/pkg Downloader & Overwrite/Exit Helper --------------------
static void parse_ver_parts(const char* s, int out[3]) {
    out[0] = out[1] = out[2] = 0;
    if (!s) return;
    while (*s == ' ' || *s == '\t' || *s == 'v' || *s == 'V') s++;
    for (int i = 0; i < 3 && *s; i++) {
        if (!isdigit((unsigned char)*s)) break;
        out[i] = (int)strtol(s, (char**)&s, 10);
        if (*s == '.') s++;
        else break;
    }
}

int app_version_is_newer(const char* cur_ver, const char* rem_ver) {
    int c[3], r[3];
    parse_ver_parts(cur_ver, c);
    parse_ver_parts(rem_ver, r);
    for (int i = 0; i < 3; i++) {
        if (r[i] != c[i]) return r[i] > c[i] ? 1 : 0;
    }
    return 0;
}

static int apply_version_json(const char* json) {
    if (!json) return -1;
    const char* ob = strchr(json, '{');
    const char* cb = ob ? strrchr(ob, '}') : NULL;
    if (!ob || !cb || cb <= ob) return -1;
    char ver[16] = {0}, purl[256] = {0}, chg[256] = {0};
    if (!parse_manifest_str(ob, cb, "version", ver, sizeof(ver)) || !ver[0]) return -1;
    parse_manifest_str(ob, cb, "pkg_url", purl, sizeof(purl));
    parse_manifest_str(ob, cb, "changelog", chg, sizeof(chg));
    uint32_t sz = parse_manifest_u32(ob, cb, "size_mb", 8);

    dl_lock();
    snprintf(g_upd.current_ver, sizeof(g_upd.current_ver), "%s", GD_APP_VERSION);
    snprintf(g_upd.remote_ver, sizeof(g_upd.remote_ver), "%s", ver);
    if (purl[0]) snprintf(g_upd.pkg_url, sizeof(g_upd.pkg_url), "%s", purl);
    else if (!g_upd.pkg_url[0]) snprintf(g_upd.pkg_url, sizeof(g_upd.pkg_url), "%s", DEFAULT_OTA_PKG_URL);
    if (chg[0]) snprintf(g_upd.changelog, sizeof(g_upd.changelog), "%s", chg);
    g_upd.size_mb = sz;
    g_upd.checked = 1;
    g_upd.update_available = app_version_is_newer(g_upd.current_ver, g_upd.remote_ver);
    g_upd.pkg_ready = check_ota_pkg_ready_unlocked();
    dl_unlock();
    return 0;
}

void app_update_get_status(AppUpdateStatus* out) {
    if (!out) return;
    dl_lock();
    if (!g_upd.current_ver[0]) snprintf(g_upd.current_ver, sizeof(g_upd.current_ver), "%s", GD_APP_VERSION);
    if (!g_upd.pkg_url[0]) snprintf(g_upd.pkg_url, sizeof(g_upd.pkg_url), "%s", DEFAULT_OTA_PKG_URL);
    g_upd.pkg_ready = check_ota_pkg_ready_unlocked();
    *out = g_upd;
    dl_unlock();
}

#ifndef SDXL_CPU
int app_update_check_now(void) {
    char* buf = (char*)malloc(8192);
    if (!buf) return -1;
    char err[160] = {0};
    buf[0] = 0;
    int rc = http_fetch_url(GITHUB_VERSION_URL, NULL, buf, 8192, 0, err, sizeof(err));
    if (rc == 0) {
        rc = apply_version_json(buf);
        if (rc == 0) {
            AppUpdateStatus s;
            app_update_get_status(&s);
            glog("ota_update: current=%s remote=%s update_available=%d pkg_ready=%d",
                 s.current_ver, s.remote_ver, s.update_available, s.pkg_ready);
        }
    } else {
        glog("ota_update: GitHub version check skipped (%s)", err);
    }
    free(buf);
    return rc;
}

static void prepare_overwrite_pkg_ps4(void) {
    sceSysmoduleLoadModuleInternal(ORBIS_SYSMODULE_INTERNAL_APP_INST_UTIL);
    int ir = sceAppInstUtilInitialize();
    int pr = sceAppInstUtilAppPrepareOverwritePkg(OTA_PKG_DEST);
    glog("ota_update: AppInstUtil init=0x%08x prepareOverwrite(%s)=0x%08x", (unsigned)ir, OTA_PKG_DEST, (unsigned)pr);
}

static void try_goldhen_rpi_install(void) {
    if (ensure_http_client() != 0) return;
    const char* rpi_url = "http://127.0.0.1:12800/api/install";
    const char* payload = "{\"type\":\"direct\",\"packages\":[\"/data/pkg/Garlic-Diffusion.pkg\"]}";
    int tpl = sceHttpCreateTemplate(g_http_ctx, "GarlicDiffusion/1.15", ORBIS_HTTP_VERSION_1_1, 1);
    if (tpl < 0) return;
    sceHttpSetConnectTimeOut(tpl, 1200000u);
    sceHttpSetSendTimeOut(tpl, 1500000u);
    sceHttpSetRecvTimeOut(tpl, 1500000u);
    int conn = sceHttpCreateConnectionWithURL(tpl, rpi_url, false);
    int req = (conn >= 0) ? sceHttpCreateRequestWithURL(conn, ORBIS_METHOD_POST, rpi_url, strlen(payload)) : -1;
    if (req >= 0) {
        sceHttpAddRequestHeader(req, "Content-Type", "application/json", 0);
        int sr = sceHttpSendRequest(req, payload, strlen(payload));
        int32_t code = 0;
        if (sr >= 0) sceHttpGetStatusCode(req, &code);
        glog("ota_update: GoldHEN RPI 127.0.0.1:12800 status=%d (sr=0x%08x)", (int)code, (unsigned)sr);
        sceHttpDeleteRequest(req);
    }
    if (conn >= 0) sceHttpDeleteConnection(conn);
    sceHttpDeleteTemplate(tpl);
}

static void* ota_dl_thread_fn(void* arg) {
    (void)arg;
    ensure_dir_dl("/data/pkg");
    char url[256], err[256] = {0};
    dl_lock();
    snprintf(url, sizeof(url), "%s", g_upd.pkg_url[0] ? g_upd.pkg_url : DEFAULT_OTA_PKG_URL);
    g_dl.file_idx = 1;
    g_dl.file_total = 1;
    snprintf(g_dl.cur_file, sizeof(g_dl.cur_file), "Garlic-Diffusion.pkg");
    snprintf(g_dl.stage, sizeof(g_dl.stage), "Downloading update to /data/pkg/Garlic-Diffusion.pkg...");
    dl_unlock();

    glog("ota_update: downloading %s -> %s", url, OTA_PKG_DEST);
    int rc = http_fetch_url(url, OTA_PKG_DEST, NULL, 0, 1, err, sizeof(err));
    if (rc == 0) {
        prepare_overwrite_pkg_ps4();
    }

    dl_lock();
    g_dl.active = 0;
    g_upd.pkg_ready = check_ota_pkg_ready_unlocked();
    if (rc == 0) {
        g_dl.finished++;
        g_dl.error[0] = 0;
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Saved /data/pkg/Garlic-Diffusion.pkg! Ready to close & install.");
        glog("ota_update: saved %s successfully", OTA_PKG_DEST);
    } else if (rc == 1) {
        snprintf(g_dl.error, sizeof(g_dl.error), "Update download cancelled");
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Download stopped");
    } else {
        snprintf(g_dl.error, sizeof(g_dl.error), "%s", err[0] ? err : "OTA PKG download failed");
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Download stopped");
        glog("ota_update: failed: %s", g_dl.error);
    }
    dl_unlock();
    return NULL;
}

int app_update_download_pkg(void) {
    dl_lock();
    if (g_dl.active) { dl_unlock(); return -2; }
    g_dl.active = 1;
    g_dl.cancel_req = 0;
    g_dl.error[0] = 0;
    g_dl.bytes_done = 0;
    g_dl.bytes_total = 0;
    g_dl.speed_mbs = 0.0;
    g_dl.file_idx = 1;
    g_dl.file_total = 1;
    snprintf(g_dl.model_id, sizeof(g_dl.model_id), "Garlic-Diffusion.pkg");
    snprintf(g_dl.repo, sizeof(g_dl.repo), "GitHub OTA");
    snprintf(g_dl.cur_file, sizeof(g_dl.cur_file), "Garlic-Diffusion.pkg");
    snprintf(g_dl.stage, sizeof(g_dl.stage), "Connecting to GitHub Releases...");
    dl_unlock();

    OrbisPthread th;
    OrbisPthreadAttr attr;
    scePthreadAttrInit(&attr);
    scePthreadAttrSetstacksize(&attr, 256 * 1024);
    scePthreadAttrSetdetachstate(&attr, 1);
    if (scePthreadCreate(&th, &attr, ota_dl_thread_fn, NULL, "gdiff_ota") != 0) {
        scePthreadAttrDestroy(&attr);
        dl_lock();
        g_dl.active = 0;
        snprintf(g_dl.error, sizeof(g_dl.error), "failed to start OTA download thread");
        dl_unlock();
        return -1;
    }
    scePthreadAttrDestroy(&attr);
    return 0;
}

static void* ota_exit_worker_fn(void* arg) {
    (void)arg;
    sceKernelUsleep(900000); // Allow HTTP response / TV toast frame to flush cleanly
    sync();
    glog("ota_update: exiting app cleanly so /data/pkg/Garlic-Diffusion.pkg can be installed");
    sceSystemServiceLoadExec("exit", NULL);
    sceKernelUsleep(300000);
    _exit(0);
    return NULL;
}

int app_update_prepare_and_exit(char* out_msg, size_t max_msg) {
    if (!check_ota_pkg_ready_unlocked()) {
        if (out_msg && max_msg)
            snprintf(out_msg, max_msg, "Garlic-Diffusion.pkg is not in /data/pkg yet — click Download Update first.");
        return -1;
    }
    prepare_overwrite_pkg_ps4();
    try_goldhen_rpi_install();
    if (out_msg && max_msg) {
        snprintf(out_msg, max_msg,
                 "Closing GarlicDiffusion on PS4 now! Open GoldHEN -> Package Installer -> /data/pkg/Garlic-Diffusion.pkg to install (your models, LoRAs & gallery in /data/garlic-diffusion stay 100%% intact).");
    }
    OrbisPthread th;
    OrbisPthreadAttr attr;
    scePthreadAttrInit(&attr);
    scePthreadAttrSetstacksize(&attr, 128 * 1024);
    scePthreadAttrSetdetachstate(&attr, 1);
    scePthreadCreate(&th, &attr, ota_exit_worker_fn, NULL, "gdiff_exit");
    scePthreadAttrDestroy(&attr);
    return 0;
}
#else
int app_update_check_now(void) {
    return apply_version_json("{\"version\":\"1.18\",\"pkg_url\":\"" DEFAULT_OTA_PKG_URL "\",\"size_mb\":7,\"changelog\":\"Test OTA update\"}");
}
int app_update_download_pkg(void) { return -1; }
int app_update_prepare_and_exit(char* out_msg, size_t max_msg) {
    if (out_msg && max_msg) snprintf(out_msg, max_msg, "Host stub");
    return -1;
}
#endif
