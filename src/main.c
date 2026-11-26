// Garlic Diffusion: Standalone SDXL Anime Image Generator for PS4 / PS4 Pro
// Title ID: PSAI00003 ("Garlic Diffusion")
// TV UI + Controller + PS4 OSK/USB Keyboard + Built-in HTTP Web UI & OpenAI API on port 8081.

#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <orbis/CommonDialog.h>
#include <orbis/ImeDialog.h>
#include <orbis/Pad.h>
#include <orbis/Sysmodule.h>
#include <orbis/SystemService.h>
#include <orbis/UserService.h>
#include <orbis/VideoOut.h>
#include <orbis/libkernel.h>

#include "face_detect.h"
#include "gallery.h"
#include "garlic_ui.h"
#include "http_srv.h"
#include "model_dl.h"
#include "png_write.h"
#include "randomizer_pools.h"
#include "sdxl.h"
#include "studio_menu.h"

#define SCREEN_W 1920
#define SCREEN_H 1080
#define ASSETS_DIR "/app0/assets"
#define MODELS_ROOT "/data/garlic-diffusion/models"
#define LORAS_ROOT "/data/garlic-diffusion/loras"
#define OUT_ROOT "/data/garlic-diffusion/out"
#define PRESETS_ROOT "/data/garlic-diffusion/presets"
#define STYLES_ROOT "/data/garlic-diffusion/styles"
#define CHARS_ROOT "/data/garlic-diffusion/characters"
#define REPO_CFG "/data/garlic-diffusion/repo.cfg"
#define UI_MODE_CFG "/data/garlic-diffusion/ui_mode.cfg"
#define LOG_FILE "/data/garlic_diffusion_log.txt"

void glog(const char* fmt, ...) {
    char buf[420];
    int h = snprintf(buf, 32, "[gdiff %8.3f] ", sceKernelGetProcessTime() / 1e6);
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + h, sizeof(buf) - (size_t)h - 2, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (n > (int)sizeof(buf) - h - 2) n = (int)sizeof(buf) - h - 2;
    buf[h + n] = '\n';
    buf[h + n + 1] = 0;
    sceKernelDebugOutText(0, buf);
    int f = open(LOG_FILE, O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (f >= 0) {
        write(f, buf, (size_t)(h + n + 1));
        fsync(f);
        close(f);
    }
}

// ---- VideoOut -----------------------------------------------------------------------------------
static int video_fd;
static OrbisKernelEqueue flip_eq;
static uint32_t* fb[2];
static uint32_t* back;
static int cur_fb = 0, flip_id = 1;
static int32_t user_id = -1, pad_fd = -1;

static int video_init(void) {
    video_fd = sceVideoOutOpen(ORBIS_VIDEO_USER_MAIN, ORBIS_VIDEO_OUT_BUS_MAIN, 0, 0);
    if (video_fd < 0) return -1;
    sceKernelCreateEqueue(&flip_eq, "gdiff flip");
    sceVideoOutAddFlipEvent(flip_eq, video_fd, 0);
    size_t sz = (2ull * SCREEN_W * SCREEN_H * 4 + 0x1FFFFF) & ~0x1FFFFFull;
    off_t off = 0;
    void* mem = NULL;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), sz, 0x200000, 3, &off)) return -2;
    if (sceKernelMapDirectMemory(&mem, sz, 0x33, 0, off, 0x200000)) return -3;
    fb[0] = (uint32_t*)mem;
    fb[1] = fb[0] + SCREEN_W * SCREEN_H;
    OrbisVideoOutBufferAttribute attr;
    sceVideoOutSetBufferAttribute(&attr, 0x80000000, 1, 0, SCREEN_W, SCREEN_H, SCREEN_W);
    if (sceVideoOutRegisterBuffers(video_fd, 0, (void**)fb, 2, &attr)) return -4;
    sceVideoOutSetFlipRate(video_fd, 0);
    back = (uint32_t*)malloc((size_t)SCREEN_W * SCREEN_H * 4);
    return back ? 0 : -5;
}

static void present(void) {
    memcpy(fb[cur_fb], back, (size_t)SCREEN_W * SCREEN_H * 4);
    sceVideoOutSubmitFlip(video_fd, cur_fb, ORBIS_VIDEO_OUT_FLIP_VSYNC, flip_id);
    for (;;) {
        OrbisVideoOutFlipStatus st;
        sceVideoOutGetFlipStatus(video_fd, &st);
        if (st.flipArg >= flip_id) break;
        OrbisKernelEvent ev;
        int cnt;
        if (sceKernelWaitEqueue(flip_eq, &ev, 1, &cnt, 0)) break;
    }
    cur_fb ^= 1;
    flip_id++;
}

// ---- App State ----------------------------------------------------------------------------------
static SdxlEngine eng;
static DiffJob g_job;
static DiffStatus g_st;
static uint8_t* g_prev_rgb = NULL;
static uint32_t g_prev_w = 0, g_prev_h = 0, g_prev_seq = 0;
static uint8_t* ui_prev_rgb = NULL;
static uint32_t ui_prev_w = 0, ui_prev_h = 0, ui_prev_seq = 0;
static char toast[128] = "";
static int g_welcome_sel = 0;
static int g_easy_framing = 2; // 0=Portrait, 1=Upper Body, 2=Full Body, 3=Wide 16:9
static int g_easy_quality = 1; // 0=Fast Speed, 1=Best Quality, 2=Best + 1080p HD
static int g_ota_notified = 0;

static int load_ui_mode_cfg(void) {
    int fd = open(UI_MODE_CFG, O_RDONLY);
    if (fd < 0) return -1;
    char buf[64] = {0};
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    const char* p = strstr(buf, "ui_mode=");
    p = p ? (p + 8) : buf;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '0') return 0;
    if (*p == '1') return 1;
    return -1;
}

static void save_ui_mode_cfg(int mode) {
    if (mode != 0 && mode != 1) return;
    int fd = open(UI_MODE_CFG, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd >= 0) {
        char buf[32];
        int n = snprintf(buf, sizeof(buf), "ui_mode=%d\n", mode);
        if (n > 0) (void)write(fd, buf, (size_t)n);
        close(fd);
    }
}

static void publish_preview_rgb(uint8_t* rgb, uint32_t w, uint32_t h) {
    http_srv_lock();
    free(g_prev_rgb);
    g_prev_rgb = rgb;
    g_prev_w = w;
    g_prev_h = h;
    g_prev_seq++;
    http_srv_unlock();
}

// Dynamic User Presets, Art Styles, Character Presets + TV Gallery + TV Model Manager state
static UserPreset g_all_presets[PRESETS_MAX_ITEMS];
static int g_num_all_presets = 0;
static UserPreset g_presets[PRESETS_MAX_ITEMS];
static int g_num_presets = 0;
static StyleItem g_styles[STYLES_MAX_ITEMS];
static int g_num_styles = 0, style_idx = -1;
static CharacterItem g_chars[CHARS_MAX_ITEMS];
static int g_num_chars = 0, char_idx = -1;
static int g_tv_gen = 0;
static GalleryItem g_gal[GALLERY_TV_MAX];
static int g_gal_count = 0, g_gal_idx = 0;
static int g_in_gallery = 0, g_gal_full = 0, g_del_arm = 0;
static char g_gal_folder[64] = ""; // ""=All, "__starred__"=Starred, "__unsorted__"=Unsorted, or folder name
static GalleryFolder g_folders[GALLERY_MAX_FOLDERS];
static int g_num_folders = 0;
static uint8_t* g_gal_rgb = NULL;
static uint32_t g_gal_w = 0, g_gal_h = 0;
static char g_gal_sel_ids[DIFF_MAX_BATCH_GAL][64];
static int g_gal_sel_n = 0;
static int g_gal_up_mode = 1; // 1=1080p HD, 2=2x HD, 3=4K UHD
static char g_easy_scene[512] = "";
static int g_gal_fmodal = 0, g_gal_fsel = 0, g_gal_ftop = 0;
static int g_gal_amodal = 0, g_gal_asel = 0; // Artwork Actions menu on (X)
static int g_pending_gal_restore = 0, g_pending_gal_run = 0;
static GalleryItem g_pending_gal_item;

// Live job timing state for real-time ETA on TV & Web UI
static uint64_t g_job_start_us = 0;
static uint32_t g_eta_done_items = 0;
static double g_eta_done_sec = 0.0;
static uint64_t g_eta_lat = 12288ull;
static float g_eta_cfg = 1.4f;
static int g_eta_ups = 0, g_eta_eco = 2, g_eta_gal_batch = 0;

static int gal_is_selected(const char* id) {
    if (!id || !id[0]) return -1;
    for (int i = 0; i < g_gal_sel_n; i++) {
        if (strcmp(g_gal_sel_ids[i], id) == 0) return i;
    }
    return -1;
}

static void gal_toggle_select(const char* id) {
    if (!id || !id[0]) return;
    int pos = gal_is_selected(id);
    if (pos >= 0) {
        for (int i = pos; i + 1 < g_gal_sel_n; i++)
            memcpy(g_gal_sel_ids[i], g_gal_sel_ids[i + 1], 64);
        g_gal_sel_n--;
    } else if (g_gal_sel_n < DIFF_MAX_BATCH_GAL) {
        snprintf(g_gal_sel_ids[g_gal_sel_n], 64, "%s", id);
        g_gal_sel_n++;
    }
}

static void gal_clear_select(void) {
    g_gal_sel_n = 0;
}

static int g_in_models = 0, g_mgr_tab = 0, g_lora_sub = 0, g_in_credits = 0, g_mdl_sel = 0, g_mdl_del_arm = 0, g_last_dl_fin = 0;
#define LORA_TV_PAGE_SZ 8
static InstalledModel g_inst[MODEL_DL_MAX_INSTALLED];
static int g_inst_cnt = 0;
static CatalogModel g_cat[MODEL_DL_MAX_CATALOG];
static int g_cat_cnt = 0;
static InstalledLora g_loras[LORA_MAX_INSTALLED];
static int g_num_loras = 0;
static CatalogLora g_lcat[LORA_MAX_CATALOG];
static int g_lcat_cnt = 0;
static int preset_idx = 0, res_idx = 4, model_idx = 0;

static void refresh_tv_presets(void) {
    g_num_all_presets = presets_list(g_all_presets, PRESETS_MAX_ITEMS);
    int allow_nsfw = 0;
    http_srv_lock();
    allow_nsfw = g_job.nsfw_allow;
    http_srv_unlock();
    int out_n = 0;
    for (int i = 0; i < g_num_all_presets && out_n < PRESETS_MAX_ITEMS; i++) {
        if (g_all_presets[i].nsfw && !allow_nsfw) continue;
        g_presets[out_n++] = g_all_presets[i];
    }
    g_num_presets = out_n;
    if (preset_idx >= g_num_presets) preset_idx = g_num_presets > 0 ? 0 : -1;
}

static void apply_tv_char_selection(int prev_idx, int new_idx) {
    http_srv_lock();
    char work_p[1024];
    snprintf(work_p, sizeof(work_p), "%s", g_job.prompt);
    if (prev_idx >= 0 && prev_idx < g_num_chars && g_chars[prev_idx].prompt[0]) {
        const char* old_cp = g_chars[prev_idx].prompt;
        size_t ol = strlen(old_cp);
        char* hit = strstr(work_p, old_cp);
        if (hit) {
            const char* rest = hit + ol;
            while (*rest == ' ' || *rest == ',') rest++;
            char stripped[1024];
            size_t pre = (size_t)(hit - work_p);
            while (pre > 0 && (work_p[pre - 1] == ' ' || work_p[pre - 1] == ',')) pre--;
            if (pre > 0 && *rest)
                snprintf(stripped, sizeof(stripped), "%.*s, %s", (int)pre, work_p, rest);
            else if (pre > 0)
                snprintf(stripped, sizeof(stripped), "%.*s", (int)pre, work_p);
            else
                snprintf(stripped, sizeof(stripped), "%s", rest);
            snprintf(work_p, sizeof(work_p), "%s", stripped);
        }
    }
    if (new_idx >= 0 && new_idx < g_num_chars && g_chars[new_idx].prompt[0]) {
        char merged[1024];
        if (work_p[0]) snprintf(merged, sizeof(merged), "%s, %s", g_chars[new_idx].prompt, work_p);
        else snprintf(merged, sizeof(merged), "%s", g_chars[new_idx].prompt);
        sdxl_clean_prompt_dupes(merged, g_job.prompt, sizeof(g_job.prompt));
        if (g_chars[new_idx].negative[0]) {
            char mneg[512];
            if (g_job.negative[0]) snprintf(mneg, sizeof(mneg), "%s, %s", g_job.negative, g_chars[new_idx].negative);
            else snprintf(mneg, sizeof(mneg), "%s", g_chars[new_idx].negative);
            sdxl_clean_prompt_dupes(mneg, g_job.negative, sizeof(g_job.negative));
        }
    } else {
        sdxl_clean_prompt_dupes(work_p, g_job.prompt, sizeof(g_job.prompt));
    }
    g_st.settings_seq++;
    http_srv_unlock();
}

static void format_a1111_params(char* out, size_t max, const char* prompt, const char* neg,
                                uint32_t steps, const char* sampler, float cfg, uint64_t seed,
                                uint32_t w, uint32_t h, const char* model, const char* lora_spec, int upscaled) {
    const char* up_str = upscaled == 3 ? ", Hires upscaler: RealESRGAN Anime v3 (4K UHD)"
                       : (upscaled > 0 ? ", Hires upscaler: RealESRGAN Anime v3" : "");
    char l_str[180] = {0};
    if (lora_spec && lora_spec[0]) snprintf(l_str, sizeof(l_str), ", Lora: \"%s\"", lora_spec);
    snprintf(out, max,
             "%s\nNegative prompt: %s\nSteps: %u, Sampler: %s, CFG scale: %.1f, Seed: %llu, Size: %ux%u, Model: %s, Clip skip: 2%s%s, Generator: GarlicDiffusion PS4 (GPLv3)",
             prompt ? prompt : "", neg ? neg : "", steps, (sampler && sampler[0]) ? sampler : "Euler a",
             (double)cfg, (unsigned long long)seed, w, h, (model && model[0]) ? model : "GarlicDiffusion",
             l_str, up_str);
}

static int is_starter_model_id(const char* id, int neo, const char** badge_out) {
    if (!id) return 0;
    if (neo) {
        if (strcmp(id, "WAI_Illustrious_v17_dmd2") == 0) {
            if (badge_out) *badge_out = "★ #1 STARTER FOR PS4 PRO · ";
            return 2;
        }
        if (strcmp(id, "Hassaku_XL_v31_dmd2") == 0 || strcmp(id, "MeinaMix_V11_8step") == 0) {
            if (badge_out) *badge_out = "★ RECOMMENDED STARTER · ";
            return 1;
        }
    } else {
        if (strcmp(id, "MeinaMix_V11_8step") == 0) {
            if (badge_out) *badge_out = "★ #1 STARTER FOR BASE PS4 · ";
            return 2;
        }
        if (strcmp(id, "Counterfeit_V25") == 0 || strcmp(id, "GhostMix_V2") == 0) {
            if (badge_out) *badge_out = "★ RECOMMENDED STARTER · ";
            return 1;
        }
    }
    if (badge_out) *badge_out = "";
    return 0;
}

static int lora_arch_compat(const char* lora_arch, int model_is_sd15) {
    if (!lora_arch || !lora_arch[0] || strcmp(lora_arch, "any") == 0) return 1;
    return model_is_sd15 ? (strcmp(lora_arch, "sd15") == 0) : (strcmp(lora_arch, "sdxl") == 0);
}

static int lora_is_installed_cat(const CatalogLora* cl) {
    for (int i = 0; i < g_num_loras; i++) {
        if (strcmp(g_loras[i].filename, cl->filename) == 0 || strcmp(g_loras[i].name, cl->id) == 0) return 1;
    }
    return 0;
}

static void models_mgr_refresh(void) {
    g_inst_cnt = model_dl_list_installed(g_inst, MODEL_DL_MAX_INSTALLED);
    g_cat_cnt = model_dl_get_catalog(g_cat, MODEL_DL_MAX_CATALOG);
    g_num_loras = lora_list_installed(g_loras, LORA_MAX_INSTALLED);
    g_lcat_cnt = lora_get_catalog(g_lcat, LORA_MAX_CATALOG);
    int total = g_mgr_tab ? (g_lora_sub ? g_lcat_cnt : g_num_loras) : (g_inst_cnt + g_cat_cnt);
    if (g_mdl_sel >= total) g_mdl_sel = total > 0 ? total - 1 : 0;
    g_mdl_del_arm = 0;
}

static void open_starter_models_mgr(int neo) {
    g_in_models = 1;
    g_mgr_tab = 0;
    models_mgr_refresh();
    g_mdl_sel = 0;
    if (g_inst_cnt == 0 && g_cat_cnt > 0) {
        for (int i = 0; i < g_cat_cnt; i++) {
            if (is_starter_model_id(g_cat[i].id, neo, NULL) == 2) {
                g_mdl_sel = i;
                break;
            }
        }
    }
}

static const uint32_t RES_LIST[][2] = {
    {512, 512}, {512, 768}, {768, 512}, {768, 768}, {768, 1024},
    {896, 512}, {1024, 576}, {1152, 640}, {1280, 704},
    {832, 1216}, {1216, 832}, {1024, 1024},
    {1024, 1344}, {1344, 1024}
};
static const int RES_COUNT = (int)(sizeof(RES_LIST) / sizeof(RES_LIST[0]));

static void sync_res_idx(uint32_t w, uint32_t h) {
    for (int i = 0; i < RES_COUNT; i++) {
        if (RES_LIST[i][0] == w && RES_LIST[i][1] == h) { res_idx = i; break; }
    }
}

static void apply_tv_preset(int idx) {
    if (idx < 0 || idx >= g_num_presets) return;
    const UserPreset* p = &g_presets[idx];
    http_srv_lock();
    int in_easy = (g_job.ui_mode == 0);
    if (!in_easy) char_idx = -1;
    sdxl_clean_prompt_dupes(p->prompt, g_job.prompt, sizeof(g_job.prompt));
    if (p->negative[0]) sdxl_clean_prompt_dupes(p->negative, g_job.negative, sizeof(g_job.negative));
    int is15 = g_st.is_sd15, neo = g_st.neo, has_up = g_st.has_upscaler;
    int land = (p->width > p->height);
    int sq = (p->width == p->height);
    if (p->is_speed && g_st.speed_steps > 0) {
        g_job.steps = g_st.speed_steps;
        g_job.cfg = g_st.speed_cfg;
        g_job.sampler = g_st.speed_sampler;
        if (land) {
            g_job.width = is15 ? 896u : 1024u;
            g_job.height = is15 ? 512u : 576u;
            g_job.upscale = has_up ? p->upscale : 0;
        } else {
            g_job.width = g_st.speed_w;
            g_job.height = g_st.speed_h;
            g_job.upscale = has_up ? g_st.speed_upscale : 0;
        }
    } else if (p->model[0] && strcmp(p->model, g_st.model) == 0) {
        g_job.width = p->width;
        g_job.height = p->height;
        g_job.steps = p->steps;
        g_job.cfg = p->cfg;
        g_job.sampler = strstr(p->sampler, "DPM") ? 3 : ((strcmp(p->sampler, "LCM") == 0) ? 2 : (strcmp(p->sampler, "Euler") == 0 ? 1 : 0));
        g_job.upscale = has_up ? p->upscale : 0;
    } else {
        if (is15) {
            g_job.width = sq ? 512u : (land ? 768u : 512u);
            g_job.height = sq ? 512u : (land ? 512u : 768u);
        } else {
            uint32_t pw = g_st.preset_w ? g_st.preset_w : 832u;
            uint32_t ph = g_st.preset_h ? g_st.preset_h : 1216u;
            g_job.width = sq ? (neo ? 1024u : 768u) : (land ? ph : pw);
            g_job.height = sq ? (neo ? 1024u : 768u) : (land ? pw : ph);
        }
    }
    uint32_t nw = g_job.width, nh = g_job.height;
    g_st.settings_seq++;
    http_srv_unlock();
    sync_res_idx(nw, nh);
}

static int ci_sub_match(const char* hay, const char* needle) {
    if (!hay || !needle || !needle[0]) return 0;
    size_t nl = strlen(needle);
    for (; *hay; hay++) {
        size_t i = 0;
        while (i < nl && hay[i]) {
            char a = hay[i], b = needle[i];
            if (a >= 'A' && a <= 'Z') a += 32;
            if (b >= 'A' && b <= 'Z') b += 32;
            if (a != b) break;
            i++;
        }
        if (i == nl) return 1;
    }
    return 0;
}

// Strip conflicting preset appearance/framing/LoRA tags when combining Character + Scene in Easy Mode
static void clean_easy_scene_tags(const char* src, int has_char, int is_anim, int is15, char* out, size_t cap) {
    out[0] = 0;
    if (!src || !src[0]) return;
    char no_lora[1024] = {0};
    size_t np = 0;
    const char* s = src;
    while (*s && np + 2 < sizeof(no_lora)) {
        if (*s == '<' && strncasecmp(s, "<lora:", 6) == 0) {
            const char* gt = strchr(s, '>');
            if (gt) {
                char ltag[160];
                size_t ll = (size_t)(gt - s + 1);
                if (ll >= sizeof(ltag)) ll = sizeof(ltag) - 1;
                memcpy(ltag, s, ll); ltag[ll] = 0;
                int drop_l = is15;
                if (has_char && (ci_sub_match(ltag, "miku") || ci_sub_match(ltag, "hifumi") || ci_sub_match(ltag, "takimoto")))
                    drop_l = 1;
                if (is_anim && (ci_sub_match(ltag, "better_eyes") || ci_sub_match(ltag, "civittop") ||
                                ci_sub_match(ltag, "civitai") || ci_sub_match(ltag, "illust")))
                    drop_l = 1;
                if (!drop_l && np + ll + 2 < sizeof(no_lora)) {
                    memcpy(no_lora + np, ltag, ll);
                    np += ll;
                    no_lora[np++] = ' ';
                }
                s = gt + 1;
                continue;
            }
        }
        no_lora[np++] = *s++;
    }
    no_lora[np] = 0;

    static const char* const FRAME_DROP[] = {
        "close-up portrait", "face focus", "upper body", "cowboy shot",
        "full body", "head to toe", "footwear", "wide angle shot", "expansive scenery"
    };
    static const char* const CHAR_CONFLICT[] = {
        "1girl", "1boy", "solo", "hatsune miku", "hatsune_miku", "hifumi takimoto", "takimoto hifumi",
        "silver hair", "white hair", "blonde hair", "pink hair", "aqua hair", "blue hair", "black hair",
        "brown hair", "purple hair", "red hair", "green hair", "twintails", "twin tails", "ponytail",
        "blue eyes", "red eyes", "crimson eyes", "golden eyes", "yellow eyes", "green eyes", "emerald eyes",
        "purple eyes", "violet eyes", "aqua eyes", "amber eyes", "heterochromia"
    };
    const char* p = no_lora;
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        const char* cm = strchr(p, ',');
        size_t tl = cm ? (size_t)(cm - p) : strlen(p);
        while (tl > 0 && p[tl - 1] == ' ') tl--;
        char tok[180];
        if (tl >= sizeof(tok)) tl = sizeof(tok) - 1;
        memcpy(tok, p, tl);
        tok[tl] = 0;
        int drop = 0;
        for (size_t k = 0; k < sizeof(FRAME_DROP) / sizeof(FRAME_DROP[0]); k++) {
            if (strcasecmp(tok, FRAME_DROP[k]) == 0) { drop = 1; break; }
        }
        if (!drop && has_char) {
            for (size_t k = 0; k < sizeof(CHAR_CONFLICT) / sizeof(CHAR_CONFLICT[0]); k++) {
                if (ci_sub_match(tok, CHAR_CONFLICT[k])) { drop = 1; break; }
            }
        }
        if (!drop && is_anim && (strcasecmp(tok, "better_eyes") == 0 || strcasecmp(tok, "civittop50anime") == 0))
            drop = 1;
        if (!drop && tok[0]) {
            if (out[0]) strncat(out, ", ", cap - strlen(out) - 1);
            strncat(out, tok, cap - strlen(out) - 1);
        }
        p = cm ? (cm + 1) : (p + tl);
    }
}

static void apply_tv_easy_framing_and_quality(void) {
    static const char* const F_TAGS[4] = {
        "close-up portrait, face focus, detailed eyes, catchlights",
        "upper body, cowboy shot, detailed hands",
        "full body, head to toe, standing, footwear",
        "wide angle shot, expansive scenery, cinematic composition"
    };
    http_srv_lock();
    int is15 = g_st.is_sd15, neo = g_st.neo, has_up = g_st.has_upscaler;
    int is_anim = ci_sub_match(g_st.model, "animagine");
    int has_char = (char_idx >= 0 && char_idx < g_num_chars && g_chars[char_idx].prompt[0]);
    const char* raw_scene = (preset_idx >= 0 && preset_idx < g_num_presets)
        ? g_presets[preset_idx].prompt
        : g_easy_scene;

    char clean_sc[768] = {0};
    clean_easy_scene_tags(raw_scene, has_char, is_anim, is15, clean_sc, sizeof(clean_sc));

    int fi = (g_easy_framing >= 0 && g_easy_framing <= 3) ? g_easy_framing : 2;
    char merged[1024] = {0};
    snprintf(merged, sizeof(merged), "%s", F_TAGS[fi]);
    if (has_char) {
        strncat(merged, ", ", sizeof(merged) - strlen(merged) - 1);
        strncat(merged, g_chars[char_idx].prompt, sizeof(merged) - strlen(merged) - 1);
    }
    if (clean_sc[0]) {
        strncat(merged, ", ", sizeof(merged) - strlen(merged) - 1);
        strncat(merged, clean_sc, sizeof(merged) - strlen(merged) - 1);
    } else if (!has_char) {
        strncat(merged, ", 1girl, solo, silver hair, blue eyes, gentle smile, cherry blossom petals, golden hour sunlight",
                sizeof(merged) - strlen(merged) - 1);
    }
    sdxl_clean_prompt_dupes(merged, g_job.prompt, sizeof(g_job.prompt));

    // Clear stale single-LoRA selector in Easy Mode so mismatched LoRAs never corrupt Animagine / SD 1.5
    g_job.lora[0] = 0;

    if (fi == 2 && !strstr(g_job.negative, "cropped legs")) {
        char mneg[512];
        snprintf(mneg, sizeof(mneg), "%s%scropped legs, feet out of frame",
                 g_job.negative, g_job.negative[0] ? ", " : "");
        sdxl_clean_prompt_dupes(mneg, g_job.negative, sizeof(g_job.negative));
    }

    int qi = (g_easy_quality >= 0 && g_easy_quality <= 2) ? g_easy_quality : 1;
    g_job.auto_quality = 1;
    g_st.auto_quality = 1;
    if (qi == 0 && g_st.speed_steps > 0) {
        g_job.steps = g_st.speed_steps;
        g_job.cfg = g_st.speed_cfg;
        g_job.sampler = g_st.speed_sampler;
        g_job.upscale = 0;
        if (fi == 3) { g_job.width = is15 ? 896u : 1024u; g_job.height = is15 ? 512u : 576u; }
        else { g_job.width = g_st.speed_w; g_job.height = g_st.speed_h; }
    } else if (qi == 2) {
        if (g_st.preset_steps > 0) {
            g_job.steps = g_st.preset_steps;
            g_job.cfg = g_st.preset_cfg;
            g_job.sampler = g_st.preset_sampler;
        }
        g_job.upscale = has_up ? 1 : 0;
        if (fi == 3) { g_job.width = is15 ? 896u : 1280u; g_job.height = is15 ? 512u : 704u; }
        else { g_job.width = is15 ? 512u : (neo ? 832u : 768u); g_job.height = is15 ? 768u : (neo ? 1216u : 1024u); }
    } else {
        if (g_st.preset_steps > 0) {
            g_job.steps = g_st.preset_steps;
            g_job.cfg = g_st.preset_cfg;
            g_job.sampler = g_st.preset_sampler;
            g_job.upscale = has_up ? g_st.preset_upscale : 0;
        }
        if (fi == 3) {
            g_job.width = is15 ? 896u : 1280u;
            g_job.height = is15 ? 512u : 704u;
            if (has_up && g_job.upscale == 0) g_job.upscale = 1;
        } else {
            g_job.width = g_st.preset_w ? g_st.preset_w : (is15 ? 512u : 832u);
            g_job.height = g_st.preset_h ? g_st.preset_h : (is15 ? 768u : 1216u);
        }
    }
    uint32_t nw = g_job.width, nh = g_job.height;
    g_st.settings_seq++;
    http_srv_unlock();
    sync_res_idx(nw, nh);
}

// Studio menu state (arrow navigation, dropdown modal, slider adjust with revert)
static int sm_focus = ROW_MODEL;
static int sm_adjust = 0;
static uint32_t sm_saved_steps = 8;
static float sm_saved_cfg = 1.4f;
static int sm_dd_open = 0, sm_dd_row = 0, sm_dd_sel = 0, sm_dd_top = 0, sm_dd_n = 0, sm_dd_cur = -1;
static int sm_dd_models_n = 0;
static char sm_dd_models[DIFF_MAX_MODELS][64];
static uint64_t sm_last_seed = 0;

static void sm_dd_clamp(void) {
    if (sm_dd_n <= 0) { sm_dd_sel = 0; sm_dd_top = 0; return; }
    if (sm_dd_sel < 0) sm_dd_sel = sm_dd_n - 1;
    if (sm_dd_sel >= sm_dd_n) sm_dd_sel = 0;
    if (sm_dd_sel < sm_dd_top) sm_dd_top = sm_dd_sel;
    if (sm_dd_sel >= sm_dd_top + SM_DD_VIS) sm_dd_top = sm_dd_sel - SM_DD_VIS + 1;
    int max_top = sm_dd_n > SM_DD_VIS ? (sm_dd_n - SM_DD_VIS) : 0;
    if (sm_dd_top > max_top) sm_dd_top = max_top;
    if (sm_dd_top < 0) sm_dd_top = 0;
}

static const uint32_t BATCH_LIST[] = {1, 2, 3, 4, 5, 8, 10, 15, 20, 50, 100};
static const int BATCH_COUNT = (int)(sizeof(BATCH_LIST) / sizeof(BATCH_LIST[0]));
static int tv_batch_opts_count(int nsfw_ok) { return nsfw_ok ? 13 : 11; }

static void sm_dd_item_label(void* ctx, int row, int i, char* out, size_t cap) {
    (void)ctx;
    switch (row) {
    case ROW_MODEL:
        if (i < sm_dd_models_n) snprintf(out, cap, "%s", sm_dd_models[i]);
        else snprintf(out, cap, "+ Download / Manage Models (Hugging Face)...");
        break;
    case ROW_LORA:
        if (i == 0) snprintf(out, cap, "Off (No LoRA)");
        else if (i - 1 < g_num_loras)
            snprintf(out, cap, "%s   [%s · %u MiB]", g_loras[i - 1].name,
                     strcmp(g_loras[i - 1].arch, "sd15") == 0 ? "SD 1.5" : "SDXL", g_loras[i - 1].size_mb);
        break;
    case ROW_PRESET: {
        int uim_p = 1;
        http_srv_lock(); uim_p = g_job.ui_mode; http_srv_unlock();
        if (i == 0) snprintf(out, cap, uim_p == 0 ? "Clear Scene Preset (Use Character / Custom Scene)" : "+ Save Current Prompt as Preset...");
        else if (i - 1 >= 0 && i - 1 < g_num_presets) snprintf(out, cap, "%s", g_presets[i - 1].name);
        break;
    }
    case ROW_CHAR:
        if (i == 0) snprintf(out, cap, "Off (No Character Overlay)");
        else if (i - 1 < g_num_chars) snprintf(out, cap, "%s  ·  %s", g_chars[i - 1].name, g_chars[i - 1].series);
        break;
    case ROW_STYLE:
        if (i == 0) snprintf(out, cap, "Off (No Art Style)");
        else if (i - 1 < g_num_styles) snprintf(out, cap, "%s", g_styles[i - 1].name);
        break;
    case ROW_RES:
        if (i >= 0 && i < RES_COUNT) sm_res_label(RES_LIST[i][0], RES_LIST[i][1], out, cap);
        break;
    case ROW_SAMPLER:
        if (i >= 0 && i < 4) snprintf(out, cap, "%s", SM_SAMPLER[i]);
        break;
    case ROW_UPSCALE:
        if (i >= 0 && i < 4) snprintf(out, cap, "%s", SM_UPSCALE[i]);
        break;
    case ROW_POWER: {
        int neo_p = 1;
        http_srv_lock(); neo_p = g_st.neo; http_srv_unlock();
        sm_power_label(3 - i, neo_p, out, cap);
        break;
    }
    case ROW_SEED:
        if (i == 0) snprintf(out, cap, "Random each time");
        else if (i == 1) {
            if (sm_last_seed) snprintf(out, cap, "Lock last seed (%llu)", (unsigned long long)sm_last_seed);
            else snprintf(out, cap, "Lock seed 42");
        } else {
            snprintf(out, cap, "Enter custom seed number...");
        }
        break;
    case ROW_BATCH: {
        int sd = 0, aq = 1, ns = 0;
        uint32_t rmask = 0;
        char bfld[64] = {0};
        http_srv_lock();
        sd = g_job.shutdown_after;
        aq = g_job.auto_quality;
        ns = g_job.nsfw_allow;
        rmask = g_job.batch_rand_mask;
        snprintf(bfld, sizeof(bfld), "%s", g_job.batch_folder);
        http_srv_unlock();
        int n_rnd = ns ? 9 : 7;
        int base = 1 + n_rnd;
        int tot_opts = tv_batch_opts_count(ns);
        static const char* const RLBL[9] = {
            "Pose (85 anime poses)",
            "Expression (75 facial expressions)",
            "Scene / Background (120 locations)",
            "Lighting (70 lighting moods)",
            "Outfit (80 outfits)",
            "Details (60 micro-details)",
            "Camera Framing (35 angles)",
            "Ecchi 500 Pool (Gravure / Alluring)",
            "Explicit Cog-Test Pool"
        };
        if (i == 0) {
            snprintf(out, cap, "Randomizer Variety Mix: %s   (press X to toggle / clear)", rmask ? "ACTIVE (Clear All)" : "OFF (Enable Mix)");
        } else if (i >= 1 && i <= n_rnd) {
            uint32_t bit = 1u << (i - 1);
            snprintf(out, cap, "  [%c] Randomize %s", (rmask & bit) ? 'x' : ' ', RLBL[i - 1]);
        } else if (i == base) {
            snprintf(out, cap, "Save Batch into Folder: %s   (press X to edit)", bfld[0] ? bfld : "Unsorted (Root)");
        } else if (i == base + 1) {
            snprintf(out, cap, "Auto-Quality & Hand/Eye Shield: %s", aq ? "ON (Recommended)" : "OFF");
        } else if (i == base + 2) {
            snprintf(out, cap, "Shut down PS4 when batch finishes: %s", sd ? "ON" : "OFF");
        } else if (i - tot_opts >= 0 && i - tot_opts < BATCH_COUNT) {
            uint32_t b = BATCH_LIST[i - tot_opts];
            if (b == 1) snprintf(out, cap, "1 image  ·  single");
            else snprintf(out, cap, "%u images  ·  sequential batch", b);
        }
        break;
    }
    default:
        out[0] = 0;
        break;
    }
}

static float sm_step_cfg(float cur, int dir) {
    float v;
    if (dir > 0) {
        if (cur < 1.99f) v = cur + 0.2f;
        else v = cur + 0.5f;
    } else {
        if (cur <= 2.01f) v = cur - 0.2f;
        else v = cur - 0.5f;
    }
    if (v < SM_CFG_MIN) v = SM_CFG_MIN;
    if (v > SM_CFG_MAX) v = SM_CFG_MAX;
    return floorf(v * 10.0f + 0.5f) * 0.1f;
}

static void blit_rgb_fit(const uint8_t* rgb, uint32_t rw, uint32_t rh, int px, int py, int pw, int ph) {
    float sx = (float)pw / (float)rw;
    float sy = (float)ph / (float)rh;
    float sc = sx < sy ? sx : sy;
    int dw = (int)(rw * sc);
    int dh = (int)(rh * sc);
    int ox = px + (pw - dw) / 2;
    int oy = py + (ph - dh) / 2;
    for (int y = 0; y < dh; y++) {
        int src_y = (int)((float)y / sc);
        if (src_y >= (int)rh) src_y = (int)rh - 1;
        uint32_t* dst_row = back + (size_t)(oy + y) * SCREEN_W + ox;
        const uint8_t* src_row = rgb + (size_t)src_y * rw * 3;
        for (int x = 0; x < dw; x++) {
            int src_x = (int)((float)x / sc);
            if (src_x >= (int)rw) src_x = (int)rw - 1;
            const uint8_t* p = src_row + src_x * 3;
            dst_row[x] = 0x80000000u | ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2];
        }
    }
}

static void gal_load_current(void) {
    free(g_gal_rgb);
    g_gal_rgb = NULL; g_gal_w = 0; g_gal_h = 0;
    g_del_arm = 0;
    if (g_gal_count <= 0) return;
    if (g_gal_idx < 0) g_gal_idx = 0;
    if (g_gal_idx >= g_gal_count) g_gal_idx = g_gal_count - 1;
    size_t psz = 0;
    uint8_t* png = gallery_load_png(g_gal[g_gal_idx].id, &psz);
    if (png) {
        g_gal_rgb = png_decode_rgb(png, psz, &g_gal_w, &g_gal_h);
        free(png);
    }
}

static void gal_refresh(void) {
    g_num_folders = gallery_list_folders(g_folders, GALLERY_MAX_FOLDERS, NULL, NULL, NULL);
    int star_only = (strcmp(g_gal_folder, "__starred__") == 0);
    const char* ff = star_only ? NULL : (g_gal_folder[0] ? g_gal_folder : NULL);
    g_gal_count = gallery_list_page_ex(g_gal, 0, GALLERY_TV_MAX, NULL, ff, star_only, NULL);
    if (g_gal_idx >= g_gal_count) g_gal_idx = g_gal_count > 0 ? g_gal_count - 1 : 0;
    if (g_in_gallery) gal_load_current();
}

static void draw_ui(void) {
    AppUpdateStatus upd;
    app_update_get_status(&upd);

    http_srv_lock();
    char cur_m[64], stg[128], err[256], pr[1024];
    float prog = g_st.progress, eta_s = g_st.eta_sec, elap_s = g_st.elapsed_sec;
    int busy = g_st.busy, eco = g_job.eco_mode, ups = g_job.upscale;
    int neo_v = g_st.neo, has_up_v = g_st.has_upscaler;
    int uim_v = g_job.ui_mode, nsfw_c = g_job.nsfw_allow;
    uint32_t vram_u = g_st.gpu_used_mb, vram_t = g_st.gpu_total_mb;
    double lsec = g_st.last_sec, tf = g_st.eff_tflops;
    uint32_t w = g_job.width, h = g_job.height, steps = g_job.steps;
    float cfg = g_job.cfg;
    snprintf(cur_m, sizeof(cur_m), "%s", g_st.model);
    snprintf(stg, sizeof(stg), "%s", g_st.stage);
    snprintf(err, sizeof(err), "%s", g_st.error);
    snprintf(pr, sizeof(pr), "%s", g_job.prompt);
    if (g_prev_seq != ui_prev_seq && g_prev_rgb && g_prev_w > 0 && g_prev_h > 0) {
        size_t psz = (size_t)g_prev_w * g_prev_h * 3;
        uint8_t* nb = (uint8_t*)malloc(psz);
        if (nb) {
            memcpy(nb, g_prev_rgb, psz);
            free(ui_prev_rgb);
            ui_prev_rgb = nb;
            ui_prev_w = g_prev_w;
            ui_prev_h = g_prev_h;
            ui_prev_seq = g_prev_seq;
        }
    }
    http_srv_unlock();

    if (eco < 0 || eco > 3) eco = 2;
    if (ups < 0 || ups > 3) ups = 0;

    // ---- 0. TWO-CHOICE FIRST-LAUNCH WELCOME SCREEN (Easy Mode vs Advanced Studio) ----
    if (uim_v < 0) {
        sm_draw_welcome(back, g_welcome_sel, neo_v, cur_m, busy, vram_u, vram_t, http_srv_address());
        present();
        return;
    }

    // ---- 1. CLEAN, SCANNABLE CREDITS, OTA UPDATES & CONTENT FILTER ----
    if (g_in_credits) {
        g_draw_bg(back);
        g_draw_header(back, "Credits & App Settings", cur_m, busy, vram_u, vram_t, http_srv_address());

        // Clean 2-line top banner
        g_rrect(back, 56, 112, 1808, 104, 18, C_SURFACE, 235);
        g_rrect_line(back, 56, 112, 1808, 104, 18, 1, C_LINE_HI, 255);
        g_rrect(back, 72, 128, 4, 72, 2, C_ACCENT_HI, 220);
        g_draw_str(back, &g_fbold, 94, 128, "GarlicDiffusion v1.15  ·  GNU GPLv3  ·  Built upon Garlic LLM", 0xFFFFFF);
        g_draw_str(back, &g_fsmall, 94, 168,
                   "Use Quiet / Silent mode for long sessions & keep PS4 vents clear. Provided 'AS IS' without warranty.",
                   0xF2A65A);

        // Left Card: 4 Spacious Community Cards
        g_rrect(back, 56, 236, 884, 734, 18, C_SURFACE, 235);
        g_rrect_line(back, 56, 236, 884, 734, 18, 1, C_LINE_HI, 255);
        g_draw_str(back, &g_fbold, 86, 260, "Project & Community", 0xFFFFFF);

        struct {
            const char* label;
            const char* url;
            const char* note;
            uint32_t col;
        } links[4] = {
            { "Author & Support (Ko-fi)", "ko-fi.com/djtomato", "DJTomato  ·  PS4 SD1.5, SDXL, LoRA & Real-ESRGAN", C_TRIANGLE },
            { "Discord Community", "discord.gg/Gbyq3kpGjc", "Share prompts, request checkpoints & Discord bots", C_CROSS },
            { "GitHub & Model Hub", "github.com/DJTOMATO/GarlicDiffusion", "huggingface.co/DJTomato/ps4-diffusion-models", C_ACCENT_HI },
            { "Upstream Foundation", "garlicllm.dev", "Leonardo Alegria  ·  Bare-metal PS4 GCN compute", C_GREEN },
        };
        for (int i = 0; i < 4; i++) {
            int ly = 314 + i * 156;
            g_rrect(back, 84, ly, 828, 138, 14, C_PANEL, 230);
            g_rrect_line(back, 84, ly, 828, 138, 14, 1, C_LINE, 255);
            g_rrect(back, 100, ly + 22, 4, 94, 2, links[i].col, 230);
            g_draw_str(back, &g_fbold, 120, ly + 20, links[i].label, 0xFFFFFF);
            g_draw_str(back, &g_fsmall, 120, ly + 58, links[i].url, links[i].col);
            g_draw_str(back, &g_fsmall, 120, ly + 92, links[i].note, C_TEXT_MUTED);
        }

        // Right Top Card: Concise License Summary (Zero Overlap)
        g_rrect(back, 980, 236, 884, 412, 18, C_SURFACE, 235);
        g_rrect_line(back, 980, 236, 884, 412, 18, 1, C_LINE_HI, 255);
        g_draw_str(back, &g_fbold, 1010, 260, "Open-Source Licenses", 0xFFFFFF);

        static const char* const LICS[5][2] = {
            { "GarlicDiffusion & Garlic LLM", "GNU GPLv3" },
            { "SDXL & Illustrious Checkpoints", "Fair AI 1.0-SD" },
            { "SD 1.5 Checkpoints", "Open RAIL-M" },
            { "DMD2, LoRAs & Real-ESRGAN", "MIT / BSD-3" },
            { "OpenOrbis & Selawik Font", "GPLv3 / OFL 1.1" },
        };
        for (int i = 0; i < 5; i++) {
            int ry = 314 + i * 62;
            g_rrect(back, 1008, ry, 828, 50, 12, C_PANEL, 220);
            g_draw_str(back, &g_fsmall, 1030, ry + 13, LICS[i][0], C_TEXT);
            int lw = g_str_width(&g_fsmall, LICS[i][1]);
            g_draw_str(back, &g_fsmall, 1814 - lw, ry + 13, LICS[i][1], C_ACCENT_HI);
        }

        // Right Bottom Card: Updates & Content Filter (Spacious & Zero Overflow)
        g_rrect(back, 980, 668, 884, 302, 18, C_SURFACE, 235);
        g_rrect_line(back, 980, 668, 884, 302, 18, 1, upd.update_available ? 0xF2A65A : C_LINE_HI, 255);
        g_draw_str(back, &g_fbold, 1010, 690, "App Updates & Content Filter", 0xFFFFFF);

        // Row A: OTA Status
        g_rrect(back, 1008, 738, 828, 68, 12, C_PANEL, 235);
        char ota_raw[180], ota_fit[180];
        uint32_t ota_col = C_ACCENT_HI;
        if (upd.pkg_ready) {
            snprintf(ota_raw, sizeof(ota_raw), "PKG Ready in /data/pkg  ·  Press R3 to Close App & Install!");
            ota_col = C_GREEN;
        } else if (upd.update_available) {
            snprintf(ota_raw, sizeof(ota_raw), "Update v%s Available!  ·  Press L3 to Download to /data/pkg", upd.remote_ver);
            ota_col = 0xF2A65A;
        } else {
            snprintf(ota_raw, sizeof(ota_raw), "Version v%s Installed  ·  Press L3 to Check GitHub", GD_APP_VERSION);
        }
        sm_fit(&g_fsmall, ota_raw, 784, ota_fit, sizeof(ota_fit));
        g_draw_str(back, &g_fsmall, 1030, 760, ota_fit, ota_col);

        // Row B: Discreet Content Filter
        g_rrect(back, 1008, 820, 828, 68, 12, C_PANEL, 235);
        char cf_raw[180], cf_fit[180];
        snprintf(cf_raw, sizeof(cf_raw), "Press Triangle  ·  Content Filter: %s",
                 nsfw_c ? "NSFW Unlocked (Adult Presets Visible)" : "SFW Guard ON (Default)");
        sm_fit(&g_fsmall, cf_raw, 784, cf_fit, sizeof(cf_fit));
        g_draw_str(back, &g_fsmall, 1030, 842, cf_fit, nsfw_c ? 0xFF9EC4 : C_GREEN);

        char note_fit[180];
        sm_fit(&g_fsmall, toast[0] ? toast : "Updates never touch your models, LoRAs or Gallery in /data/garlic-diffusion.",
               820, note_fit, sizeof(note_fit));
        g_draw_str(back, &g_fsmall, 1012, 912, note_fit, toast[0] ? C_GREEN : C_TEXT_MUTED);

        const GHint fhints[] = {
            { GBTN_CIRCLE, "Back" },
            { GBTN_TRIANGLE, nsfw_c ? "Set SFW" : "Allow NSFW" },
            { GBTN_L3, (upd.update_available && !upd.pkg_ready) ? "Download PKG" : "Check Update" },
            { GBTN_R3, upd.pkg_ready ? "Close & Install" : "Models" },
            { GBTN_SQUARE, "Gallery" },
        };
        g_draw_footer(back, fhints, 5, "Unofficial PS4 Homebrew · GPLv3");
        present();
        return;
    }

    // ---- 2. MODEL MANAGER & HUGGING FACE DOWNLOADER (Garlic LLM style) ----
    if (g_in_models) {
        g_draw_bg(back);
        char repo[128], rsub[200];
        model_dl_get_repo(repo, sizeof(repo));
        snprintf(rsub, sizeof(rsub), "[L1/R1] %s  ·  HF: %s",
                 g_mgr_tab ? "LoRA Manager & Downloader" : "Model Manager & Downloader",
                 repo[0] ? repo : "DJTomato/ps4-diffusion-models");
        g_draw_header(back, rsub, cur_m, busy, vram_u, vram_t, http_srv_address());

        ModelDlStatus dl;
        model_dl_get_status(&dl);
        int y0 = 110;
        if (dl.active || dl.error[0] || dl.stage[0]) {
            g_rrect(back, 56, y0, 1808, 74, 16, C_SURFACE, 240);
            g_rrect_line(back, 56, y0, 1808, 74, 16, 1, C_LINE_HI, 255);
            if (dl.active) {
                float pct = dl.bytes_total ? (100.0f * (float)dl.bytes_done / (float)dl.bytes_total) : 0.0f;
                char eta_dl[40] = {0};
                if (dl.speed_mbs > 0.05 && dl.bytes_total > dl.bytes_done) {
                    float rem_s = (float)((double)(dl.bytes_total - dl.bytes_done) / (dl.speed_mbs * 1048576.0));
                    sm_fmt_eta(rem_s, eta_dl, sizeof(eta_dl));
                }
                char dmsg[256], dmsg_fit[256];
                snprintf(dmsg, sizeof(dmsg), "Downloading %s [%d/%d] %s  ·  %llu / %llu MiB (%.0f%%)  ·  %.1f MB/s%s%s",
                         dl.model_id, dl.file_idx, dl.file_total, dl.cur_file,
                         (unsigned long long)(dl.bytes_done >> 20), (unsigned long long)(dl.bytes_total >> 20),
                         (double)pct, dl.speed_mbs, eta_dl[0] ? "  ·  " : "", eta_dl);
                sm_fit(&g_fsmall, dmsg, 1750, dmsg_fit, sizeof(dmsg_fit));
                g_draw_str(back, &g_fsmall, 80, y0 + 14, dmsg_fit, C_ACCENT_HI);
                g_rrect(back, 80, y0 + 46, 1760, 12, 6, 0x1B3156, 255);
                int bw = (int)(1760.0f * (pct / 100.0f));
                if (bw > 1760) bw = 1760;
                if (bw > 0) g_rrect(back, 80, y0 + 46, bw < 12 ? 12 : bw, 12, 6, C_GREEN, 255);
            } else {
                char st_fit[220];
                sm_fit(&g_fsmall, dl.error[0] ? dl.error : dl.stage, 1740, st_fit, sizeof(st_fit));
                g_draw_str(back, &g_fsmall, 80, y0 + 26, st_fit, dl.error[0] ? C_RED : C_GREEN);
            }
            y0 += 88;
        }

        if (!g_mgr_tab && g_inst_cnt == 0 && !dl.active) {
            char b1[200], b2[200];
            sm_fit(&g_fbold, neo_v ? "First-Time Setup (PS4 Pro): Pick a ★ STARTER Checkpoint below & press X to download!"
                                   : "First-Time Setup (Base PS4): Pick ★ MeinaMix_V11_8step below & press X to download!", 1740, b1, sizeof(b1));
            sm_fit(&g_fsmall, "Your first downloaded model loads onto the PS4 GPU automatically! Press L1/R1 anytime for 0-VRAM LoRAs.", 1740, b2, sizeof(b2));
            g_rrect(back, 56, y0, 1808, 84, 16, 0x0E2444, 245);
            g_rrect_line(back, 56, y0, 1808, 84, 16, 1, 0xF2A65A, 255);
            g_draw_str(back, &g_fbold, 82, y0 + 12, b1, 0xF2A65A);
            g_draw_str(back, &g_fsmall, 82, y0 + 44, b2, C_TEXT);
            y0 += 98;
        }

        int idx = 0;
        if (!g_mgr_tab) {
            if (g_inst_cnt > 0) {
                g_draw_str(back, &g_fbold, 56, y0, "Installed Models on PS4 (/data/garlic-diffusion/models)   ·   Press L1/R1 for LoRAs", 0xFFFFFF);
                y0 += 34;
                for (int i = 0; i < g_inst_cnt && y0 < 520; i++, idx++) {
                    int sel = (idx == g_mdl_sel);
                    int act = (strcmp(g_inst[i].name, cur_m) == 0);
                    g_rrect(back, 56, y0, 1808, 44, 12, sel ? C_SURFACE_HI : C_SURFACE, sel ? 255 : 210);
                    g_rrect_line(back, 56, y0, 1808, 44, 12, 1, sel ? C_ACCENT_HI : C_LINE, 255);
                    if (act || sel) g_circle(back, 80.0f, (float)(y0 + 22), 5.0f, 0.0f, act ? C_GREEN : C_ACCENT_HI, 255);
                    char row[240], rfit[240];
                    snprintf(row, sizeof(row), "%s   [%s · %u MiB]   %s%s",
                             g_inst[i].name, g_inst[i].arch, g_inst[i].size_mb,
                             act ? "ACTIVE · " : "", g_inst[i].detail);
                    sm_fit(&g_fsmall, row, 1720, rfit, sizeof(rfit));
                    g_draw_str(back, &g_fsmall, 100, y0 + 11, rfit, act ? C_GREEN : (g_inst[i].complete ? C_TEXT : C_RED));
                    y0 += 50;
                }
                idx = g_inst_cnt;
                y0 += 8;
            }
            int avail_rows = (935 - (y0 + 34)) / 50;
            if (avail_rows < 3) avail_rows = 3;
            int cat_sel = g_mdl_sel - g_inst_cnt;
            int cat_top = 0;
            if (cat_sel >= avail_rows) cat_top = cat_sel - avail_rows + 1;
            if (cat_top > g_cat_cnt - avail_rows) cat_top = g_cat_cnt > avail_rows ? (g_cat_cnt - avail_rows) : 0;
            char chdr[160];
            snprintf(chdr, sizeof(chdr), "Downloadable Checkpoints (1-Click HTTPS · %d Available · Press L1/R1 for LoRAs)", g_cat_cnt);
            g_draw_str(back, &g_fbold, 56, y0, chdr, 0xFFFFFF);
            y0 += 34;
            for (int i = cat_top; i < g_cat_cnt && y0 < 935; i++) {
                int cidx = g_inst_cnt + i;
                int sel = (cidx == g_mdl_sel);
                const char* sbadge = "";
                int st_tier = is_starter_model_id(g_cat[i].id, neo_v, &sbadge);
                g_rrect(back, 56, y0, 1808, 44, 12, sel ? C_SURFACE_HI : C_PANEL, sel ? 255 : 210);
                g_rrect_line(back, 56, y0, 1808, 44, 12, 1, sel ? C_ACCENT_HI : (st_tier ? C_GREEN : C_LINE), 255);
                char row[256], fit_r[256];
                snprintf(row, sizeof(row), "%s%s   [%s · ~%u MiB]   %s",
                         sbadge, g_cat[i].id, g_cat[i].arch, g_cat[i].size_mb, g_cat[i].desc);
                sm_fit(&g_fsmall, row, 1740, fit_r, sizeof(fit_r));
                g_draw_str(back, &g_fsmall, 84, y0 + 11, fit_r, sel ? 0xFFFFFF : (st_tier ? C_GREEN : C_TEXT_DIM));
                y0 += 50;
            }
        } else {
            char cur_lr[80] = {0};
            int cur_sd15 = 0;
            http_srv_lock();
            snprintf(cur_lr, sizeof(cur_lr), "%s", g_job.lora);
            cur_sd15 = g_st.is_sd15;
            http_srv_unlock();
            const char* act_arch_lbl = cur_sd15 ? "SD 1.5" : "SDXL";

            // Sub-tab bar: [Installed (N)]  [Catalog (M)]  +  Active arch badge  +  Page indicator
            int list_cnt = g_lora_sub ? g_lcat_cnt : g_num_loras;
            int total_pages = list_cnt > 0 ? ((list_cnt + LORA_TV_PAGE_SZ - 1) / LORA_TV_PAGE_SZ) : 1;
            int cur_page = list_cnt > 0 ? (g_mdl_sel / LORA_TV_PAGE_SZ) : 0;
            if (cur_page >= total_pages) cur_page = total_pages - 1;
            int page_start = cur_page * LORA_TV_PAGE_SZ;
            int page_end = page_start + LORA_TV_PAGE_SZ;
            if (page_end > list_cnt) page_end = list_cnt;

            int compat_inst = 0, compat_cat = 0;
            for (int i = 0; i < g_num_loras; i++) if (lora_arch_compat(g_loras[i].arch, cur_sd15)) compat_inst++;
            for (int i = 0; i < g_lcat_cnt; i++) if (lora_arch_compat(g_lcat[i].arch, cur_sd15)) compat_cat++;

            char t_inst[80], t_inst_f[80], t_cat[80], t_cat_f[80], t_info[140];
            snprintf(t_inst, sizeof(t_inst), "Installed (%d · %d %s)", g_num_loras, compat_inst, act_arch_lbl);
            snprintf(t_cat, sizeof(t_cat), "Catalog (%d · %d %s)", g_lcat_cnt, compat_cat, act_arch_lbl);
            sm_fit(&g_fbold, t_inst, 320, t_inst_f, sizeof(t_inst_f));
            sm_fit(&g_fbold, t_cat, 320, t_cat_f, sizeof(t_cat_f));
            g_rrect(back, 56, y0, 356, 40, 12, !g_lora_sub ? C_ACCENT : C_SURFACE, 245);
            g_rrect_line(back, 56, y0, 356, 40, 12, 1, !g_lora_sub ? C_ACCENT_HI : C_LINE, 255);
            g_draw_str(back, &g_fbold, 74, y0 + 10, t_inst_f, !g_lora_sub ? 0xFFFFFF : C_TEXT_DIM);

            g_rrect(back, 424, y0, 356, 40, 12, g_lora_sub ? C_ACCENT : C_SURFACE, 245);
            g_rrect_line(back, 424, y0, 356, 40, 12, 1, g_lora_sub ? C_ACCENT_HI : C_LINE, 255);
            g_draw_str(back, &g_fbold, 442, y0 + 10, t_cat_f, g_lora_sub ? 0xFFFFFF : C_TEXT_DIM);

            snprintf(t_info, sizeof(t_info), "Active: %s  |  Page %d / %d  (L2/R2 Switch · Left/Right Page · Square: DL All %s)%s",
                     act_arch_lbl, cur_page + 1, total_pages, act_arch_lbl,
                     dl.queue_len > 0 ? "  [QUEUED]" : "");
            char t_info_fit[160];
            sm_fit(&g_fsmall, t_info, 1050, t_info_fit, sizeof(t_info_fit));
            g_draw_str(back, &g_fsmall, 800, y0 + 11, t_info_fit, C_ACCENT_HI);
            y0 += 52;

            if (list_cnt == 0) {
                g_rrect(back, 56, y0, 1808, 78, 14, C_SURFACE, 220);
                g_rrect_line(back, 56, y0, 1808, 78, 14, 1, C_LINE, 255);
                g_draw_str(back, &g_fbold, 84, y0 + 16,
                           g_lora_sub ? "No catalog LoRAs available." : "No LoRAs installed on PS4 yet.", C_TEXT);
                g_draw_str(back, &g_fsmall, 84, y0 + 46,
                           "Press [L2/R2] to switch to Download Catalog or [SQUARE] to download all compatible LoRAs!", C_TEXT_DIM);
            } else if (!g_lora_sub) {
                for (int i = page_start; i < page_end && y0 + 78 <= 948; i++) {
                    int sel = (i == g_mdl_sel);
                    int act = (cur_lr[0] && strcmp(g_loras[i].name, cur_lr) == 0);
                    int ok_arch = lora_arch_compat(g_loras[i].arch, cur_sd15);
                    g_rrect(back, 56, y0, 1808, 74, 14, sel ? C_SURFACE_HI : (ok_arch ? C_SURFACE : C_PANEL), sel ? 255 : (ok_arch ? 225 : 175));
                    g_rrect_line(back, 56, y0, 1808, 74, 14, 1, sel ? C_ACCENT_HI : (act ? C_GREEN : (ok_arch ? C_LINE : 0x5A3A22)), 255);
                    if (act || sel) g_circle(back, 80.0f, (float)(y0 + 37), 6.0f, 0.0f, act ? C_GREEN : (ok_arch ? C_ACCENT_HI : C_RED), 255);

                    char l1[220], l1f[220], l2[240], l2f[240];
                    snprintf(l1, sizeof(l1), "%s   [%s · %s]",
                             g_loras[i].name, g_loras[i].arch,
                             act ? "ACTIVE IN STUDIO" : (ok_arch ? "COMPATIBLE" : "INCOMPATIBLE WITH ACTIVE MODEL"));
                    snprintf(l2, sizeof(l2), "%u MiB  ·  Rank %u  ·  %s  ·  %s",
                             g_loras[i].size_mb, g_loras[i].rank,
                             g_loras[i].has_te ? "UNet + CLIP" : "UNet only",
                             !ok_arch ? "Switch model arch before using (blocked)"
                                      : (act ? "Press X to unselect  ·  OPTIONS (x2) to delete"
                                             : "Press X to select in Studio  ·  OPTIONS (x2) to delete"));
                    sm_fit(&g_fbold, l1, 1720, l1f, sizeof(l1f));
                    sm_fit(&g_fsmall, l2, 1720, l2f, sizeof(l2f));
                    g_draw_str(back, &g_fbold, 100, y0 + 12, l1f, act ? C_GREEN : (ok_arch ? 0xFFFFFF : 0xF2A65A));
                    g_draw_str(back, &g_fsmall, 100, y0 + 42, l2f, ok_arch ? C_TEXT_DIM : 0xB88B60);
                    y0 += 82;
                }
            } else {
                for (int i = page_start; i < page_end && y0 + 78 <= 948; i++) {
                    int sel = (i == g_mdl_sel);
                    int ok_arch = lora_arch_compat(g_lcat[i].arch, cur_sd15);
                    int has = lora_is_installed_cat(&g_lcat[i]);
                    int qd = !has && (lora_dl_is_queued(g_lcat[i].filename) || lora_dl_is_queued(g_lcat[i].id));
                    g_rrect(back, 56, y0, 1808, 74, 14, sel ? C_SURFACE_HI : (ok_arch ? C_PANEL : 0x0B1528), sel ? 255 : (ok_arch ? 225 : 170));
                    g_rrect_line(back, 56, y0, 1808, 74, 14, 1, sel ? C_ACCENT_HI : (has ? C_GREEN : (qd ? 0xF2A65A : (ok_arch ? C_LINE : 0x4A3222))), 255);

                    const char* st_tag = has ? "INSTALLED" : (qd ? "QUEUED / DOWNLOADING" : (ok_arch ? "COMPATIBLE · PRESS X TO DOWNLOAD/QUEUE" : "OTHER ARCH"));
                    char l1[240], l1f[240], l2[260], l2f[260];
                    snprintf(l1, sizeof(l1), "%s   [%s · ~%u MiB · rec %.2f · %s]",
                             g_lcat[i].title, g_lcat[i].arch, g_lcat[i].size_mb, (double)g_lcat[i].default_weight, st_tag);
                    snprintf(l2, sizeof(l2), "%s%s%s",
                             g_lcat[i].desc, g_lcat[i].trigger[0] ? "  ·  Trigger: " : "", g_lcat[i].trigger);
                    sm_fit(&g_fbold, l1, 1740, l1f, sizeof(l1f));
                    sm_fit(&g_fsmall, l2, 1740, l2f, sizeof(l2f));
                    g_draw_str(back, &g_fbold, 84, y0 + 12, l1f, has ? C_GREEN : (qd ? 0xF2A65A : (ok_arch ? (sel ? 0xFFFFFF : C_TEXT) : 0xA68A6E)));
                    g_draw_str(back, &g_fsmall, 84, y0 + 42, l2f, ok_arch ? C_TEXT_DIM : 0x7A6A5A);
                    y0 += 82;
                }
            }
        }

        if (toast[0]) g_draw_str(back, &g_fsmall, 56, 956, toast, g_mdl_del_arm ? C_RED : C_GREEN);
        const GHint mhints_mdl[] = {
            { GBTN_UPDOWN, "Select" },
            { GBTN_L1, "LoRAs Tab" },
            { GBTN_CROSS, "Load / Download" },
            { GBTN_TRIANGLE, "Cancel Download" },
            { GBTN_OPTIONS, "Delete (x2)" },
            { GBTN_CIRCLE, "Back" },
        };
        const GHint mhints_lora[] = {
            { GBTN_UPDOWN, "Select" },
            { GBTN_LEFTRIGHT, "Prev/Next Page" },
            { GBTN_L1, "Models" },
            { GBTN_L2, "Installed/Catalog" },
            { GBTN_CROSS, g_lora_sub ? "Download / Queue" : "Use LoRA" },
            { GBTN_SQUARE, "Download All Compatible" },
            { GBTN_TRIANGLE, "Cancel" },
            { GBTN_CIRCLE, "Back" },
        };
        g_draw_footer(back, g_mgr_tab ? mhints_lora : mhints_mdl, g_mgr_tab ? 8 : 6,
                      "Incompatible LoRAs are blocked automatically");
        present();
        return;
    }

    // ---- 3. TV GALLERY VIEWER (Garlic LLM style) ----
    if (g_in_gallery) {
        if (g_gal_full && g_gal_rgb) {
            g_fill(back, 0, 0, SCREEN_W, SCREEN_H, 0x040812, 255);
            blit_rgb_fit(g_gal_rgb, g_gal_w, g_gal_h, 0, 0, SCREEN_W, SCREEN_H);
            present();
            return;
        }
        g_draw_bg(back);
        char gal_sub[128];
        const char* fn_lbl = !g_gal_folder[0] ? "All Images"
            : (strcmp(g_gal_folder, "__starred__") == 0 ? "Starred Favorites"
            : (strcmp(g_gal_folder, "__unsorted__") == 0 ? "Unsorted (Root)" : g_gal_folder));
        snprintf(gal_sub, sizeof(gal_sub), "Gallery  ·  Folder: %s", fn_lbl);
        g_draw_header(back, gal_sub, cur_m, busy, vram_u, vram_t, http_srv_address());
        if (g_gal_count <= 0) {
            g_rrect(back, 56, 120, 740, 220, 18, C_SURFACE, 235);
            g_rrect_line(back, 56, 120, 740, 220, 18, 1, C_LINE_HI, 255);
            g_draw_str(back, &g_fbold, 84, 154, g_gal_folder[0] ? "No images in this folder/filter" : "No saved images yet", 0xFFFFFF);
            g_draw_str(back, &g_fsmall, 84, 196, "Press L1 to switch folder view, or Circle to return to Studio.", C_TEXT_DIM);
            const GHint ehints[] = {{ GBTN_L1, "Switch Folder" }, { GBTN_CIRCLE, "Back to Studio" }};
            g_draw_footer(back, ehints, 2, "");
            present();
            return;
        }
        const GalleryItem* it = &g_gal[g_gal_idx];
        int is_sel = (gal_is_selected(it->id) >= 0);
        int pg_cur = (g_gal_idx / 12) + 1;
        int pg_tot = ((g_gal_count + 11) / 12);
        g_rrect(back, 56, 112, 612, 868, 18, C_SURFACE, 235);
        g_rrect_line(back, 56, 112, 612, 868, 18, is_sel ? 2 : 1, is_sel ? C_GREEN : C_LINE_HI, 255);

        char hdr[128], hdr_fit[128];
        snprintf(hdr, sizeof(hdr), "Artwork %d/%d (Pg %d/%d) · %ux%u%s",
                 g_gal_idx + 1, g_gal_count, pg_cur, pg_tot,
                 it->width, it->height, it->upscaled == 3 ? " 4K" : (it->upscaled ? " HD" : ""));
        sm_fit(&g_fbold, hdr, 556, hdr_fit, sizeof(hdr_fit));
        g_draw_str(back, &g_fbold, 82, 132, hdr_fit, 0xFFFFFF);

        char info[160], info_fit[160];
        snprintf(info, sizeof(info), "%s · %u stp · CFG %.1f · %s",
                 it->model[0] ? it->model : "Model", it->steps, (double)it->cfg, it->sampler);
        sm_fit(&g_fsmall, info, 556, info_fit, sizeof(info_fit));
        g_draw_str(back, &g_fsmall, 82, 168, info_fit, C_ACCENT_HI);
        snprintf(info, sizeof(info), "Seed %llu · %.1fs%s",
                 (unsigned long long)it->seed, it->sec, it->starred ? " · ★ Starred" : "");
        sm_fit(&g_fsmall, info, 556, info_fit, sizeof(info_fit));
        g_draw_str(back, &g_fsmall, 82, 196, info_fit, C_TEXT_DIM);

        // Prompt inside Garlic LLM blue user bubble
        int rad_b[4] = {22, 22, 8, 22};
        g_rrect4(back, 78, 232, 568, 250, rad_b, 0, C_USER, 245);
        g_draw_wrapped(back, &g_fsmall, 98, 248, 528, 26, 0xFFFFFF, it->prompt[0] ? it->prompt : "(no prompt metadata)", 8);

        // Pill 1: Prominent [R1] Add / Move to Folder button with real controller badge!
        g_rrect(back, 78, 492, 568, 52, 14, 0x0F2B48, 248);
        g_rrect_line(back, 78, 492, 568, 52, 14, 1, C_GREEN, 255);
        int r1w = g_draw_btn(back, GBTN_R1, 94, 518);
        char fld_btn_txt[160], fld_btn_fit[160];
        snprintf(fld_btn_txt, sizeof(fld_btn_txt), "Add to Folder   (Now: %s)",
                 it->folder[0] ? it->folder : "Unsorted");
        sm_fit(&g_fsmall, fld_btn_txt, 520 - r1w, fld_btn_fit, sizeof(fld_btn_fit));
        g_draw_str(back, &g_fsmall, 94 + r1w + 12, 505, fld_btn_fit, C_GREEN);

        // Pill 2: Prominent (X) Artwork Actions Menu button!
        g_rrect(back, 78, 552, 568, 52, 14, 0x102344, 248);
        g_rrect_line(back, 78, 552, 568, 52, 14, 1, C_ACCENT_HI, 255);
        int xbw = g_draw_btn(back, GBTN_CROSS, 94, 578);
        char x_fit[160];
        sm_fit(&g_fsmall, "Artwork Actions  (Folder · Zoom · Load Prompt · Star)", 520 - xbw, x_fit, sizeof(x_fit));
        g_draw_str(back, &g_fsmall, 94 + xbw + 12, 565, x_fit, 0xFFFFFF);

        static const char* const UP_MODES[4] = {"", "1080p HD", "2x HD", "4K UHD"};
        g_rrect(back, 78, 614, 568, 92, 14, C_PANEL, 235);
        g_rrect_line(back, 78, 614, 568, 92, 14, 1, g_gal_sel_n > 0 ? C_GREEN : C_LINE, 255);
        char sel_ln1[160], sel_ln2[160], f1[160], f2[160];
        if (g_gal_sel_n > 0) {
            snprintf(sel_ln1, sizeof(sel_ln1), "%sBatch Selected: %d  (R1 Move · OPTIONS Delete)",
                     is_sel ? "[x] " : "", g_gal_sel_n);
        } else {
            snprintf(sel_ln1, sizeof(sel_ln1), "R2: Multi-Select  ·  L1: Filter  ·  OPTIONS: Delete");
        }
        snprintf(sel_ln2, sizeof(sel_ln2), "L3: Upscale -> %s  (L2 cycles target)",
                 UP_MODES[(g_gal_up_mode >= 1 && g_gal_up_mode <= 3) ? g_gal_up_mode : 1]);
        sm_fit(&g_fsmall, sel_ln1, 532, f1, sizeof(f1));
        sm_fit(&g_fsmall, sel_ln2, 532, f2, sizeof(f2));
        g_draw_str(back, &g_fsmall, 96, 626, f1, g_gal_sel_n > 0 ? C_GREEN : C_TEXT);
        g_draw_str(back, &g_fsmall, 96, 662, f2, C_ACCENT_HI);

        if (toast[0]) g_draw_wrapped(back, &g_fsmall, 82, 718, 556, 26, g_del_arm ? C_RED : C_GREEN, toast, 2);

        char nav_ln[128], web_ln[128];
        sm_fit(&g_fsmall, "Up/Down jumps 12 artworks  ·  Square: Fullscreen", 556, nav_ln, sizeof(nav_ln));
        g_draw_str(back, &g_fsmall, 82, 784, nav_ln, C_TEXT_MUTED);
        snprintf(web_ln, sizeof(web_ln), "Web Gallery & ZIP: http://%s", http_srv_address());
        sm_fit(&g_fsmall, web_ln, 556, f1, sizeof(f1));
        g_draw_str(back, &g_fsmall, 82, 814, f1, C_GREEN);

        // Right image showcase card
        g_rrect(back, 692, 112, 1172, 868, 18, C_PANEL, 245);
        g_rrect_line(back, 692, 112, 1172, 868, 18, is_sel ? 2 : 1, is_sel ? C_GREEN : C_LINE_HI, 255);
        if (g_gal_rgb && g_gal_w && g_gal_h) {
            blit_rgb_fit(g_gal_rgb, g_gal_w, g_gal_h, 708, 128, 1140, 836);
        } else {
            g_draw_str(back, &g_fsmall, 1120, 530, "Preview unavailable for this PNG", C_TEXT_MUTED);
        }
        if (is_sel) {
            g_rrect(back, 716, 134, 210, 38, 19, 0x0E3A2A, 240);
            g_rrect_line(back, 716, 134, 210, 38, 19, 2, C_GREEN, 255);
            g_draw_str(back, &g_fbold, 738, 140, "SELECTED", C_GREEN);
        }

        // Artwork Quick-Actions Modal on (X) — makes Add to Folder & every action 100% obvious!
        if (g_gal_amodal) {
            g_fill(back, 0, 0, SCREEN_W, SCREEN_H, 0x020611, 185);
            int mw = 740, mh = 90 + 6 * 54 + 16;
            int mx = (SCREEN_W - mw) / 2, my = (SCREEN_H - mh) / 2 - 12;
            g_rrect(back, mx, my, mw, mh, 22, C_PANEL, 252);
            g_rrect_line(back, mx, my, mw, mh, 22, 2, C_ACCENT_HI, 255);
            g_draw_str(back, &g_fbold, mx + 30, my + 22, "Artwork Actions", 0xFFFFFF);
            char act_items[6][128];
            if (g_gal_sel_n > 0)
                snprintf(act_items[0], sizeof(act_items[0]), "Add / Move %d Selected Images to Folder...", g_gal_sel_n);
            else
                snprintf(act_items[0], sizeof(act_items[0]), "Add / Move to Folder...   (Current: %s)", it->folder[0] ? it->folder : "Unsorted");
            snprintf(act_items[1], sizeof(act_items[1]), "View Fullscreen (1:1 Zoom)");
            snprintf(act_items[2], sizeof(act_items[2]), "Load Prompt & Settings into Studio");
            snprintf(act_items[3], sizeof(act_items[3]), "Re-Render Variation in Studio (New Seed)");
            snprintf(act_items[4], sizeof(act_items[4]), "AI Upscale Artwork -> %s", UP_MODES[(g_gal_up_mode >= 1 && g_gal_up_mode <= 3) ? g_gal_up_mode : 1]);
            snprintf(act_items[5], sizeof(act_items[5]), it->starred ? "Unstar Favorite" : "Star as Favorite (★)");
            for (int k = 0; k < 6; k++) {
                int ry = my + 72 + k * 54;
                int sel = (k == g_gal_asel);
                g_rrect(back, mx + 20, ry, mw - 40, 46, 12, sel ? C_SURFACE_HI : C_SURFACE, sel ? 255 : 165);
                if (sel) g_rrect_line(back, mx + 20, ry, mw - 40, 46, 12, 2, k == 0 ? C_GREEN : C_ACCENT_HI, 255);
                char afit[140];
                sm_fit(&g_fsmall, act_items[k], mw - 84, afit, sizeof(afit));
                g_draw_str(back, &g_fsmall, mx + 40, ry + 11, afit, k == 0 ? C_GREEN : (sel ? 0xFFFFFF : C_TEXT));
            }
            const GHint ah[] = { { GBTN_UPDOWN, "Choose Action" }, { GBTN_CROSS, "Confirm" }, { GBTN_CIRCLE, "Close" } };
            g_draw_footer(back, ah, 3, "");
            present();
            return;
        }

        // Visual "Add / Move to Folder" Modal Overlay
        if (g_gal_fmodal) {
            g_fill(back, 0, 0, SCREEN_W, SCREEN_H, 0x020611, 185);
            int fn_tot = 2 + g_num_folders;
            int vis = fn_tot < 9 ? fn_tot : 9;
            int mw = 760, mh = 90 + vis * 52 + 16;
            int mx = (SCREEN_W - mw) / 2, my = (SCREEN_H - mh) / 2 - 12;
            g_rrect(back, mx, my, mw, mh, 22, C_PANEL, 252);
            g_rrect_line(back, mx, my, mw, mh, 22, 2, C_ACCENT_HI, 255);
            char mhdr[128];
            if (g_gal_sel_n > 0) snprintf(mhdr, sizeof(mhdr), "Move %d Selected Artworks to Folder", g_gal_sel_n);
            else snprintf(mhdr, sizeof(mhdr), "Add / Move Artwork to Folder");
            g_draw_str(back, &g_fbold, mx + 30, my + 22, mhdr, 0xFFFFFF);

            for (int k = 0; k < vis; k++) {
                int idx_f = g_gal_ftop + k;
                if (idx_f >= fn_tot) break;
                int ry = my + 72 + k * 52;
                int sel = (idx_f == g_gal_fsel);
                g_rrect(back, mx + 20, ry, mw - 40, 46, 12, sel ? C_SURFACE_HI : C_SURFACE, sel ? 255 : 165);
                if (sel) g_rrect_line(back, mx + 20, ry, mw - 40, 46, 12, 2, C_ACCENT_HI, 255);
                char flbl[160], ffit[160];
                if (idx_f == 0) {
                    snprintf(flbl, sizeof(flbl), "+ Create New Folder...");
                } else if (idx_f == 1) {
                    snprintf(flbl, sizeof(flbl), "Unsorted (Remove from Folder)");
                } else {
                    snprintf(flbl, sizeof(flbl), "Folder: %s   (%d images)",
                             g_folders[idx_f - 2].name, g_folders[idx_f - 2].count);
                }
                sm_fit(&g_fsmall, flbl, mw - 88, ffit, sizeof(ffit));
                g_draw_str(back, &g_fsmall, mx + 44, ry + 11, ffit, idx_f == 0 ? C_GREEN : (sel ? 0xFFFFFF : C_TEXT));
            }
            const GHint fm_hints[] = {
                { GBTN_UPDOWN, "Choose Folder" },
                { GBTN_CROSS, "Confirm" },
                { GBTN_CIRCLE, "Cancel" },
            };
            g_draw_footer(back, fm_hints, 3, "");
            present();
            return;
        }

        const GHint ghints[] = {
            { GBTN_LEFTRIGHT, "Browse" },
            { GBTN_CROSS, "Actions" },
            { GBTN_R1, "Add to Folder" },
            { GBTN_L1, "Filter" },
            { GBTN_SQUARE, "Zoom" },
            { GBTN_R2, is_sel ? "Unselect" : "Select" },
            { GBTN_L3, "Upscale" },
            { GBTN_CIRCLE, "Back" },
        };
        g_draw_footer(back, ghints, (int)(sizeof(ghints) / sizeof(ghints[0])), "");
        present();
        return;
    }

    // ---- 4. MAIN STUDIO SCREEN (Dropdowns, Sliders & D-Pad/X/O Navigation) ----
    {
        char neg_buf[512], lora_buf[80], bfld_buf[64];
        float lscale_v = 0.8f;
        int smp_v = 0, ready_v = 0, sd_v = 0, nsfw_v = 0, aq_v = 1;
        uint32_t batch_v = 1, rmask_v = 0;
        uint64_t seed_v = 0;
        http_srv_lock();
        snprintf(neg_buf, sizeof(neg_buf), "%s", g_job.negative);
        snprintf(lora_buf, sizeof(lora_buf), "%s", g_job.lora);
        snprintf(bfld_buf, sizeof(bfld_buf), "%s", g_job.batch_folder);
        lscale_v = g_job.lora_scale;
        smp_v = g_job.sampler;
        seed_v = g_job.seed;
        batch_v = g_job.batch_count ? g_job.batch_count : 1u;
        rmask_v = g_job.batch_rand_mask;
        nsfw_v = g_job.nsfw_allow;
        aq_v = g_job.auto_quality;
        sd_v = g_job.shutdown_after;
        ready_v = g_st.ready;
        http_srv_unlock();

        StudioView sv;
        memset(&sv, 0, sizeof(sv));
        sv.model = cur_m;
        sv.lora_name = lora_buf[0] ? lora_buf : NULL;
        sv.lora_scale = lscale_v;
        sv.neo = neo_v;
        sv.busy = busy;
        sv.ready = ready_v;
        sv.has_upscaler = has_up_v;
        sv.ui_mode = (uim_v == 0) ? 0 : 1;
        sv.easy_framing = g_easy_framing;
        sv.easy_quality = g_easy_quality;
        sv.update_ver = upd.update_available ? upd.remote_ver : NULL;
        sv.vram_u = vram_u;
        sv.vram_t = vram_t;
        sv.prog = prog;
        sv.stage = stg;
        sv.err = err;
        sv.toast = toast;
        sv.addr = http_srv_address();
        sv.preset_name = (preset_idx >= 0 && preset_idx < g_num_presets) ? g_presets[preset_idx].name : NULL;
        sv.char_name = (char_idx >= 0 && char_idx < g_num_chars) ? g_chars[char_idx].name : NULL;
        sv.char_series = (char_idx >= 0 && char_idx < g_num_chars) ? g_chars[char_idx].series : NULL;
        sv.style_name = (style_idx >= 0 && style_idx < g_num_styles) ? g_styles[style_idx].name : NULL;
        sv.prompt = pr;
        sv.neg = neg_buf;
        sv.w = w; sv.h = h; sv.steps = steps; sv.cfg = cfg;
        sv.sampler = smp_v; sv.upscale = ups; sv.eco = eco;
        sv.batch = batch_v; sv.rand_mask = rmask_v;
        sv.batch_folder = bfld_buf; sv.nsfw_allow = nsfw_v; sv.auto_quality = aq_v;
        sv.shutdown_after = sd_v;
        sv.seed = seed_v; sv.last_seed = sm_last_seed;
        sv.last_sec = lsec; sv.last_tf = tf;
        sv.eta_sec = eta_s; sv.elapsed_sec = elap_s;
        sv.preview = ui_prev_rgb; sv.pw = ui_prev_w; sv.ph = ui_prev_h;
        sv.focus = sm_focus; sv.adjust = sm_adjust;
        sv.dd_open = sm_dd_open; sv.dd_row = sm_dd_row;
        sv.dd_sel = sm_dd_sel; sv.dd_top = sm_dd_top;
        sv.dd_n = sm_dd_n; sv.dd_cur = sm_dd_cur;
        sv.dd_label = sm_dd_item_label;
        sv.dd_ctx = NULL;
        sm_draw_studio(back, &sv);
        present();
    }
}

static float compute_live_eta(uint32_t bcur, uint32_t btot) {
    int eco = (g_eta_eco >= 0 && g_eta_eco <= 3) ? g_eta_eco : 2;
    double tf = (eng.stats.bench_mode_tf[eco] > 0.2) ? eng.stats.bench_mode_tf[eco] : (eng.neo ? 2.6 : 1.3);
    double lat_sc = (double)(g_eta_lat ? g_eta_lat : 12288ull) / 12288.0;
    double tf_sc = 2.8 / tf;

    if (g_eta_gal_batch) {
        double rem_cur = 4.5 * tf_sc;
        if (eng.rt_phase == 4 && eng.rt_tiles_total > 0) {
            double per_t = (eng.rt_tiles_done > 0 && eng.rt_tile_avg_sec > 0.01) ? eng.rt_tile_avg_sec : (0.55 * tf_sc);
            rem_cur = (double)(eng.rt_tiles_total - eng.rt_tiles_done) * per_t;
        }
        double per_img = (g_eta_done_items > 0) ? (g_eta_done_sec / (double)g_eta_done_items)
                       : ((eng.rt_phase == 4 && eng.rt_tiles_done > 0 && eng.rt_tile_avg_sec > 0.01)
                          ? ((double)eng.rt_tiles_total * eng.rt_tile_avg_sec + 0.5) : (5.5 * tf_sc));
        uint32_t rem_imgs = (btot > bcur) ? (btot - bcur) : 0u;
        return (float)(rem_cur + (double)rem_imgs * per_img);
    }

    double cfg_passes = (g_eta_cfg > 1.001f) ? 2.0 : 1.0;
    double est_step = (eng.rt_step_done > 0 && eng.rt_step_avg_sec > 0.01)
        ? eng.rt_step_avg_sec
        : ((eng.is_sd15 ? 0.55 : 1.75) * cfg_passes * lat_sc * tf_sc);
    double est_vae = (eng.is_sd15 ? 2.6 : 7.8) * lat_sc * tf_sc;
    double est_up  = (g_eta_ups > 0 && eng.has_upscaler) ? ((g_eta_ups == 3 ? 9.5 : 4.8) * lat_sc * tf_sc) : 0.0;
    uint32_t tot_st = eng.rt_steps_total ? eng.rt_steps_total : 6u;

    double rem_cur = 0.0;
    if (eng.rt_phase == 1) {
        rem_cur = 0.8 + (double)tot_st * est_step + est_vae + est_up;
    } else if (eng.rt_phase == 2) {
        uint32_t rem_s = (tot_st > eng.rt_step_done) ? (tot_st - eng.rt_step_done) : 0u;
        rem_cur = (double)rem_s * est_step + est_vae + est_up;
    } else if (eng.rt_phase == 3) {
        if (eng.rt_tiles_done > 0 && eng.rt_tile_avg_sec > 0.01 && eng.rt_tiles_total >= eng.rt_tiles_done) {
            rem_cur = (double)(eng.rt_tiles_total - eng.rt_tiles_done) * eng.rt_tile_avg_sec + est_up;
        } else if (eng.rt_tiles_total > 0) {
            rem_cur = est_vae * (1.0 - (double)eng.rt_tiles_done / (double)eng.rt_tiles_total) + est_up;
        } else {
            rem_cur = est_vae + est_up;
        }
    } else if (eng.rt_phase == 4) {
        if (eng.rt_tiles_done > 0 && eng.rt_tile_avg_sec > 0.01 && eng.rt_tiles_total >= eng.rt_tiles_done) {
            rem_cur = (double)(eng.rt_tiles_total - eng.rt_tiles_done) * eng.rt_tile_avg_sec;
        } else if (eng.rt_tiles_total > 0) {
            rem_cur = est_up * (1.0 - (double)eng.rt_tiles_done / (double)eng.rt_tiles_total);
        } else {
            rem_cur = est_up;
        }
    } else {
        return 0.0f;
    }

    uint32_t rem_imgs = (btot > bcur && bcur > 0) ? (btot - bcur) : 0u;
    double per_img = (g_eta_done_items > 0) ? (g_eta_done_sec / (double)g_eta_done_items)
                   : ((double)tot_st * est_step + est_vae + est_up + 1.0);
    return (float)(rem_cur + (double)rem_imgs * per_img);
}

static void on_progress(void* user, float frac, const char* msg) {
    (void)user;
    uint64_t now_us = sceKernelGetProcessTime();
    http_srv_lock();
    uint32_t bcur = g_st.batch_cur, btot = g_st.batch_total;
    if (btot > 1 && bcur >= 1) {
        g_st.progress = ((float)(bcur - 1u) + frac) / (float)btot;
        snprintf(g_st.stage, sizeof(g_st.stage), "[%u/%u] %s", bcur, btot, msg);
    } else {
        g_st.progress = frac;
        snprintf(g_st.stage, sizeof(g_st.stage), "%s", msg);
    }
    if (g_job_start_us > 0 && now_us >= g_job_start_us) {
        g_st.elapsed_sec = (float)((double)(now_us - g_job_start_us) / 1e6);
    }
    if (eng.mem_size > 0) g_st.gpu_used_mb = (uint32_t)(eng.next >> 20);
    g_st.eta_sec = compute_live_eta(bcur ? bcur : 1u, btot ? btot : 1u);
    http_srv_unlock();
}

static int on_cancel(void* user) {
    (void)user;
    http_srv_lock();
    int c = g_job.cancel_req;
    http_srv_unlock();
    return c;
}

static int edit_text_osk(const wchar_t* title, char* target, size_t target_sz, int numeric_only) {
    static wchar_t buf[513];
    memset(buf, 0, sizeof(buf));
    http_srv_lock();
    for (int i = 0; target[i] && i < 512; i++) buf[i] = (uint8_t)target[i];
    http_srv_unlock();

    // Wait for Cross/Circle release so the button that opened the OSK doesn't leak into it
    if (pad_fd >= 0) {
        for (int w = 0; w < 25; w++) {
            OrbisPadData pd;
            if (scePadReadState(pad_fd, &pd) != 0 || !(pd.buttons & (ORBIS_PAD_BUTTON_CROSS | ORBIS_PAD_BUTTON_CIRCLE)))
                break;
            sceKernelUsleep(16000);
        }
    }

    OrbisImeDialogSetting s;
    memset(&s, 0, sizeof(s));
    s.userId = user_id;
    s.type = numeric_only ? ORBIS_TYPE_NUMBER : ORBIS_TYPE_DEFAULT;
    s.enterLabel = ORBIS_BUTTON_LABEL_DEFAULT;
    s.maxTextLength = numeric_only ? 20 : 512;
    s.inputTextBuffer = buf;
    s.posx = SCREEN_W / 2;
    s.posy = SCREEN_H / 2;
    s.horizontalAlignment = ORBIS_H_CENTER;
    s.verticalAlignment = ORBIS_V_CENTER;
    s.placeholder = numeric_only ? L"Enter number, then press R2 (Done)" : L"Type text, then press R2 (Done)";
    s.title = title;
    int ir = sceImeDialogInit(&s, NULL);
    if (ir < 0) {
        glog("osk: sceImeDialogInit failed 0x%08x", (unsigned)ir);
        return 0;
    }
    int frames = 0, running = 0;
    for (;;) {
        OrbisDialogStatus st = sceImeDialogGetStatus();
        if (st == ORBIS_DIALOG_STATUS_STOPPED) break;
        if (st == ORBIS_DIALOG_STATUS_RUNNING) running = 1;
        if (st == ORBIS_DIALOG_STATUS_NONE && (running || ++frames > 300)) break;
        draw_ui();
    }
    OrbisDialogResult res;
    memset(&res, 0, sizeof(res));
    sceImeDialogGetResult(&res);
    sceImeDialogTerm();

    // Drain pad briefly after closing OSK so R2/X/O doesn't trigger Studio actions
    if (pad_fd >= 0) {
        for (int w = 0; w < 10; w++) {
            OrbisPadData pd;
            if (scePadReadState(pad_fd, &pd) != 0 || !pd.buttons) break;
            sceKernelUsleep(16000);
        }
    }

    if (res.endstatus == ORBIS_DIALOG_OK) {
        http_srv_lock();
        int n = 0;
        for (int i = 0; buf[i] && n < (int)target_sz - 1; i++)
            target[n++] = (buf[i] < 128) ? (char)buf[i] : ' ';
        target[n] = 0;
        http_srv_unlock();
        glog("osk: saved %d chars", n);
        return 1;
    }
    glog("osk: closed endstatus=%d", (int)res.endstatus);
    return 0;
}

static int file_exists(const char* path) {
    int f = open(path, O_RDONLY);
    if (f < 0) return 0;
    close(f);
    return 1;
}

static int str_contains_ci(const char* hay, const char* needle) {
    size_t nl = strlen(needle);
    for (; *hay; hay++) {
        size_t i = 0;
        while (i < nl && hay[i]) {
            char a = hay[i], b = needle[i];
            if (a >= 'A' && a <= 'Z') a += 32;
            if (b >= 'A' && b <= 'Z') b += 32;
            if (a != b) break;
            i++;
        }
        if (i == nl) return 1;
    }
    return 0;
}

static int model_is_installed_name(const char* name) {
    if (!name || !name[0]) return 0;
    for (int i = 0; i < g_st.num_models; i++) {
        if (strcmp(g_st.models[i], name) == 0) return 1;
    }
    return 0;
}

static void apply_gallery_item_to_job_locked(const GalleryItem* it, int new_seed, int is_sd15) {
    if (!it) return;
    // Restore prompt + LoRA tags (avoid duplicating if <lora:...> is already in prompt)
    if (it->lora[0] && !strstr(it->prompt, "<lora:")) {
        char ptmp[1024] = {0};
        char lcpy[128];
        snprintf(lcpy, sizeof(lcpy), "%s", it->lora);
        char* tok = lcpy;
        while (tok && *tok) {
            char* comma = strchr(tok, ',');
            if (comma) *comma = 0;
            while (*tok == ' ' || *tok == '\t') tok++;
            if (*tok) {
                char tag[128];
                if (strchr(tok, ':')) snprintf(tag, sizeof(tag), "<lora:%s> ", tok);
                else snprintf(tag, sizeof(tag), "<lora:%s:0.75> ", tok);
                strncat(ptmp, tag, sizeof(ptmp) - strlen(ptmp) - 1);
            }
            tok = comma ? (comma + 1) : NULL;
        }
        strncat(ptmp, it->prompt, sizeof(ptmp) - strlen(ptmp) - 1);
        snprintf(g_job.prompt, sizeof(g_job.prompt), "%s", ptmp);
        g_job.lora[0] = 0;
    } else if (it->prompt[0]) {
        snprintf(g_job.prompt, sizeof(g_job.prompt), "%s", it->prompt);
        g_job.lora[0] = 0;
    }
    if (it->negative[0]) snprintf(g_job.negative, sizeof(g_job.negative), "%s", it->negative);

    // Restore base generation resolution (handle upscaled gallery items cleanly!)
    uint32_t w = it->width, h = it->height;
    int needs_base_map = (it->upscaled > 0) ||
                         (is_sd15 && (w > 896u || h > 896u)) ||
                         (!is_sd15 && (w > 1280u || h > 1344u));
    if (w < 256u || h < 256u || needs_base_map) {
        if (is_sd15) {
            if (w > h + 64u) { w = 768u; h = 512u; }
            else if (h > w + 64u) { w = 512u; h = 768u; }
            else { w = 512u; h = 512u; }
        } else {
            if (w > h + 64u) { w = 1280u; h = 704u; }
            else if (h > w + 64u) { w = 832u; h = 1216u; }
            else { w = 1024u; h = 1024u; }
        }
    } else {
        w = (w / 64u) * 64u;
        h = (h / 64u) * 64u;
    }
    if (!eng.neo && ((uint64_t)(w / 8u) * (h / 8u) > 12288ull)) {
        int land = (w > h);
        w = land ? 1024u : 768u;
        h = land ? 768u : 1024u;
    }
    g_job.width = w;
    g_job.height = h;
    if (it->steps > 0) g_job.steps = it->steps;
    if (it->cfg > 0.0f) g_job.cfg = it->cfg;
    if (it->sampler[0]) {
        g_job.sampler = strstr(it->sampler, "DPM") ? 3 : ((strcmp(it->sampler, "LCM") == 0) ? 2 : (strcmp(it->sampler, "Euler") == 0 ? 1 : 0));
    }
    g_job.upscale = (it->upscaled >= 0 && it->upscaled <= 3) ? (eng.has_upscaler ? it->upscaled : 0) : 0;
    if (it->eco >= 0 && it->eco <= 3) g_job.eco_mode = it->eco;
    g_job.seed = new_seed ? 0ull : it->seed;
    g_st.settings_seq++;

    for (int i = 0; i < (int)(sizeof(RES_LIST) / sizeof(RES_LIST[0])); i++) {
        if (RES_LIST[i][0] == g_job.width && RES_LIST[i][1] == g_job.height) { res_idx = i; break; }
    }
}

static void apply_model_preset(const char* model_dir, const char* name, int is_sd15) {
    ModelPreset pr, sp;
    sdxl_load_preset(model_dir, name, is_sd15, &pr);
    sdxl_load_speed_preset(name, is_sd15, &sp);
    // Clamp 1024x1344 to 768x1024 on Base PS4 (<4500 MiB compact scratch)
    if (!eng.neo && ((uint64_t)(pr.width / 8u) * (pr.height / 8u) > 12288ull)) {
        int land = (pr.width > pr.height);
        pr.width = land ? 1024u : 768u;
        pr.height = land ? 768u : 1024u;
    }
    if (!eng.neo && ((uint64_t)(sp.width / 8u) * (sp.height / 8u) > 12288ull)) {
        int land = (sp.width > sp.height);
        sp.width = land ? 1024u : 768u;
        sp.height = land ? 768u : 1024u;
    }
    http_srv_lock();
    // Auto-clear selected Studio LoRA if its architecture does not match the newly loaded model
    if (g_job.lora[0]) {
        InstalledLora* tmp_l = (InstalledLora*)malloc(sizeof(InstalledLora) * LORA_MAX_INSTALLED);
        if (tmp_l) {
            int ntl = lora_list_installed(tmp_l, LORA_MAX_INSTALLED);
            for (int i = 0; i < ntl; i++) {
                if (strcmp(tmp_l[i].name, g_job.lora) == 0 && !lora_arch_compat(tmp_l[i].arch, is_sd15)) {
                    glog("Cleared incompatible Studio LoRA '%s' (%s) after switching to %s",
                         g_job.lora, tmp_l[i].arch, is_sd15 ? "SD1.5" : "SDXL");
                    g_job.lora[0] = 0;
                    break;
                }
            }
            free(tmp_l);
        }
    }
    g_job.width = pr.width;
    g_job.height = pr.height;
    g_job.steps = pr.steps;
    g_job.cfg = pr.cfg;
    g_job.sampler = pr.sampler;
    g_job.upscale = eng.has_upscaler ? pr.upscale : 0;
    snprintf(g_job.negative, sizeof(g_job.negative), "%s", pr.negative);
    g_st.preset_w = pr.width;
    g_st.preset_h = pr.height;
    g_st.preset_steps = pr.steps;
    g_st.preset_cfg = pr.cfg;
    g_st.preset_sampler = pr.sampler;
    g_st.preset_upscale = eng.has_upscaler ? pr.upscale : 0;
    snprintf(g_st.preset_desc, sizeof(g_st.preset_desc), "%s", pr.description);
    snprintf(g_st.preset_prompt, sizeof(g_st.preset_prompt), "%s", pr.prompt);
    snprintf(g_st.preset_negative, sizeof(g_st.preset_negative), "%s", pr.negative);
    g_st.speed_w = sp.width;
    g_st.speed_h = sp.height;
    g_st.speed_steps = sp.steps;
    g_st.speed_cfg = sp.cfg;
    g_st.speed_sampler = sp.sampler;
    g_st.speed_upscale = eng.has_upscaler ? sp.upscale : 0;
    snprintf(g_st.speed_desc, sizeof(g_st.speed_desc), "%s", sp.description);
    g_st.settings_seq++;
    http_srv_unlock();
    for (int i = 0; i < (int)(sizeof(RES_LIST) / sizeof(RES_LIST[0])); i++) {
        if (RES_LIST[i][0] == pr.width && RES_LIST[i][1] == pr.height) { res_idx = i; break; }
    }
    glog("preset (%s): %ux%u %usteps cfg=%.1f sampler=%d upscale=%d | %s",
         name, pr.width, pr.height, pr.steps, (double)pr.cfg, pr.sampler, pr.upscale, pr.description);
}

static int cmp_model_name(const void* a, const void* b) {
    return strcmp((const char*)a, (const char*)b);
}

static int scan_models(void) {
    char found[DIFF_MAX_MODELS][64];
    int n = 0;
    DIR* d = opendir(MODELS_ROOT);
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL && n < DIFF_MAX_MODELS) {
            if (ent->d_name[0] == '.') continue;
            char test[300];
            snprintf(test, sizeof(test), "%s/%s/unet.psd1", MODELS_ROOT, ent->d_name);
            if (file_exists(test)) {
                snprintf(found[n], sizeof(found[n]), "%s", ent->d_name);
                n++;
            }
        }
        closedir(d);
    }
    if (n > 1) qsort(found, (size_t)n, sizeof(found[0]), cmp_model_name);
    http_srv_lock();
    g_st.num_models = n;
    for (int i = 0; i < n; i++) snprintf(g_st.models[i], sizeof(g_st.models[i]), "%s", found[i]);
    http_srv_unlock();
    return n;
}

extern struct { int can_do_threads, threaded, secure; volatile int threads_minus_1; } __libc;

static void log_console_info(void) {
    OrbisKernelSwVersion v;
    memset(&v, 0, sizeof(v));
    v.Size = sizeof(v);
    int r = sceKernelGetSystemSwVersion(&v);
    v.VersionString[sizeof(v.VersionString) - 1] = 0;
    const char* fw = v.VersionString;
    while (*fw == ' ' || *fw == '0') fw++;
    size_t flex = 0;
    ((int32_t(*)(size_t*))sceKernelAvailableFlexibleMemorySize)(&flex);
    sdxl_read_temp_c(NULL, NULL);
    glog("console: fw %s (0x%08x); PS4 Pro hw=%d Neo=%d; direct %u MiB, flex free %u MiB; TSC %.1f MHz",
         r ? "?" : fw, (unsigned)v.Version, sceKernelHasNeoMode(), sceKernelIsNeoMode(),
         (unsigned)(sceKernelGetDirectMemorySize() >> 20), (unsigned)(flex >> 20),
         sceKernelGetTscFrequency() / 1e6);
}

static void* gpu_worker_main(void* arg) {
    (void)arg;
    // Phase 1: Initialize GPU, kernels, self-test, upscaler, and CU power benchmark on background thread
    if (sdxl_init(&eng, ASSETS_DIR, NULL, on_progress, NULL)) {
        http_srv_lock();
        g_st.busy = 0;
        snprintf(g_st.error, sizeof(g_st.error), "GPU init failed: %s", eng.error);
        http_srv_unlock();
        glog("sdxl_init (GPU) failed: %s", eng.error);
        return NULL;
    }
    http_srv_lock();
    g_st.gpu_ready = 1;
    g_st.neo = eng.neo;
    g_st.has_upscaler = eng.has_upscaler;
    g_st.gpu_used_mb = (uint32_t)(eng.next >> 20);
    g_st.gpu_total_mb = (uint32_t)(eng.mem_size >> 20);
    if (!eng.neo) g_job.eco_mode = 1;
    g_st.eco_mode = g_job.eco_mode;
    g_st.bench_gemm_tflops = eng.stats.bench_gemm_tflops;
    g_st.bench_attn_ms = eng.stats.bench_attn_ms;
    memcpy(g_st.bench_mode_tf, eng.stats.bench_mode_tf, sizeof(g_st.bench_mode_tf));
    http_srv_unlock();

    // Phase 2: Scan models & auto-fallback across candidates without blocking the 60 FPS UI!
    int n_found = scan_models();
    int loaded_ok = 0, skipped_cnt = 0;
    char skipped_msg[256] = {0};
    if (n_found > 0) {
        int order[DIFF_MAX_MODELS];
        int used[DIFF_MAX_MODELS] = {0};
        int oc = 0;
        for (int i = 0; i < n_found; i++)
            if (!used[i] && str_contains_ci(g_st.models[i], "dmd2")) { order[oc++] = i; used[i] = 1; }
        for (int i = 0; i < n_found; i++)
            if (!used[i] && str_contains_ci(g_st.models[i], "8step")) { order[oc++] = i; used[i] = 1; }
        for (int i = 0; i < n_found; i++)
            if (!used[i] && strcmp(g_st.models[i], "waiNSFWIllustrious_v140") == 0) { order[oc++] = i; used[i] = 1; }
        for (int i = 0; i < n_found; i++)
            if (!used[i]) { order[oc++] = i; used[i] = 1; }

        glog("found %d model(s); selecting first complete model...", n_found);
        for (int k = 0; k < oc; k++) {
            int idx = order[k];
            char model_dir[256];
            snprintf(model_dir, sizeof(model_dir), "%s/%s", MODELS_ROOT, g_st.models[idx]);
            glog("trying model [%d/%d]: %s", k + 1, oc, model_dir);
            if (sdxl_reload_model(&eng, model_dir, on_progress, NULL) == 0) {
                model_idx = idx;
                apply_model_preset(model_dir, eng.model_name, eng.is_sd15);
                loaded_ok = 1;
                break;
            }
            skipped_cnt++;
            snprintf(skipped_msg, sizeof(skipped_msg), "%s", eng.error);
            glog("skipping model %s: %s", g_st.models[idx], eng.error);
        }
    }

    http_srv_lock();
    g_st.busy = 0;
    g_job.cancel_req = 0;
    if (loaded_ok) {
        g_st.ready = 1;
        g_st.is_sd15 = eng.is_sd15;
        g_st.gpu_used_mb = (uint32_t)(eng.next >> 20);
        g_st.progress = 1.0f;
        g_st.error[0] = 0;
        snprintf(g_st.model, sizeof(g_st.model), "%s", eng.model_name);
        if (skipped_cnt > 0) {
            snprintf(g_st.stage, sizeof(g_st.stage), "Ready! %s (skipped %d incomplete) | http://%s",
                     sdxl_eco_name_ex(g_job.eco_mode, eng.neo), skipped_cnt, http_srv_address());
        } else {
            snprintf(g_st.stage, sizeof(g_st.stage), "Ready! %s | http://%s",
                     sdxl_eco_name_ex(g_job.eco_mode, eng.neo), http_srv_address());
        }
    } else {
        g_st.ready = 0;
        g_st.model[0] = 0;
        g_st.progress = 0.0f;
        if (n_found == 0) {
            snprintf(g_st.error, sizeof(g_st.error), "No models in %s - upload & Rescan", MODELS_ROOT);
        } else {
            snprintf(g_st.error, sizeof(g_st.error), "No complete model loaded (%s) - pick another or Rescan", skipped_msg);
        }
        snprintf(g_st.stage, sizeof(g_st.stage), "Pick a complete model in Web UI or press [OPTIONS]");
    }
    http_srv_unlock();
    if (loaded_ok) {
        glog("ready: %s (%u / %u MiB GPU, Neo=%d, skipped=%d)", eng.model_name, (unsigned)(eng.next >> 20),
             (unsigned)(eng.mem_size >> 20), eng.neo, skipped_cnt);
    } else {
        glog("booted with no active model (%s) - Web UI & controller remain interactive", g_st.error);
    }

    for (;;) {
        // Check for queued Gallery Upscale / Batch Upscale / Resize job (from Web UI or TV Gallery)
        int gop = 0, gal_n = 0;
        char gids[DIFF_MAX_BATCH_GAL][64];
        uint32_t gw = 0, gh = 0;
        int geco = 2;
        float gden = 0.42f;
        http_srv_lock();
        if (g_job.gal_op && !g_st.busy) {
            gop = g_job.gal_op;
            if (gop >= 1 && gop <= 3 && g_job.gal_ids_n > 0) {
                gal_n = g_job.gal_ids_n > DIFF_MAX_BATCH_GAL ? DIFF_MAX_BATCH_GAL : g_job.gal_ids_n;
                for (int k = 0; k < gal_n; k++) snprintf(gids[k], sizeof(gids[0]), "%s", g_job.gal_ids[k]);
            } else {
                gal_n = 1;
                snprintf(gids[0], sizeof(gids[0]), "%s", g_job.gal_id);
            }
            gw = g_job.gal_w; gh = g_job.gal_h;
            gden = g_job.gal_denoise > 0.05f ? g_job.gal_denoise : 0.42f;
            geco = g_job.eco_mode;
            g_job.gal_op = 0;
            g_job.gal_ids_n = 0;
            g_st.busy = 1;
            g_st.error[0] = 0;
            g_st.progress = 0.0f;
            g_st.eta_sec = 0.0f;
            g_st.elapsed_sec = 0.0f;
            g_st.batch_cur = 1u;
            g_st.batch_total = (uint32_t)gal_n;
        }
        http_srv_unlock();
        if (gop) {
            g_job_start_us = sceKernelGetProcessTime();
            g_eta_done_items = 0;
            g_eta_done_sec = 0.0;
            g_eta_eco = geco;
            g_eta_gal_batch = 1;
            int done_cnt = 0, was_cancelled = 0;
            char last_nid[64] = {0};
            uint32_t last_dw = 0, last_dh = 0;

            for (int gi = 0; gi < gal_n; gi++) {
                if (on_cancel(NULL)) { was_cancelled = 1; break; }
                http_srv_lock();
                g_st.batch_cur = (uint32_t)(gi + 1);
                g_st.batch_total = (uint32_t)gal_n;
                http_srv_unlock();

                uint64_t item_t0 = sceKernelGetProcessTime();
                GalleryItem src_meta;
                size_t psz = 0;
                uint8_t* spng = (gallery_get(gids[gi], &src_meta) == 0) ? gallery_load_png(gids[gi], &psz) : NULL;
                uint32_t sw = 0, sh = 0;
                uint8_t* srgb = spng ? png_decode_rgb(spng, psz, &sw, &sh) : NULL;
                free(spng);
                if (!srgb) {
                    glog("gallery op skip unreadable %s", gids[gi]);
                    continue;
                }
                uint8_t* drgb = NULL;
                uint32_t dw = 0, dh = 0;
                int ok = -1;
                if (gop >= 1 && gop <= 3) {
                    ok = sdxl_upscale_rgb(&eng, srgb, sw, sh, &drgb, &dw, &dh, gop, geco, on_progress, on_cancel, NULL);
                } else if (gop == 4) {
                    on_progress(NULL, 0.5f, "Resizing image...");
                    drgb = rgb_resize(srgb, sw, sh, gw, gh);
                    dw = gw; dh = gh;
                    ok = drgb ? 0 : -1;
                } else if (gop == 5) {
                    on_progress(NULL, 0.02f, "Detecting anime face (CPU)...");
                    FaceBox faces[2];
                    int nf = face_detect_anime(srgb, sw, sh, faces, 2);
                    if (nf <= 0) {
                        http_srv_lock();
                        snprintf(g_st.error, sizeof(g_st.error), "No anime face detected in %s", gids[gi]);
                        http_srv_unlock();
                        ok = -1;
                    } else {
                        FaceBox crop_box;
                        face_box_expand_square(&faces[0], sw, sh, 1.85f, &crop_box);
                        uint32_t fsz = (eng.is_sd15 || !eng.neo) ? 512u : 768u;
                        uint8_t* fcrop = face_crop_resize(srgb, sw, sh, &crop_box, fsz, fsz);
                        uint8_t* fref  = fcrop ? (uint8_t*)malloc((size_t)fsz * fsz * 3u) : NULL;
                        if (fcrop && fref) {
                            char fprompt[1024], fneg[512];
                            snprintf(fprompt, sizeof(fprompt),
                                     "close-up portrait, detailed face, detailed eyes, symmetric pupils, catchlights, %s",
                                     src_meta.prompt[0] ? src_meta.prompt : "1girl, anime");
                            snprintf(fneg, sizeof(fneg), "%s",
                                     src_meta.negative[0] ? src_meta.negative
                                                          : "lowres, bad eyes, asymmetrical eyes, cross-eyed, deformed face, blurry, worst quality");
                            SdxlParams fp;
                            fp.prompt = fprompt;
                            fp.negative = fneg;
                            fp.width = fsz;
                            fp.height = fsz;
                            fp.steps = g_st.preset_steps ? g_st.preset_steps : (eng.is_sd15 ? 20u : 8u);
                            fp.cfg_scale = g_st.preset_cfg > 0.5f ? g_st.preset_cfg : (eng.is_sd15 ? 6.5f : 1.4f);
                            fp.seed = src_meta.seed ? (src_meta.seed ^ 0xFA5E1234ull) : 424242ull;
                            fp.eco_mode = geco;
                            fp.sampler = g_st.preset_sampler;
                            glog("fix_face %s: box=(%d,%d,%dx%d) score=%.2f -> %ux%u denoise=%.2f",
                                 gids[gi], crop_box.x, crop_box.y, crop_box.w, crop_box.h, (double)faces[0].score, fsz, fsz, (double)gden);
                            ok = sdxl_img2img(&eng, &fp, fcrop, gden, fref, on_progress, on_cancel, NULL);
                            if (ok == 0) {
                                drgb = (uint8_t*)malloc((size_t)sw * sh * 3u);
                                if (drgb) {
                                    memcpy(drgb, srgb, (size_t)sw * sh * 3u);
                                    face_blend_back(drgb, sw, sh, fref, fsz, fsz, &crop_box, 0.15f);
                                    dw = sw; dh = sh;
                                } else {
                                    ok = -1;
                                }
                            } else if (ok < 0 && eng.error[0]) {
                                http_srv_lock();
                                snprintf(g_st.error, sizeof(g_st.error), "%s", eng.error);
                                http_srv_unlock();
                            }
                        }
                        free(fcrop);
                        free(fref);
                    }
                }
                free(srgb);
                if (ok == 1) { free(drgb); was_cancelled = 1; break; }
                if (ok == 0 && drgb) {
                    size_t nsz = 0;
                    src_meta.width = dw; src_meta.height = dh;
                    if (gop >= 1 && gop <= 3) src_meta.upscaled = gop;
                    char pmeta[2048];
                    format_a1111_params(pmeta, sizeof(pmeta), src_meta.prompt, src_meta.negative,
                                        src_meta.steps, src_meta.sampler, src_meta.cfg, src_meta.seed,
                                        dw, dh, src_meta.model, src_meta.lora, src_meta.upscaled);
                    uint8_t* npng = png_encode_rgb_meta(drgb, dw, dh, pmeta, &nsz);
                    if (npng) {
                        char nid[64] = {0};
                        http_srv_lock();
                        snprintf(g_st.stage, sizeof(g_st.stage), "Staging upscaled image in RAM...");
                        http_srv_unlock();
                        gallery_stage_enqueue(npng, nsz, drgb, dw, dh, &src_meta, nid, sizeof(nid));
                        publish_preview_rgb(drgb, dw, dh);
                        snprintf(last_nid, sizeof(last_nid), "%s", nid);
                        last_dw = dw; last_dh = dh;
                        done_cnt++;
                        http_srv_lock();
                        free(g_st.png_data);
                        g_st.png_data = npng; g_st.png_size = nsz;
                        g_st.width = dw; g_st.height = dh;
                        g_st.upscale_sec = eng.stats.upscale_ms / 1000.0;
                        g_st.image_gen_id++;
                        snprintf(g_st.last_gallery_id, sizeof(g_st.last_gallery_id), "%s", nid);
                        http_srv_unlock();
                        glog("gallery op=%d [%d/%d] %s -> %s (%ux%u)", gop, gi + 1, gal_n, gids[gi], nid, dw, dh);
                    } else {
                        free(drgb);
                    }
                } else {
                    free(drgb);
                }
                g_eta_done_items++;
                g_eta_done_sec += (double)(sceKernelGetProcessTime() - item_t0) / 1e6;
            }
            g_eta_gal_batch = 0;
            g_job_start_us = 0;
            http_srv_lock();
            g_st.batch_cur = 0;
            g_st.batch_total = 0;
            g_st.eta_sec = 0.0f;
            g_st.progress = done_cnt > 0 ? 1.0f : 0.0f;
            if (was_cancelled) {
                snprintf(g_st.stage, sizeof(g_st.stage), "Cancelled (%d/%d upscaled)", done_cnt, gal_n);
            } else if (done_cnt > 1) {
                snprintf(g_st.stage, sizeof(g_st.stage), "Batch upscaled %d images (latest: %s %ux%u)", done_cnt, last_nid, last_dw, last_dh);
            } else if (done_cnt == 1) {
                snprintf(g_st.stage, sizeof(g_st.stage), "Saved %s (%ux%u)", last_nid, last_dw, last_dh);
            } else {
                snprintf(g_st.error, sizeof(g_st.error), "Gallery operation failed");
            }
            g_st.busy = 0;
            http_srv_unlock();
        }

        // Check for runtime model switch request (controller OPTIONS or Web UI / A1111)
        char sw_name[64] = {0};
        http_srv_lock();
        if (g_job.switch_req && !g_st.busy) {
            snprintf(sw_name, sizeof(sw_name), "%s", g_job.switch_model);
            g_job.switch_req = 0;
            g_st.busy = 1;
            g_st.error[0] = 0;
            g_st.progress = 0.0f;
        }
        http_srv_unlock();
        if (sw_name[0]) {
            char sw_dir[256];
            snprintf(sw_dir, sizeof(sw_dir), "%s/%s", MODELS_ROOT, sw_name);
            glog("switching model -> %s", sw_dir);
            int sr = sdxl_reload_model(&eng, sw_dir, on_progress, NULL);
            if (sr == 0) {
                apply_model_preset(sw_dir, eng.model_name, eng.is_sd15);
                http_srv_lock();
                for (int i = 0; i < g_st.num_models; i++) {
                    if (strcmp(g_st.models[i], eng.model_name) == 0) { model_idx = i; break; }
                }
                g_st.ready = 1;
                g_st.is_sd15 = eng.is_sd15;
                g_st.gpu_used_mb = (uint32_t)(eng.next >> 20);
                g_st.active_lora[0] = 0;
                g_st.error[0] = 0;
                snprintf(g_st.model, sizeof(g_st.model), "%s", eng.model_name);
                snprintf(g_st.stage, sizeof(g_st.stage), "Switched to %s (%ux%u, %u stp, CFG %.1f)",
                         eng.model_name, g_job.width, g_job.height, g_job.steps, (double)g_job.cfg);
                g_st.progress = 1.0f;
                g_st.busy = 0;
                int cur_m_ui = g_job.ui_mode;
                int do_gal_restore = g_pending_gal_restore;
                int do_gal_run = g_pending_gal_run;
                g_pending_gal_restore = 0;
                g_pending_gal_run = 0;
                if (do_gal_restore) {
                    apply_gallery_item_to_job_locked(&g_pending_gal_item, do_gal_run, eng.is_sd15);
                    if (do_gal_run) {
                        g_job.cancel_req = 0;
                        g_job.pending = 1;
                    }
                }
                http_srv_unlock();
                if (!do_gal_restore && cur_m_ui == 0) apply_tv_easy_framing_and_quality();
                if (do_gal_restore && !do_gal_run) {
                    snprintf(toast, sizeof(toast), "Loaded model & all settings from %s", g_pending_gal_item.id);
                } else {
                    toast[0] = 0;
                }
            } else if (sr == -2) {
                // Pre-flight rejected incomplete/corrupt model BEFORE unloading current weights!
                glog("model switch rejected (keeping %s): %s", eng.model_name[0] ? eng.model_name : "none", eng.error);
                http_srv_lock();
                snprintf(g_st.error, sizeof(g_st.error), "%s", eng.error);
                g_st.progress = g_st.ready ? 1.0f : 0.0f;
                g_st.busy = 0;
                http_srv_unlock();
            } else {
                // Failed mid-load after unloading previous model; keep GPU alive so user can pick another model!
                glog("model switch failed: %s", eng.error);
                http_srv_lock();
                g_st.ready = 0;
                g_st.model[0] = 0;
                g_st.gpu_used_mb = (uint32_t)(eng.next >> 20);
                snprintf(g_st.error, sizeof(g_st.error), "%s", eng.error);
                g_st.progress = 0.0f;
                g_st.busy = 0;
                http_srv_unlock();
            }
        }

        // Check for pending job (controller or HTTP)
        DiffJob run_job;
        int do_run = 0;
        http_srv_lock();
        if (g_job.pending) {
            run_job = g_job;
            g_job.pending = 0;
            g_st.busy = 1;
            g_st.error[0] = 0;
            g_st.progress = 0.0f;
            do_run = 1;
        }
        http_srv_unlock();

        if (do_run) {
            toast[0] = 0;
            if (g_tv_gen) {
                g_tv_gen = 0;
                char tmp_p[1024], tmp_n[512];
                snprintf(tmp_p, sizeof(tmp_p), "%s", run_job.prompt);
                snprintf(tmp_n, sizeof(tmp_n), "%s", run_job.negative);
                if (style_idx >= 0 && style_idx < g_num_styles && g_styles[style_idx].prompt[0]) {
                    char merged[1024];
                    snprintf(merged, sizeof(merged), "%s, %s", tmp_p, g_styles[style_idx].prompt);
                    snprintf(tmp_p, sizeof(tmp_p), "%s", merged);
                    if (g_styles[style_idx].negative[0]) {
                        char mn[512];
                        snprintf(mn, sizeof(mn), "%s, %s", tmp_n, g_styles[style_idx].negative);
                        snprintf(tmp_n, sizeof(tmp_n), "%s", mn);
                    }
                }
                sdxl_clean_prompt_dupes(tmp_p, run_job.prompt, sizeof(run_job.prompt));
                sdxl_clean_prompt_dupes(tmp_n, run_job.negative, sizeof(run_job.negative));
            }
            uint32_t max_dim = (eng.is_sd15 || !eng.neo) ? 1280u : 1344u;
            uint64_t max_lat = (eng.is_sd15 || !eng.neo) ? 12288ull : 21504ull;
            uint32_t rw = (run_job.width + 32u) & ~63u;
            uint32_t rh = (run_job.height + 32u) & ~63u;
            if (rw < 256u) rw = 256u;
            if (rh < 256u) rh = 256u;
            if (rw > max_dim) rw = max_dim;
            if (rh > max_dim) rh = max_dim;
            while ((uint64_t)(rw / 8u) * (rh / 8u) > max_lat) {
                if (rw >= rh) rw -= 64u;
                else rh -= 64u;
            }
            uint32_t rsteps = run_job.steps < 1u ? 20u : (run_job.steps > 64u ? 64u : run_job.steps);
            float rcfg = run_job.cfg <= 0.0f ? 7.0f : run_job.cfg;
            uint32_t btotal = run_job.batch_count;
            if (btotal < 1u) btotal = 1u;
            if (btotal > DIFF_MAX_BATCH_GEN) btotal = DIFF_MAX_BATCH_GEN;

            if (run_job.batch_folder[0]) {
                gallery_create_folder(run_job.batch_folder, run_job.batch_folder, sizeof(run_job.batch_folder));
            }

            // Publish active in-progress job settings so browser reload / second PC immediately sees exact active settings!
            g_job_start_us = sceKernelGetProcessTime();
            g_eta_done_items = 0;
            g_eta_done_sec = 0.0;
            g_eta_lat = (uint64_t)(rw / 8u) * (rh / 8u);
            g_eta_cfg = rcfg;
            g_eta_ups = run_job.upscale;
            g_eta_eco = run_job.eco_mode;
            g_eta_gal_batch = 0;

            http_srv_lock();
            snprintf(g_st.cur_prompt, sizeof(g_st.cur_prompt), "%s", run_job.prompt);
            snprintf(g_st.cur_negative, sizeof(g_st.cur_negative), "%s", run_job.negative);
            snprintf(g_st.cur_lora, sizeof(g_st.cur_lora), "%s", run_job.lora);
            snprintf(g_st.cur_folder, sizeof(g_st.cur_folder), "%s", run_job.batch_folder);
            g_st.cur_lora_scale = run_job.lora_scale;
            g_st.cur_w = rw;
            g_st.cur_h = rh;
            g_st.cur_steps = rsteps;
            g_st.cur_cfg = rcfg;
            g_st.cur_sampler = run_job.sampler;
            g_st.cur_upscale = run_job.upscale;
            g_st.cur_eco = run_job.eco_mode;
            g_st.cur_seed = run_job.seed;
            g_st.cur_batch = btotal;
            g_st.cur_rand_mask = run_job.batch_rand_mask;
            g_st.nsfw_allow = run_job.nsfw_allow;
            g_st.auto_quality = run_job.auto_quality;
            g_st.eta_sec = 0.0f;
            g_st.elapsed_sec = 0.0f;
            g_st.settings_seq++;
            http_srv_unlock();

            int batch_ok = 1;
            for (uint32_t bi = 0; batch_ok && bi < btotal; bi++) {
                http_srv_lock();
                int cancelled = g_job.cancel_req;
                g_st.batch_cur = bi + 1u;
                g_st.batch_total = btotal;
                g_st.progress = 0.0f;
                http_srv_unlock();
                if (cancelled) { batch_ok = 0; break; }

                uint64_t seed = (bi == 0 && run_job.seed)
                    ? run_job.seed
                    : ((sceKernelReadTsc() ^ ((uint64_t)(bi + 1u) * 2654435761ull)) & 0xFFFFFFFFull);
                if (seed == 0) seed = 1;
                sm_last_seed = seed;

                // Per-image wildcard + 500-word SFW/Ecchi randomizer + Auto-Quality & Hand/Eye Anatomy Shield
                char iter_prompt[1024], iter_neg[512], clean_prompt[1024], lora_spec[384];
                static uint64_t s_rand_nonce = 0;
                uint64_t rseed = seed ^ sceKernelReadTsc() ^ ((++s_rand_nonce + (uint64_t)bi) * 0x9E3779B97F4A7C15ull);
                sdxl_expand_wildcards(run_job.prompt, rseed, run_job.nsfw_allow, run_job.batch_rand_mask,
                                      iter_prompt, sizeof(iter_prompt));
                snprintf(iter_neg, sizeof(iter_neg), "%s", run_job.negative);
                sdxl_apply_quality_shield(eng.model_name, eng.is_sd15, run_job.auto_quality, run_job.nsfw_allow,
                                          iter_prompt, sizeof(iter_prompt), iter_neg, sizeof(iter_neg));
                sdxl_extract_loras(iter_prompt, run_job.lora, run_job.lora_scale,
                                   clean_prompt, sizeof(clean_prompt), lora_spec, sizeof(lora_spec));
                int lrc = sdxl_sync_loras(&eng, LORAS_ROOT, lora_spec, on_progress, NULL);
                http_srv_lock();
                snprintf(g_st.active_lora, sizeof(g_st.active_lora), "%s", eng.applied_lora_spec);
                http_srv_unlock();
                if (lrc != 0) {
                    http_srv_lock();
                    snprintf(g_st.error, sizeof(g_st.error), "%s", eng.error[0] ? eng.error : "LoRA error");
                    http_srv_unlock();
                    batch_ok = 0;
                    break;
                }
                SdxlParams p = {clean_prompt, iter_neg, rw, rh, rsteps, rcfg, seed, run_job.eco_mode, run_job.sampler};
                uint8_t* rgb = (uint8_t*)malloc((size_t)p.width * p.height * 3);
                static const char* SMP_LOG2[4] = {"EulerA", "Euler", "LCM", "DPM2MKarras"};
                glog("gen start [%u/%u] %s %ux%u steps=%u cfg=%.1f %s %s seed=%llu",
                     bi + 1u, btotal, eng.is_sd15 ? "SD1.5" : "SDXL", p.width, p.height, p.steps, (double)p.cfg_scale,
                     SMP_LOG2[(p.sampler >= 0 && p.sampler <= 3) ? p.sampler : 0], sdxl_eco_name_ex(p.eco_mode, eng.neo), (unsigned long long)seed);
                int rc = rgb ? sdxl_generate(&eng, &p, rgb, on_progress, on_cancel, NULL) : -1;
                uint32_t out_w = p.width, out_h = p.height;
                if (rc == 0 && run_job.upscale > 0 && eng.has_upscaler) {
                    uint8_t* up_rgb = NULL;
                    uint32_t uw = 0, uh = 0;
                    int ur = sdxl_upscale_rgb(&eng, rgb, p.width, p.height, &up_rgb, &uw, &uh,
                                              run_job.upscale, p.eco_mode, on_progress, on_cancel, NULL);
                    if (ur == 0 && up_rgb) {
                        free(rgb);
                        rgb = up_rgb;
                        out_w = uw;
                        out_h = uh;
                        eng.last_gen_sec += eng.stats.upscale_ms / 1000.0;
                    } else if (ur == 1) {
                        rc = 1;
                    }
                }
                if (rc == 0) {
                    static const char* SMP_META[4] = {"Euler a", "Euler", "LCM", "DPM++ 2M Karras"};
                    const char* smp_str = SMP_META[(p.sampler >= 0 && p.sampler <= 3) ? p.sampler : 0];
                    int did_up = (out_w != p.width || out_h != p.height) ? run_job.upscale : 0;
                    char pmeta[2048];
                    format_a1111_params(pmeta, sizeof(pmeta), iter_prompt, p.negative, p.steps, smp_str,
                                        p.cfg_scale, seed, out_w, out_h, eng.model_name, eng.applied_lora_spec, did_up);
                    size_t png_sz = 0;
                    uint8_t* png = png_encode_rgb_meta(rgb, out_w, out_h, pmeta, &png_sz);
                    char saved_id[64] = {0};
                    if (png && png_sz) {
                        GalleryItem gm;
                        memset(&gm, 0, sizeof(gm));
                        snprintf(gm.model, sizeof(gm.model), "%s", eng.model_name);
                        snprintf(gm.prompt, sizeof(gm.prompt), "%s", iter_prompt);
                        snprintf(gm.negative, sizeof(gm.negative), "%s", p.negative);
                        snprintf(gm.sampler, sizeof(gm.sampler), "%s", smp_str);
                        snprintf(gm.lora, sizeof(gm.lora), "%s", eng.applied_lora_spec);
                        snprintf(gm.folder, sizeof(gm.folder), "%s", run_job.batch_folder);
                        gm.width = out_w; gm.height = out_h; gm.steps = p.steps;
                        gm.cfg = p.cfg_scale; gm.seed = seed; gm.eco = p.eco_mode;
                        gm.upscaled = did_up;
                        gm.sec = eng.last_gen_sec;
                        http_srv_lock();
                        snprintf(g_st.stage, sizeof(g_st.stage), "Staging PNG in RAM buffer...");
                        http_srv_unlock();
                        if (gallery_stage_enqueue(png, png_sz, rgb, out_w, out_h, &gm, saved_id, sizeof(saved_id)) == 0) {
                            glog("gallery staged %s (%ux%u)", saved_id, out_w, out_h);
                        }
                    }
                    publish_preview_rgb(rgb, out_w, out_h);
                    http_srv_lock();
                    free(g_st.png_data);
                    g_st.png_data = png;
                    g_st.png_size = png_sz;
                    g_st.width = out_w;
                    g_st.height = out_h;
                    g_st.steps = p.steps;
                    g_st.cfg = p.cfg_scale;
                    g_st.seed = seed;
                    g_st.eco_mode = p.eco_mode;
                    g_st.last_sec = eng.last_gen_sec;
                    g_st.unet_gpu_sec = eng.stats.unet_gpu_ms / 1000.0;
                    g_st.vae_sec = eng.stats.vae_ms / 1000.0;
                    g_st.upscale_sec = eng.stats.upscale_ms / 1000.0;
                    g_st.cooldown_sec = eng.stats.cooldown_ms / 1000.0;
                    g_st.eff_tflops = eng.stats.effective_tflops;
                    g_st.temp_peak_c = eng.stats.temp_peak_c;
                    g_st.image_gen_id++;
                    if (saved_id[0]) snprintf(g_st.last_gallery_id, sizeof(g_st.last_gallery_id), "%s", saved_id);
                    if (btotal > 1u) {
                        snprintf(g_st.stage, sizeof(g_st.stage), "Done [%u/%u] %ux%u in %.1fs (%.2f TF | %s)",
                                 bi + 1u, btotal, out_w, out_h, eng.last_gen_sec, eng.stats.effective_tflops,
                                 sdxl_eco_name_ex(p.eco_mode, eng.neo));
                    } else {
                        snprintf(g_st.stage, sizeof(g_st.stage), "Done %ux%u in %.1fs (%.2f TF | %s)",
                                 out_w, out_h, eng.last_gen_sec, eng.stats.effective_tflops,
                                 sdxl_eco_name_ex(p.eco_mode, eng.neo));
                    }
                    http_srv_unlock();
                    glog("gen done [%u/%u] %ux%u in %.1fs (%zu bytes PNG)", bi + 1u, btotal, out_w, out_h, eng.last_gen_sec, png_sz);
                    g_eta_done_items++;
                    g_eta_done_sec += eng.last_gen_sec;
                } else {
                    free(rgb);
                    http_srv_lock();
                    if (rc == 1) snprintf(g_st.stage, sizeof(g_st.stage), btotal > 1u ? "Batch cancelled at [%u/%u]" : "Cancelled", bi + 1u, btotal);
                    else snprintf(g_st.error, sizeof(g_st.error), "%s", eng.error[0] ? eng.error : "generation error");
                    http_srv_unlock();
                    glog("gen ended rc=%d (%s)", rc, eng.error);
                    batch_ok = 0;
                    break;
                }
            }
            int do_shutdown = 0;
            g_job_start_us = 0;
            http_srv_lock();
            if (batch_ok && g_job.shutdown_after) do_shutdown = 1;
            g_st.batch_cur = 0;
            g_st.batch_total = 0;
            g_st.eta_sec = 0.0f;
            if (!do_shutdown) g_st.busy = 0;
            http_srv_unlock();

            if (do_shutdown) {
                int aborted = 0;
                for (int sec = 10; sec > 0 && !aborted; sec--) {
                    http_srv_lock();
                    snprintf(g_st.stage, sizeof(g_st.stage), "Batch done! Shutting down PS4 in %ds... (O / Cancel to abort)", sec);
                    g_st.progress = (float)sec / 10.0f;
                    http_srv_unlock();
                    for (int t = 0; t < 10; t++) {
                        if (on_cancel(NULL)) { aborted = 1; break; }
                        http_srv_lock();
                        if (!g_job.shutdown_after) aborted = 1;
                        http_srv_unlock();
                        if (aborted) break;
                        sceKernelUsleep(100000);
                    }
                }
                http_srv_lock();
                g_job.shutdown_after = 0;
                g_st.shutdown_after = 0;
                g_job.cancel_req = 0;
                if (aborted) {
                    snprintf(g_st.stage, sizeof(g_st.stage), "Auto-shutdown aborted");
                    g_st.busy = 0;
                }
                http_srv_unlock();
                if (!aborted) {
                    glog("auto-shutdown: flushing RAM staging buffer and HDD before powering off PS4...");
                    gallery_stage_flush_all();
                    sync();
                    sceKernelUsleep(500000);
                    OrbisKernelEventFlag evf = NULL;
                    if (sceKernelOpenEventFlag(&evf, "SceSysCoreReboot") == 0 && evf) {
                        sceKernelSetEventFlag(evf, 0x4000);
                        sceKernelCloseEventFlag(evf);
                    }
                    // Fallback: SIGUSR2 (31) to init (PID 1) triggers clean OrbisOS power-off
                    syscall(37, 1, 31);
                    for (;;) sceKernelUsleep(500000);
                }
            }
        }


        if (!gop && !sw_name[0] && !do_run) {
            sceKernelUsleep(20000);
        }
    }
    return NULL;
}

int main(void) {
    __libc.threads_minus_1 = 1;
    if (rename(LOG_FILE, "/data/garlic_diffusion_log_prev.txt")) unlink(LOG_FILE);
    glog("GarlicDiffusion v1.15 starting (GPLv3 · 60FPS Non-Blocking UI + Easy/Adv Studio + GitHub OTA)");
    log_console_info();
    gallery_init(OUT_ROOT, PRESETS_ROOT);
    gallery_stage_init(sceKernelIsNeoMode() ? (64u << 20) : (32u << 20));
    styles_init(STYLES_ROOT);
    characters_init(CHARS_ROOT);
    model_dl_init(MODELS_ROOT, REPO_CFG);
    lora_dl_init(LORAS_ROOT);
    refresh_tv_presets();
    g_num_styles = styles_list(g_styles, STYLES_MAX_ITEMS);
    g_num_chars = characters_list(g_chars, CHARS_MAX_ITEMS);
    g_num_loras = lora_list_installed(g_loras, LORA_MAX_INSTALLED);
    sceSysmoduleLoadModule(ORBIS_SYSMODULE_IME_DIALOG);
    sceCommonDialogInitialize();
    sceUserServiceInitialize(NULL);
    sceUserServiceGetInitialUser(&user_id);
    scePadInit();
    pad_fd = scePadOpen(user_id, 0, 0, NULL);
    if (video_init()) { glog("video_init failed"); return 1; }
    sceSystemServiceHideSplashScreen();

    memset(&g_job, 0, sizeof(g_job));
    memset(&g_st, 0, sizeof(g_st));
    if (g_num_presets > 0) {
        snprintf(g_job.prompt, sizeof(g_job.prompt), "%s", g_presets[0].prompt);
        snprintf(g_job.negative, sizeof(g_job.negative), "%s", g_presets[0].negative);
    } else {
        snprintf(g_job.prompt, sizeof(g_job.prompt),
                 "masterpiece, best quality, 1girl, silver hair, blue eyes, glowing sakura petals, night sky, anime style");
        snprintf(g_job.negative, sizeof(g_job.negative),
                 "lowres, bad anatomy, bad hands, text, error, blurry, worst quality, low quality");
    }
    g_job.width = 768;
    g_job.height = 768;
    g_job.steps = 20;
    g_job.cfg = 7.0f;
    g_job.lora_scale = 0.80f;
    g_job.seed = 0;
    g_job.batch_count = 1;
    g_job.nsfw_allow = 0;   // Default: SFW Guard ON
    g_job.auto_quality = 1; // Default: Auto-Quality & Hand/Eye Shield ON
    g_job.ui_mode = load_ui_mode_cfg(); // -1 = show 2-choice Welcome Screen, 0 = Easy, 1 = Advanced
    g_st.nsfw_allow = 0;
    g_st.auto_quality = 1;
    g_st.ui_mode = g_job.ui_mode;
    g_st.neo = sceKernelIsNeoMode();
    g_st.busy = 1;
    g_job.eco_mode = g_st.neo ? 2 : 1;
    g_st.eco_mode = g_job.eco_mode;
    snprintf(g_st.stage, sizeof(g_st.stage), "Initializing PS4 GPU & loading checkpoint in background...");

    http_srv_start(&g_job, &g_st);

    // Spawn dedicated background GPU thread so PS4 TV UI runs at silky-smooth 60 FPS at all times!
    OrbisPthread gpu_th;
    OrbisPthreadAttr gattr;
    scePthreadAttrInit(&gattr);
    scePthreadAttrSetstacksize(&gattr, 1024 * 1024);
    if (scePthreadCreate(&gpu_th, &gattr, gpu_worker_main, NULL, "gdiff_gpu") != 0) {
        glog("FATAL: failed to create gdiff_gpu thread");
        return 1;
    }
    scePthreadAttrDestroy(&gattr);

    draw_ui();
    uint32_t prev_btn = 0;
    uint32_t hold_dir = 0;
    int hold_ticks = 0;
    int idle_ticks = 0;
    for (;;) {
        int ui_dirty = 0;
        OrbisPadData pd;
        uint32_t btn = 0, dir_now = 0;
        if (pad_fd >= 0 && scePadReadState(pad_fd, &pd) == 0) {
            btn = pd.buttons;
            dir_now = btn & (ORBIS_PAD_BUTTON_UP | ORBIS_PAD_BUTTON_DOWN | ORBIS_PAD_BUTTON_LEFT | ORBIS_PAD_BUTTON_RIGHT);
            if (pd.leftStick.y < 56) dir_now |= ORBIS_PAD_BUTTON_UP;
            else if (pd.leftStick.y > 200) dir_now |= ORBIS_PAD_BUTTON_DOWN;
            if (pd.leftStick.x < 56) dir_now |= ORBIS_PAD_BUTTON_LEFT;
            else if (pd.leftStick.x > 200) dir_now |= ORBIS_PAD_BUTTON_RIGHT;
        }
        uint32_t pressed = btn & ~prev_btn;
        prev_btn = btn;

        // Smooth directional auto-repeat (D-Pad + Left Stick): immediate on edge, repeat after ~290ms every ~80ms
        uint32_t nav = 0;
        if (dir_now != hold_dir) {
            nav = dir_now & ~hold_dir;
            hold_dir = dir_now;
            hold_ticks = 0;
        } else if (dir_now != 0) {
            hold_ticks++;
            if (hold_ticks >= 18 && ((hold_ticks - 18) % 5) == 0) nav = dir_now;
        }

        // Reload user presets, styles, characters, or rescan models if requested by Web UI
        http_srv_lock();
        int pdirty = g_job.presets_dirty; g_job.presets_dirty = 0;
        int sdirty = g_job.styles_dirty;  g_job.styles_dirty = 0;
        int cdirty = g_job.chars_dirty;   g_job.chars_dirty = 0;
        int ldirty = g_job.loras_dirty;   g_job.loras_dirty = 0;
        int do_rescan = g_job.rescan_req;
        http_srv_unlock();
        if (pdirty) refresh_tv_presets();
        if (sdirty) { g_num_styles = styles_list(g_styles, STYLES_MAX_ITEMS); if (style_idx >= g_num_styles) style_idx = -1; }
        if (cdirty) {
            char keep_cid[64] = {0};
            if (char_idx >= 0 && char_idx < g_num_chars) snprintf(keep_cid, sizeof(keep_cid), "%s", g_chars[char_idx].id);
            g_num_chars = characters_list(g_chars, CHARS_MAX_ITEMS);
            if (keep_cid[0]) {
                int found = -1;
                for (int ci = 0; ci < g_num_chars; ci++) {
                    if (strcmp(g_chars[ci].id, keep_cid) == 0) { found = ci; break; }
                }
                char_idx = found;
            } else if (char_idx >= g_num_chars) {
                char_idx = -1;
            }
        }
        if (ldirty) g_num_loras = lora_list_installed(g_loras, LORA_MAX_INSTALLED);
        if (do_rescan) {
            int nm = scan_models();
            g_num_loras = lora_list_installed(g_loras, LORA_MAX_INSTALLED);
            http_srv_lock();
            g_job.rescan_req = 0;
            http_srv_unlock();
            glog("rescanned models: found %d", nm);
            if (g_in_models) models_mgr_refresh();
            ui_dirty = 1;
        }
        // Auto-rescan when an in-app HTTPS model download completes
        {
            ModelDlStatus dl;
            model_dl_get_status(&dl);
            if (dl.finished != g_last_dl_fin) {
                g_last_dl_fin = dl.finished;
                scan_models();
                g_num_loras = lora_list_installed(g_loras, LORA_MAX_INSTALLED);
                if (g_in_models) models_mgr_refresh();
                int auto_sw = 0;
                http_srv_lock();
                if (!g_st.ready && g_st.gpu_ready && !g_st.busy && !g_job.switch_req) {
                    for (int mi = 0; mi < g_st.num_models; mi++) {
                        if (strcmp(g_st.models[mi], dl.model_id) == 0) {
                            snprintf(g_job.switch_model, sizeof(g_job.switch_model), "%s", dl.model_id);
                            g_job.switch_req = 1;
                            auto_sw = 1;
                            break;
                        }
                    }
                }
                http_srv_unlock();
                if (auto_sw) {
                    g_in_models = 0;
                    snprintf(toast, sizeof(toast), "Downloaded %s — auto-loading onto PS4 GPU!", dl.model_id);
                } else {
                    snprintf(toast, sizeof(toast), "Downloaded %s!", dl.model_id);
                }
                ui_dirty = 1;
            }
            if (g_in_models && dl.active) ui_dirty = 1;
            AppUpdateStatus u_chk;
            app_update_get_status(&u_chk);
            if (u_chk.update_available && !g_ota_notified) {
                g_ota_notified = 1;
                snprintf(toast, sizeof(toast), "Update v%s available! Open Credits (O) to download to /data/pkg", u_chk.remote_ver);
                ui_dirty = 1;
            }
        }

        int cur_uim = 1, cur_busy = 0, cur_has_up = 0;
        http_srv_lock();
        cur_uim = g_job.ui_mode;
        cur_busy = g_st.busy;
        cur_has_up = g_st.has_upscaler;
        http_srv_unlock();

        if (cur_uim < 0) {
            if (nav & (ORBIS_PAD_BUTTON_LEFT | ORBIS_PAD_BUTTON_RIGHT)) {
                g_welcome_sel ^= 1;
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_CROSS) {
                int chosen = g_welcome_sel ? 1 : 0;
                int zero_models = 0, neo_now = 1;
                http_srv_lock();
                g_job.ui_mode = chosen;
                g_st.ui_mode = chosen;
                g_st.settings_seq++;
                zero_models = (!g_st.ready && g_st.num_models == 0);
                neo_now = g_st.neo;
                http_srv_unlock();
                save_ui_mode_cfg(chosen);
                sm_focus = 0;
                if (chosen == 0) apply_tv_easy_framing_and_quality();
                if (zero_models) {
                    open_starter_models_mgr(neo_now);
                    snprintf(toast, sizeof(toast), "No checkpoints installed yet — press X on a ★ STARTER model to download!");
                } else {
                    snprintf(toast, sizeof(toast), chosen == 0
                             ? "Easy Mode active! Press R1 anytime for Advanced Studio."
                             : "Advanced Studio active! Press R1 anytime for Easy Mode.");
                }
                ui_dirty = 1;
            }
        } else if (g_in_credits) {
            if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                g_in_credits = 0; toast[0] = 0; ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) {
                http_srv_lock();
                g_job.nsfw_allow = !g_job.nsfw_allow;
                g_st.nsfw_allow = g_job.nsfw_allow;
                int ns = g_job.nsfw_allow;
                g_st.settings_seq++;
                http_srv_unlock();
                refresh_tv_presets();
                snprintf(toast, sizeof(toast), ns
                         ? "NSFW Mode ENABLED (Ecchi & Research Presets Unlocked)"
                         : "SFW Guard ENABLED (Default)");
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_L3) {
                AppUpdateStatus up;
                app_update_get_status(&up);
                if (up.update_available && !up.pkg_ready) {
                    if (app_update_download_pkg() == 0) {
                        g_in_credits = 0; g_in_models = 1; models_mgr_refresh();
                        snprintf(toast, sizeof(toast), "Downloading Garlic-Diffusion.pkg to /data/pkg...");
                    } else {
                        snprintf(toast, sizeof(toast), "Another download is already running");
                    }
                } else {
                    snprintf(toast, sizeof(toast), "Checking GitHub for updates...");
                    draw_ui();
                    app_update_check_now();
                    app_update_get_status(&up);
                    if (up.update_available)
                        snprintf(toast, sizeof(toast), "Update v%s found! Press L3 again to download to /data/pkg", up.remote_ver);
                    else
                        snprintf(toast, sizeof(toast), "You are on the latest version (v%s)", GD_APP_VERSION);
                }
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_R3) {
                AppUpdateStatus up;
                app_update_get_status(&up);
                if (up.pkg_ready) {
                    app_update_prepare_and_exit(toast, sizeof(toast));
                } else {
                    g_in_credits = 0; g_in_models = 1; g_mdl_sel = 0; models_mgr_refresh();
                }
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_SQUARE) {
                g_in_credits = 0; g_in_gallery = 1; g_gal_full = 0; g_gal_idx = 0; gal_refresh(); ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_OPTIONS) {
                g_in_credits = 0; g_in_models = 1; g_mdl_sel = 0; models_mgr_refresh(); ui_dirty = 1;
            }
        } else if (g_in_models) {
            int total = g_mgr_tab ? (g_lora_sub ? g_lcat_cnt : g_num_loras) : (g_inst_cnt + g_cat_cnt);
            if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                g_in_models = 0; g_mdl_del_arm = 0; toast[0] = 0; ui_dirty = 1;
            } else if (pressed & (ORBIS_PAD_BUTTON_L1 | ORBIS_PAD_BUTTON_R1)) {
                g_mgr_tab ^= 1; g_mdl_sel = 0; g_mdl_del_arm = 0; toast[0] = 0;
                models_mgr_refresh(); ui_dirty = 1;
            } else if (g_mgr_tab && (pressed & (ORBIS_PAD_BUTTON_L2 | ORBIS_PAD_BUTTON_R2))) {
                g_lora_sub ^= 1; g_mdl_sel = 0; g_mdl_del_arm = 0; toast[0] = 0;
                models_mgr_refresh(); ui_dirty = 1;
            } else if (g_mgr_tab && (nav & ORBIS_PAD_BUTTON_RIGHT) && total > LORA_TV_PAGE_SZ) {
                int np = (g_mdl_sel / LORA_TV_PAGE_SZ + 1) * LORA_TV_PAGE_SZ;
                g_mdl_sel = (np < total) ? np : 0;
                g_mdl_del_arm = 0; ui_dirty = 1;
            } else if (g_mgr_tab && (nav & ORBIS_PAD_BUTTON_LEFT) && total > LORA_TV_PAGE_SZ) {
                int cur_p = g_mdl_sel / LORA_TV_PAGE_SZ;
                int last_p = (total - 1) / LORA_TV_PAGE_SZ;
                g_mdl_sel = (cur_p > 0 ? (cur_p - 1) : last_p) * LORA_TV_PAGE_SZ;
                g_mdl_del_arm = 0; ui_dirty = 1;
            } else if ((nav & ORBIS_PAD_BUTTON_DOWN) && total > 0) {
                g_mdl_sel = (g_mdl_sel + 1) % total; g_mdl_del_arm = 0; ui_dirty = 1;
            } else if ((nav & ORBIS_PAD_BUTTON_UP) && total > 0) {
                g_mdl_sel = (g_mdl_sel + total - 1) % total; g_mdl_del_arm = 0; ui_dirty = 1;
            } else if (g_mgr_tab && (pressed & ORBIS_PAD_BUTTON_SQUARE)) {
                int is15_now = 0;
                http_srv_lock(); is15_now = g_st.is_sd15; http_srv_unlock();
                int qn = lora_dl_queue_all(is15_now ? "sd15" : "sdxl");
                if (qn > 0) snprintf(toast, sizeof(toast), "Queued %d compatible %s LoRA(s) for download!", qn, is15_now ? "SD 1.5" : "SDXL");
                else if (qn == 0) snprintf(toast, sizeof(toast), "All compatible %s LoRAs are already installed!", is15_now ? "SD 1.5" : "SDXL");
                else snprintf(toast, sizeof(toast), "Cannot queue LoRAs while a checkpoint download is active");
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) {
                model_dl_cancel(); snprintf(toast, sizeof(toast), "Cancelled download & cleared queue"); ui_dirty = 1;
            } else if ((pressed & ORBIS_PAD_BUTTON_CROSS) && total > 0) {
                if (!g_mgr_tab) {
                    if (g_mdl_sel < g_inst_cnt) {
                        http_srv_lock();
                        snprintf(g_job.switch_model, sizeof(g_job.switch_model), "%s", g_inst[g_mdl_sel].name);
                        g_job.switch_req = 1;
                        http_srv_unlock();
                        g_in_models = 0; toast[0] = 0; ui_dirty = 1;
                    } else {
                        const CatalogModel* cm = &g_cat[g_mdl_sel - g_inst_cnt];
                        int rc = model_dl_start(cm->id, cm->repo, cm->files);
                        if (rc == 0) snprintf(toast, sizeof(toast), "Downloading %s...", cm->id);
                        else {
                            ModelDlStatus dl; model_dl_get_status(&dl);
                            snprintf(toast, sizeof(toast), "%s", dl.error[0] ? dl.error : "Cannot start download");
                        }
                        ui_dirty = 1;
                    }
                } else if (!g_lora_sub) {
                    int is15_now = 0;
                    http_srv_lock(); is15_now = g_st.is_sd15; http_srv_unlock();
                    if (!lora_arch_compat(g_loras[g_mdl_sel].arch, is15_now)) {
                        snprintf(toast, sizeof(toast), "Blocked: %s is %s (active model is %s)",
                                 g_loras[g_mdl_sel].name, g_loras[g_mdl_sel].arch, is15_now ? "SD 1.5" : "SDXL");
                    } else {
                        http_srv_lock();
                        if (strcmp(g_job.lora, g_loras[g_mdl_sel].name) == 0) {
                            g_job.lora[0] = 0;
                            snprintf(toast, sizeof(toast), "Cleared Studio LoRA");
                        } else {
                            snprintf(g_job.lora, sizeof(g_job.lora), "%s", g_loras[g_mdl_sel].name);
                            if (fabsf(g_job.lora_scale) < 0.05f) g_job.lora_scale = 0.80f;
                            snprintf(toast, sizeof(toast), "Selected LoRA: %s (%.2f)", g_job.lora, (double)g_job.lora_scale);
                        }
                        g_st.settings_seq++;
                        http_srv_unlock();
                    }
                    ui_dirty = 1;
                } else {
                    const CatalogLora* cl = &g_lcat[g_mdl_sel];
                    if (lora_is_installed_cat(cl)) {
                        snprintf(toast, sizeof(toast), "%s is already installed (press L2/R2 for Installed view)", cl->title);
                    } else {
                        int rc = lora_dl_start(cl->id, cl->url, cl->filename);
                        if (rc == 0) snprintf(toast, sizeof(toast), "Downloading LoRA %s...", cl->id);
                        else if (rc == 1) {
                            ModelDlStatus dl; model_dl_get_status(&dl);
                            snprintf(toast, sizeof(toast), "Queued LoRA %s (%d in queue)", cl->id, dl.queue_len);
                        } else {
                            ModelDlStatus dl; model_dl_get_status(&dl);
                            snprintf(toast, sizeof(toast), "%s", dl.error[0] ? dl.error : "Cannot start LoRA download");
                        }
                    }
                    ui_dirty = 1;
                }
            } else if ((pressed & ORBIS_PAD_BUTTON_OPTIONS) &&
                       ((!g_mgr_tab && g_mdl_sel < g_inst_cnt) || (g_mgr_tab && !g_lora_sub && g_mdl_sel < g_num_loras))) {
                const char* dname = g_mgr_tab ? g_loras[g_mdl_sel].name : g_inst[g_mdl_sel].name;
                if (!g_mdl_del_arm) {
                    g_mdl_del_arm = 1;
                    snprintf(toast, sizeof(toast), "Press [OPTIONS] again to DELETE %s", dname);
                } else if (!g_mgr_tab) {
                    http_srv_lock();
                    char act[64]; snprintf(act, sizeof(act), "%s", g_st.model);
                    http_srv_unlock();
                    int rc = model_dl_delete(dname, act);
                    g_mdl_del_arm = 0;
                    if (rc == 0) { scan_models(); models_mgr_refresh(); snprintf(toast, sizeof(toast), "Deleted model"); }
                    else if (rc == -2) snprintf(toast, sizeof(toast), "Switch away from active model before deleting");
                    else snprintf(toast, sizeof(toast), "Delete failed");
                } else {
                    int rc = lora_delete(dname);
                    g_mdl_del_arm = 0;
                    if (rc == 0) {
                        http_srv_lock();
                        if (strcmp(g_job.lora, dname) == 0) g_job.lora[0] = 0;
                        http_srv_unlock();
                        models_mgr_refresh();
                        snprintf(toast, sizeof(toast), "Deleted LoRA");
                    } else snprintf(toast, sizeof(toast), "Delete failed");
                }
                ui_dirty = 1;
            }
        } else if (g_in_gallery) {
            if (g_gal_full) {
                if (pressed) { g_gal_full = 0; ui_dirty = 1; }
            } else if (g_gal_amodal) {
                if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                    g_gal_amodal = 0; ui_dirty = 1;
                } else if (nav & ORBIS_PAD_BUTTON_DOWN) {
                    g_gal_asel = (g_gal_asel + 1) % 6; ui_dirty = 1;
                } else if (nav & ORBIS_PAD_BUTTON_UP) {
                    g_gal_asel = (g_gal_asel + 5) % 6; ui_dirty = 1;
                } else if ((pressed & ORBIS_PAD_BUTTON_CROSS) && g_gal_count > 0) {
                    const GalleryItem* it = &g_gal[g_gal_idx];
                    g_gal_amodal = 0;
                    if (g_gal_asel == 0) {
                        g_num_folders = gallery_list_folders(g_folders, GALLERY_MAX_FOLDERS, NULL, NULL, NULL);
                        g_gal_fsel = (g_num_folders > 0) ? 2 : 0;
                        for (int k = 0; k < g_num_folders; k++)
                            if (strcmp(g_folders[k].name, it->folder) == 0) { g_gal_fsel = 2 + k; break; }
                        g_gal_ftop = (g_gal_fsel >= 9) ? (g_gal_fsel - 8) : 0;
                        g_gal_fmodal = 1;
                    } else if (g_gal_asel == 1) {
                        if (g_gal_rgb) g_gal_full = 1;
                    } else if (g_gal_asel == 2 || g_gal_asel == 3) {
                        int run_now = (g_gal_asel == 3);
                        http_srv_lock();
                        int same_mdl = (!it->model[0] || strcmp(it->model, g_st.model) == 0);
                        int has_mdl = (!same_mdl && model_is_installed_name(it->model));
                        if (has_mdl) {
                            g_pending_gal_item = *it;
                            g_pending_gal_restore = 1;
                            g_pending_gal_run = run_now;
                            snprintf(g_job.switch_model, sizeof(g_job.switch_model), "%s", it->model);
                            g_job.switch_req = 1;
                            g_st.settings_seq++;
                        } else {
                            apply_gallery_item_to_job_locked(it, run_now, g_st.is_sd15);
                            if (run_now) {
                                g_job.cancel_req = 0;
                                g_job.pending = 1;
                            }
                        }
                        http_srv_unlock();
                        if (has_mdl) {
                            snprintf(toast, sizeof(toast), "Switching to %s & restoring all settings...", it->model);
                        } else if (!same_mdl && it->model[0]) {
                            snprintf(toast, sizeof(toast), "Model '%s' not installed — loaded settings on active model", it->model);
                        } else if (!run_now) {
                            snprintf(toast, sizeof(toast), "Loaded all settings from %s", it->id);
                        } else {
                            toast[0] = 0;
                        }
                        g_in_gallery = 0;
                    } else if (g_gal_asel == 4 && cur_has_up) {
                        http_srv_lock();
                        g_job.gal_ids_n = 1;
                        snprintf(g_job.gal_ids[0], sizeof(g_job.gal_ids[0]), "%s", it->id);
                        snprintf(g_job.gal_id, sizeof(g_job.gal_id), "%s", it->id);
                        g_job.gal_op = g_gal_up_mode;
                        g_job.cancel_req = 0;
                        http_srv_unlock();
                        g_in_gallery = 0; toast[0] = 0;
                    } else if (g_gal_asel == 5) {
                        int nxt_star = !it->starred;
                        gallery_set_starred(it->id, nxt_star);
                        snprintf(toast, sizeof(toast), nxt_star ? "Starred artwork!" : "Removed star");
                        gal_refresh();
                    }
                    ui_dirty = 1;
                }
            } else if (g_gal_fmodal) {
                int fn_tot = 2 + g_num_folders;
                if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                    g_gal_fmodal = 0; ui_dirty = 1;
                } else if (nav & ORBIS_PAD_BUTTON_DOWN) {
                    g_gal_fsel = (g_gal_fsel + 1) % fn_tot;
                    if (g_gal_fsel < g_gal_ftop) g_gal_ftop = g_gal_fsel;
                    if (g_gal_fsel >= g_gal_ftop + 9) g_gal_ftop = g_gal_fsel - 8;
                    ui_dirty = 1;
                } else if (nav & ORBIS_PAD_BUTTON_UP) {
                    g_gal_fsel = (g_gal_fsel + fn_tot - 1) % fn_tot;
                    if (g_gal_fsel < g_gal_ftop) g_gal_ftop = g_gal_fsel;
                    if (g_gal_fsel >= g_gal_ftop + 9) g_gal_ftop = g_gal_fsel - 8;
                    ui_dirty = 1;
                } else if (pressed & ORBIS_PAD_BUTTON_CROSS) {
                    char clean_f[64] = {0};
                    int apply_mv = 0;
                    g_gal_fmodal = 0;
                    if (g_gal_fsel == 0) {
                        char raw_f[64] = "";
                        if (edit_text_osk(L"New Folder Name", raw_f, sizeof(raw_f), 0) &&
                            gallery_clean_folder_name(raw_f, clean_f, sizeof(clean_f)) > 0) {
                            gallery_create_folder(clean_f, clean_f, sizeof(clean_f));
                            apply_mv = 1;
                        }
                    } else if (g_gal_fsel == 1) {
                        clean_f[0] = 0;
                        apply_mv = 1;
                    } else if (g_gal_fsel - 2 < g_num_folders) {
                        snprintf(clean_f, sizeof(clean_f), "%s", g_folders[g_gal_fsel - 2].name);
                        apply_mv = 1;
                    }
                    if (apply_mv && g_gal_count > 0) {
                        if (g_gal_sel_n > 0) {
                            int mv = gallery_set_folder_batch((const char (*)[64])g_gal_sel_ids, g_gal_sel_n, clean_f);
                            gal_clear_select();
                            snprintf(toast, sizeof(toast), clean_f[0] ? "Added %d images to [%s]" : "Moved %d images to Unsorted", mv, clean_f);
                        } else {
                            gallery_set_folder(g_gal[g_gal_idx].id, clean_f);
                            snprintf(toast, sizeof(toast), clean_f[0] ? "Added to folder [%s]" : "Moved to Unsorted", clean_f);
                        }
                        gal_refresh();
                    }
                    ui_dirty = 1;
                }
            } else if (pressed || nav) {
                static const char* const UP_M_LBL[4] = {"", "1080p HD", "2x HD", "4K UHD"};
                if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                    g_in_gallery = 0; g_del_arm = 0; toast[0] = 0; ui_dirty = 1;
                } else if (pressed & ORBIS_PAD_BUTTON_SQUARE) {
                    if (g_gal_rgb) { g_gal_full = 1; ui_dirty = 1; }
                } else if (nav & ORBIS_PAD_BUTTON_RIGHT) {
                    if (g_gal_count > 0) { g_gal_idx = (g_gal_idx + 1) % g_gal_count; toast[0] = 0; gal_load_current(); ui_dirty = 1; }
                } else if (nav & ORBIS_PAD_BUTTON_LEFT) {
                    if (g_gal_count > 0) { g_gal_idx = (g_gal_idx + g_gal_count - 1) % g_gal_count; toast[0] = 0; gal_load_current(); ui_dirty = 1; }
                } else if (nav & ORBIS_PAD_BUTTON_DOWN) {
                    if (g_gal_count > 0) { g_gal_idx = (g_gal_idx + 12 < g_gal_count) ? (g_gal_idx + 12) : (g_gal_count - 1); toast[0] = 0; gal_load_current(); ui_dirty = 1; }
                } else if (nav & ORBIS_PAD_BUTTON_UP) {
                    if (g_gal_count > 0) { g_gal_idx = (g_gal_idx >= 12) ? (g_gal_idx - 12) : 0; toast[0] = 0; gal_load_current(); ui_dirty = 1; }
                } else if ((pressed & (ORBIS_PAD_BUTTON_R1 | ORBIS_PAD_BUTTON_R3)) && g_gal_count > 0) {
                    g_num_folders = gallery_list_folders(g_folders, GALLERY_MAX_FOLDERS, NULL, NULL, NULL);
                    g_gal_fsel = (g_num_folders > 0) ? 2 : 0;
                    for (int k = 0; k < g_num_folders; k++) {
                        if (strcmp(g_folders[k].name, g_gal[g_gal_idx].folder) == 0) { g_gal_fsel = 2 + k; break; }
                    }
                    g_gal_ftop = 0;
                    if (g_gal_fsel >= 9) g_gal_ftop = g_gal_fsel - 8;
                    g_gal_fmodal = 1;
                    ui_dirty = 1;
                } else if ((pressed & ORBIS_PAD_BUTTON_R2) && g_gal_count > 0) {
                    gal_toggle_select(g_gal[g_gal_idx].id);
                    int s = gal_is_selected(g_gal[g_gal_idx].id) >= 0;
                    snprintf(toast, sizeof(toast), "%s (%d selected)", s ? "Selected artwork" : "Deselected artwork", g_gal_sel_n);
                    ui_dirty = 1;
                } else if (pressed & ORBIS_PAD_BUTTON_L2) {
                    g_gal_up_mode = (g_gal_up_mode % 3) + 1;
                    snprintf(toast, sizeof(toast), "Upscale target: %s", UP_M_LBL[g_gal_up_mode]);
                    ui_dirty = 1;
                } else if (pressed & ORBIS_PAD_BUTTON_L1) {
                    // Cycle Gallery Folder view: All -> Starred -> Unsorted -> each user folder
                    g_num_folders = gallery_list_folders(g_folders, GALLERY_MAX_FOLDERS, NULL, NULL, NULL);
                    int cur_f = 0;
                    if (strcmp(g_gal_folder, "__starred__") == 0) cur_f = 1;
                    else if (strcmp(g_gal_folder, "__unsorted__") == 0) cur_f = 2;
                    else if (g_gal_folder[0]) {
                        for (int k = 0; k < g_num_folders; k++)
                            if (strcmp(g_folders[k].name, g_gal_folder) == 0) { cur_f = 3 + k; break; }
                    }
                    int nxt = (cur_f + 1) % (3 + g_num_folders);
                    if (nxt == 0) g_gal_folder[0] = 0;
                    else if (nxt == 1) snprintf(g_gal_folder, sizeof(g_gal_folder), "__starred__");
                    else if (nxt == 2) snprintf(g_gal_folder, sizeof(g_gal_folder), "__unsorted__");
                    else snprintf(g_gal_folder, sizeof(g_gal_folder), "%s", g_folders[nxt - 3].name);
                    g_gal_idx = 0;
                    gal_refresh();
                    toast[0] = 0;
                    ui_dirty = 1;
                } else if ((pressed & ORBIS_PAD_BUTTON_CROSS) && g_gal_count > 0) {
                    g_gal_asel = 0; // Highlight "Add / Move to Folder..." first!
                    g_gal_amodal = 1;
                    ui_dirty = 1;
                } else if ((pressed & ORBIS_PAD_BUTTON_TRIANGLE) && g_gal_count > 0) {
                    const GalleryItem* it = &g_gal[g_gal_idx];
                    http_srv_lock();
                    if (it->prompt[0]) snprintf(g_job.prompt, sizeof(g_job.prompt), "%s", it->prompt);
                    if (it->negative[0]) snprintf(g_job.negative, sizeof(g_job.negative), "%s", it->negative);
                    if (it->steps) g_job.steps = it->steps;
                    if (it->cfg > 0) g_job.cfg = it->cfg;
                    g_job.sampler = strstr(it->sampler, "DPM") ? 3 : ((strcmp(it->sampler, "LCM") == 0) ? 2 : (strcmp(it->sampler, "Euler") == 0 ? 1 : 0));
                    g_job.seed = 0;
                    g_job.cancel_req = 0;
                    g_job.pending = 1;
                    g_st.settings_seq++;
                    http_srv_unlock();
                    g_in_gallery = 0; toast[0] = 0; ui_dirty = 1;
                } else if ((pressed & ORBIS_PAD_BUTTON_L3) && g_gal_count > 0 && cur_has_up) {
                    http_srv_lock();
                    if (g_gal_sel_n > 0) {
                        g_job.gal_ids_n = g_gal_sel_n;
                        for (int k = 0; k < g_gal_sel_n; k++)
                            snprintf(g_job.gal_ids[k], sizeof(g_job.gal_ids[0]), "%s", g_gal_sel_ids[k]);
                        snprintf(g_job.gal_id, sizeof(g_job.gal_id), "%s", g_gal_sel_ids[0]);
                    } else {
                        g_job.gal_ids_n = 1;
                        snprintf(g_job.gal_ids[0], sizeof(g_job.gal_ids[0]), "%s", g_gal[g_gal_idx].id);
                        snprintf(g_job.gal_id, sizeof(g_job.gal_id), "%s", g_gal[g_gal_idx].id);
                    }
                    g_job.gal_op = g_gal_up_mode;
                    g_job.cancel_req = 0;
                    http_srv_unlock();
                    gal_clear_select();
                    g_in_gallery = 0; toast[0] = 0; ui_dirty = 1;
                } else if ((pressed & ORBIS_PAD_BUTTON_OPTIONS) && g_gal_count > 0) {
                    if (!g_del_arm) {
                        g_del_arm = 1;
                        if (g_gal_sel_n > 0)
                            snprintf(toast, sizeof(toast), "Press [OPTIONS] again to BATCH DELETE %d selected images!", g_gal_sel_n);
                        else
                            snprintf(toast, sizeof(toast), "Press [OPTIONS] again to DELETE %s", g_gal[g_gal_idx].id);
                    } else if (g_gal_sel_n > 0) {
                        int del_n = 0;
                        for (int k = 0; k < g_gal_sel_n; k++)
                            if (gallery_delete(g_gal_sel_ids[k]) == 0) del_n++;
                        gal_clear_select();
                        g_del_arm = 0;
                        snprintf(toast, sizeof(toast), "Batch deleted %d images", del_n);
                        gal_refresh();
                    } else {
                        gallery_delete(g_gal[g_gal_idx].id);
                        g_del_arm = 0;
                        snprintf(toast, sizeof(toast), "Deleted image");
                        gal_refresh();
                    }
                    ui_dirty = 1;
                }
            }
        } else if (sm_dd_open) {
            // ---- DROPDOWN MODAL (Up/Down + L1/R1 page, X Select, O Cancel) ----
            if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                sm_dd_open = 0; ui_dirty = 1;
            } else if (nav & ORBIS_PAD_BUTTON_DOWN) {
                sm_dd_sel++; sm_dd_clamp(); ui_dirty = 1;
            } else if (nav & ORBIS_PAD_BUTTON_UP) {
                sm_dd_sel--; sm_dd_clamp(); ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_R1) {
                sm_dd_sel += SM_DD_VIS;
                if (sm_dd_sel >= sm_dd_n) sm_dd_sel = sm_dd_n - 1;
                sm_dd_clamp(); ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_L1) {
                sm_dd_sel -= SM_DD_VIS;
                if (sm_dd_sel < 0) sm_dd_sel = 0;
                sm_dd_clamp(); ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_CROSS) {
                int sel = sm_dd_sel, row = sm_dd_row;
                sm_dd_open = 0;
                toast[0] = 0;
                switch (row) {
                case ROW_MODEL:
                    if (sel < sm_dd_models_n) {
                        http_srv_lock();
                        snprintf(g_job.switch_model, sizeof(g_job.switch_model), "%s", sm_dd_models[sel]);
                        g_job.switch_req = 1;
                        http_srv_unlock();
                        snprintf(toast, sizeof(toast), "Loading %s...", sm_dd_models[sel]);
                    } else {
                        g_in_models = 1; g_mdl_sel = 0; g_mdl_del_arm = 0; models_mgr_refresh();
                    }
                    break;
                case ROW_LORA:
                    http_srv_lock();
                    if (sel <= 0 || sel - 1 >= g_num_loras) {
                        g_job.lora[0] = 0;
                        snprintf(toast, sizeof(toast), "LoRA: Off");
                    } else {
                        snprintf(g_job.lora, sizeof(g_job.lora), "%s", g_loras[sel - 1].name);
                        if (fabsf(g_job.lora_scale) < 0.05f) g_job.lora_scale = 0.80f;
                        snprintf(toast, sizeof(toast), "LoRA: %s (%.2f)", g_job.lora, (double)g_job.lora_scale);
                    }
                    http_srv_unlock();
                    break;
                case ROW_PRESET:
                    if (cur_uim == 0) {
                        if (sel == 0) {
                            preset_idx = -1;
                            g_easy_scene[0] = 0;
                            apply_tv_easy_framing_and_quality();
                            snprintf(toast, sizeof(toast), "Cleared scene preset (kept character)");
                        } else if (sel - 1 >= 0 && sel - 1 < g_num_presets) {
                            int keep_char = char_idx;
                            if (g_presets[sel - 1].is_speed) {
                                g_easy_quality = 0;
                                char_idx = keep_char;
                                apply_tv_easy_framing_and_quality();
                                snprintf(toast, sizeof(toast), "Easy Speed: Fast (kept character)");
                            } else {
                                preset_idx = sel - 1;
                                char_idx = keep_char;
                                apply_tv_easy_framing_and_quality();
                                snprintf(toast, sizeof(toast), "Scene: %s", g_presets[preset_idx].name);
                            }
                        }
                    } else if (sel == 0) {
                        char pname[64] = "";
                        if (edit_text_osk(L"Save Preset Name", pname, sizeof(pname), 0) && pname[0]) {
                            UserPreset np;
                            memset(&np, 0, sizeof(np));
                            snprintf(np.name, sizeof(np.name), "%s", pname);
                            static const char* SMP_N4[4] = {"Euler a", "Euler", "LCM", "DPM++ 2M Karras"};
                            http_srv_lock();
                            snprintf(np.model, sizeof(np.model), "%s", g_st.model);
                            snprintf(np.prompt, sizeof(np.prompt), "%s", g_job.prompt);
                            snprintf(np.negative, sizeof(np.negative), "%s", g_job.negative);
                            snprintf(np.sampler, sizeof(np.sampler), "%s", SMP_N4[(g_job.sampler >= 0 && g_job.sampler <= 3) ? g_job.sampler : 0]);
                            np.width = g_job.width; np.height = g_job.height;
                            np.steps = g_job.steps; np.cfg = g_job.cfg;
                            np.upscale = g_job.upscale; np.eco = g_job.eco_mode;
                            http_srv_unlock();
                            char sid[64] = {0};
                            if (presets_save(&np, sid, sizeof(sid)) == 0) {
                                refresh_tv_presets();
                                for (int k = 0; k < g_num_presets; k++) {
                                    if (strcmp(g_presets[k].id, sid) == 0) { preset_idx = k; break; }
                                }
                                snprintf(toast, sizeof(toast), "Saved preset: %s", pname);
                            } else {
                                snprintf(toast, sizeof(toast), "Failed to save preset");
                            }
                        }
                    } else if (sel - 1 >= 0 && sel - 1 < g_num_presets) {
                        preset_idx = sel - 1;
                        apply_tv_preset(preset_idx);
                        snprintf(toast, sizeof(toast), "Preset: %s", g_presets[preset_idx].name);
                    }
                    break;
                case ROW_CHAR: {
                    int prev_c = char_idx;
                    char_idx = sel - 1;
                    if (char_idx >= g_num_chars) char_idx = -1;
                    if (cur_uim == 0) {
                        apply_tv_easy_framing_and_quality();
                    } else {
                        apply_tv_char_selection(prev_c, char_idx);
                    }
                    if (char_idx >= 0) snprintf(toast, sizeof(toast), "Character: %s (%s)", g_chars[char_idx].name, g_chars[char_idx].series);
                    else snprintf(toast, sizeof(toast), "Character: Any / Custom");
                    break;
                }
                case ROW_STYLE:
                    style_idx = sel - 1;
                    if (style_idx >= g_num_styles) style_idx = -1;
                    if (style_idx >= 0) snprintf(toast, sizeof(toast), "Art Style: %s", g_styles[style_idx].name);
                    else snprintf(toast, sizeof(toast), "Art Style: Off");
                    break;
                case ROW_RES:
                    if (sel >= 0 && sel < RES_COUNT) {
                        res_idx = sel;
                        http_srv_lock();
                        g_job.width = RES_LIST[res_idx][0];
                        g_job.height = RES_LIST[res_idx][1];
                        http_srv_unlock();
                    }
                    break;
                case ROW_SAMPLER:
                    if (sel >= 0 && sel <= 3) {
                        http_srv_lock(); g_job.sampler = sel; http_srv_unlock();
                    }
                    break;
                case ROW_UPSCALE:
                    if (sel >= 0 && sel <= 3) {
                        http_srv_lock(); g_job.upscale = sel; http_srv_unlock();
                    }
                    break;
                case ROW_POWER:
                    if (sel >= 0 && sel <= 3) {
                        int eco_val = 3 - sel;
                        http_srv_lock(); g_job.eco_mode = eco_val; g_st.eco_mode = eco_val; http_srv_unlock();
                    }
                    break;
                case ROW_SEED:
                    if (sel == 0) {
                        http_srv_lock(); g_job.seed = 0; http_srv_unlock();
                        snprintf(toast, sizeof(toast), "Seed: Random each time");
                    } else if (sel == 1) {
                        uint64_t ls = sm_last_seed ? sm_last_seed : 42ull;
                        http_srv_lock(); g_job.seed = ls; http_srv_unlock();
                        snprintf(toast, sizeof(toast), "Seed locked to %llu", (unsigned long long)ls);
                    } else {
                        char sbuf[32] = {0};
                        if (g_job.seed) snprintf(sbuf, sizeof(sbuf), "%llu", (unsigned long long)g_job.seed);
                        if (edit_text_osk(L"Custom Seed (0 = Random)", sbuf, sizeof(sbuf), 1)) {
                            uint64_t nv = strtoull(sbuf, NULL, 10);
                            http_srv_lock(); g_job.seed = nv; http_srv_unlock();
                            if (nv) snprintf(toast, sizeof(toast), "Seed locked to %llu", (unsigned long long)nv);
                            else snprintf(toast, sizeof(toast), "Seed: Random each time");
                        }
                    }
                    break;
                case ROW_BATCH: {
                    int ns_b = 0;
                    http_srv_lock(); ns_b = g_job.nsfw_allow; http_srv_unlock();
                    int n_rnd = ns_b ? 9 : 7;
                    int base = 1 + n_rnd;
                    int tot_opts = tv_batch_opts_count(ns_b);
                    static const char* const RSHORT[9] = {"Pose", "Expression", "Scene", "Lighting", "Outfit", "Details", "Camera", "Ecchi 500", "Cog Test"};
                    if (sel == 0) {
                        http_srv_lock();
                        uint32_t def_m = RAND_POSE | RAND_EXPR | RAND_SCENE | RAND_LIGHT | RAND_DETAIL | (ns_b ? RAND_ECCHI : 0u);
                        g_job.batch_rand_mask = g_job.batch_rand_mask ? 0u : def_m;
                        g_st.cur_rand_mask = g_job.batch_rand_mask;
                        uint32_t nm = g_job.batch_rand_mask;
                        g_st.settings_seq++;
                        http_srv_unlock();
                        sm_dd_open = 1;
                        snprintf(toast, sizeof(toast), nm ? "Enabled Variety Mix!" : "Cleared all Randomizer pools");
                    } else if (sel >= 1 && sel <= n_rnd) {
                        uint32_t bit = 1u << (sel - 1);
                        http_srv_lock();
                        g_job.batch_rand_mask ^= bit;
                        g_st.cur_rand_mask = g_job.batch_rand_mask;
                        int on_now = (g_job.batch_rand_mask & bit) != 0;
                        g_st.settings_seq++;
                        http_srv_unlock();
                        sm_dd_open = 1;
                        snprintf(toast, sizeof(toast), "Randomizer %s: %s", RSHORT[sel - 1], on_now ? "ON" : "OFF");
                    } else if (sel == base) {
                        char fbuf[64] = {0};
                        http_srv_lock(); snprintf(fbuf, sizeof(fbuf), "%s", g_job.batch_folder); http_srv_unlock();
                        if (edit_text_osk(L"Batch Target Folder (empty = Unsorted)", fbuf, sizeof(fbuf), 0)) {
                            char clean_f[64] = {0};
                            if (gallery_clean_folder_name(fbuf, clean_f, sizeof(clean_f)) > 0)
                                gallery_create_folder(clean_f, clean_f, sizeof(clean_f));
                            http_srv_lock();
                            snprintf(g_job.batch_folder, sizeof(g_job.batch_folder), "%s", clean_f);
                            g_st.settings_seq++;
                            http_srv_unlock();
                            snprintf(toast, sizeof(toast), clean_f[0] ? "Batch folder: [%s]" : "Batch folder: Unsorted", clean_f);
                        }
                    } else if (sel == base + 1) {
                        http_srv_lock();
                        g_job.auto_quality = !g_job.auto_quality;
                        g_st.auto_quality = g_job.auto_quality;
                        int aq = g_job.auto_quality;
                        g_st.settings_seq++;
                        http_srv_unlock();
                        sm_dd_open = 1;
                        snprintf(toast, sizeof(toast), aq ? "Auto-Quality & Hand/Eye Shield ON" : "Auto-Quality Shield OFF");
                    } else if (sel == base + 2) {
                        http_srv_lock();
                        g_job.shutdown_after = !g_job.shutdown_after;
                        g_st.shutdown_after = g_job.shutdown_after;
                        int sd = g_job.shutdown_after;
                        http_srv_unlock();
                        sm_dd_open = 1;
                        snprintf(toast, sizeof(toast), sd ? "PS4 will shut down after batch finishes" : "Auto-shutdown OFF");
                    } else if (sel - tot_opts >= 0 && sel - tot_opts < BATCH_COUNT) {
                        http_srv_lock(); g_job.batch_count = BATCH_LIST[sel - tot_opts]; http_srv_unlock();
                    }
                    break;
                }
                }
                ui_dirty = 1;
            }
        } else if (sm_adjust) {
            // ---- SLIDER ADJUST MODE (Left/Right + L1/R1 coarse, X Confirm, O Revert) ----
            if (pressed & ORBIS_PAD_BUTTON_CROSS) {
                sm_adjust = 0; ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                http_srv_lock();
                if (sm_focus == ROW_STEPS) g_job.steps = sm_saved_steps;
                else if (sm_focus == ROW_CFG) g_job.cfg = sm_saved_cfg;
                http_srv_unlock();
                sm_adjust = 0; ui_dirty = 1;
            } else if ((nav & (ORBIS_PAD_BUTTON_LEFT | ORBIS_PAD_BUTTON_RIGHT)) ||
                       (pressed & (ORBIS_PAD_BUTTON_L1 | ORBIS_PAD_BUTTON_R1))) {
                int dir = ((nav & ORBIS_PAD_BUTTON_RIGHT) || (pressed & ORBIS_PAD_BUTTON_R1)) ? 1 : -1;
                int big = (pressed & (ORBIS_PAD_BUTTON_L1 | ORBIS_PAD_BUTTON_R1)) ? 1 : 0;
                http_srv_lock();
                if (sm_focus == ROW_STEPS) {
                    int ns = (int)g_job.steps + dir * (big ? 5 : 1);
                    if (ns < SM_STEPS_MIN) ns = SM_STEPS_MIN;
                    if (ns > SM_STEPS_MAX) ns = SM_STEPS_MAX;
                    g_job.steps = (uint32_t)ns;
                } else if (sm_focus == ROW_CFG) {
                    if (big) {
                        float nc = g_job.cfg + (float)dir * 1.0f;
                        if (nc < SM_CFG_MIN) nc = SM_CFG_MIN;
                        if (nc > SM_CFG_MAX) nc = SM_CFG_MAX;
                        g_job.cfg = floorf(nc * 10.0f + 0.5f) * 0.1f;
                    } else {
                        g_job.cfg = sm_step_cfg(g_job.cfg, dir);
                    }
                }
                http_srv_unlock();
                ui_dirty = 1;
            }
        } else if (cur_uim == 0) {
            // ---- EASY MODE STUDIO (6 Beginner-Proof Rows + R1 Toggle to Advanced) ----
            if (sm_focus >= EROW_COUNT) sm_focus = 0;
            if (pressed & ORBIS_PAD_BUTTON_R1) {
                http_srv_lock(); g_job.ui_mode = 1; g_st.ui_mode = 1; g_st.settings_seq++; http_srv_unlock();
                save_ui_mode_cfg(1);
                sm_focus = 0;
                snprintf(toast, sizeof(toast), "Switched to Advanced Studio (press R1 for Easy Mode)");
                ui_dirty = 1;
            } else if (nav & ORBIS_PAD_BUTTON_DOWN) {
                sm_focus = (sm_focus + 1) % EROW_COUNT;
                toast[0] = 0; ui_dirty = 1;
            } else if (nav & ORBIS_PAD_BUTTON_UP) {
                sm_focus = (sm_focus + EROW_COUNT - 1) % EROW_COUNT;
                toast[0] = 0; ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_L2) {
                if (g_num_presets > 0) {
                    int keep_char = char_idx;
                    for (int tries = 0; tries < 8; tries++) {
                        int cand = (int)((sceKernelReadTsc() + (uint64_t)tries * 7919u) % (uint64_t)g_num_presets);
                        if (!g_presets[cand].is_speed) { preset_idx = cand; break; }
                    }
                    char_idx = keep_char;
                    apply_tv_easy_framing_and_quality();
                    snprintf(toast, sizeof(toast), "Surprise Scene: %s", g_presets[preset_idx].name);
                    ui_dirty = 1;
                }
            } else if (pressed & ORBIS_PAD_BUTTON_R2) {
                if (edit_text_osk(L"Describe Scene or Action", g_easy_scene, sizeof(g_easy_scene), 0)) {
                    preset_idx = -1;
                    apply_tv_easy_framing_and_quality();
                    snprintf(toast, sizeof(toast), "Custom scene updated!");
                }
                ui_dirty = 1;
            } else if (nav & (ORBIS_PAD_BUTTON_LEFT | ORBIS_PAD_BUTTON_RIGHT)) {
                int dir = (nav & ORBIS_PAD_BUTTON_RIGHT) ? 1 : -1;
                toast[0] = 0;
                if (sm_focus == EROW_CHAR && g_num_chars > 0) {
                    int n = g_num_chars + 1;
                    char_idx = ((char_idx + 1 + n + dir) % n) - 1;
                    apply_tv_easy_framing_and_quality();
                    if (char_idx >= 0) snprintf(toast, sizeof(toast), "Character: %s", g_chars[char_idx].name);
                    else snprintf(toast, sizeof(toast), "Character: Any / Custom");
                } else if (sm_focus == EROW_SCENE && g_num_presets > 0) {
                    int keep_char = char_idx;
                    int base = (preset_idx < 0) ? 0 : preset_idx;
                    for (int stp = 0; stp < g_num_presets; stp++) {
                        base = (base + g_num_presets + dir) % g_num_presets;
                        if (!g_presets[base].is_speed) break;
                    }
                    preset_idx = base;
                    char_idx = keep_char;
                    apply_tv_easy_framing_and_quality();
                    snprintf(toast, sizeof(toast), "Scene: %s", g_presets[preset_idx].name);
                } else if (sm_focus == EROW_FRAMING) {
                    g_easy_framing = (g_easy_framing + 4 + dir) % 4;
                    apply_tv_easy_framing_and_quality();
                } else if (sm_focus == EROW_QUALITY) {
                    g_easy_quality = (g_easy_quality + 3 + dir) % 3;
                    apply_tv_easy_framing_and_quality();
                }
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_CROSS) {
                toast[0] = 0;
                if (sm_focus == EROW_CHAR) {
                    sm_dd_row = ROW_CHAR; sm_dd_n = g_num_chars + 1;
                    sm_dd_cur = char_idx + 1; sm_dd_sel = char_idx + 1;
                    sm_dd_top = 0; sm_dd_clamp(); sm_dd_open = 1;
                } else if (sm_focus == EROW_SCENE) {
                    sm_dd_row = ROW_PRESET; sm_dd_n = g_num_presets + 1;
                    sm_dd_cur = (preset_idx >= 0) ? (preset_idx + 1) : 0;
                    sm_dd_sel = sm_dd_cur; sm_dd_top = 0; sm_dd_clamp(); sm_dd_open = 1;
                } else if (sm_focus == EROW_FRAMING) {
                    g_easy_framing = (g_easy_framing + 1) % 4;
                    apply_tv_easy_framing_and_quality();
                } else if (sm_focus == EROW_QUALITY) {
                    g_easy_quality = (g_easy_quality + 1) % 3;
                    apply_tv_easy_framing_and_quality();
                } else if (sm_focus == EROW_MODEL) {
                    scan_models();
                    http_srv_lock();
                    sm_dd_models_n = g_st.num_models;
                    sm_dd_cur = -1;
                    for (int i = 0; i < sm_dd_models_n; i++) {
                        snprintf(sm_dd_models[i], sizeof(sm_dd_models[i]), "%s", g_st.models[i]);
                        if (strcmp(sm_dd_models[i], g_st.model) == 0) sm_dd_cur = i;
                    }
                    http_srv_unlock();
                    sm_dd_row = ROW_MODEL; sm_dd_n = sm_dd_models_n + 1;
                    sm_dd_sel = sm_dd_cur >= 0 ? sm_dd_cur : 0;
                    sm_dd_top = 0; sm_dd_clamp(); sm_dd_open = 1;
                } else if (sm_focus == EROW_CREDITS) {
                    g_in_credits = 1; toast[0] = 0;
                } else if (sm_focus == EROW_GEN) {
                    apply_tv_easy_framing_and_quality();
                    int open_mgr = 0, neo_m = 1;
                    http_srv_lock();
                    neo_m = g_st.neo;
                    if (g_st.ready && !g_st.busy) {
                        g_job.cancel_req = 0; g_tv_gen = 1; g_job.pending = 1;
                    } else if (g_st.busy) {
                        snprintf(toast, sizeof(toast), "GPU is busy — press O to cancel current job");
                    } else {
                        open_mgr = 1;
                    }
                    http_srv_unlock();
                    if (open_mgr) {
                        open_starter_models_mgr(neo_m);
                        snprintf(toast, sizeof(toast), "Pick a ★ STARTER Checkpoint and press X to download!");
                    }
                }
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_L3) {
                g_in_credits = 1; toast[0] = 0; ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) {
                apply_tv_easy_framing_and_quality();
                int open_mgr = 0, neo_m = 1;
                http_srv_lock();
                neo_m = g_st.neo;
                if (g_st.ready && !g_st.busy) {
                    g_job.cancel_req = 0; g_tv_gen = 1; g_job.pending = 1;
                } else if (g_st.busy) {
                    snprintf(toast, sizeof(toast), "GPU is busy — press O to cancel");
                } else {
                    open_mgr = 1;
                }
                http_srv_unlock();
                if (open_mgr) {
                    open_starter_models_mgr(neo_m);
                    snprintf(toast, sizeof(toast), "Pick a ★ STARTER Checkpoint and press X to download!");
                }
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_SQUARE) {
                g_in_gallery = 1; g_gal_full = 0; g_gal_idx = 0; toast[0] = 0; gal_refresh(); ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_OPTIONS) {
                g_in_models = 1; g_mdl_sel = 0; g_mdl_del_arm = 0; toast[0] = 0; models_mgr_refresh(); ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                if (cur_busy) {
                    http_srv_lock(); g_job.cancel_req = 1; http_srv_unlock();
                    snprintf(toast, sizeof(toast), "Cancelling current job...");
                } else {
                    g_in_credits = 1; toast[0] = 0;
                }
                ui_dirty = 1;
            }
        } else {
            // ---- ADVANCED STUDIO SETTINGS LIST ----
            if (pressed & ORBIS_PAD_BUTTON_R1) {
                http_srv_lock(); g_job.ui_mode = 0; g_st.ui_mode = 0; g_st.settings_seq++; http_srv_unlock();
                save_ui_mode_cfg(0);
                sm_focus = 0;
                apply_tv_easy_framing_and_quality();
                snprintf(toast, sizeof(toast), "Switched to Easy Mode (press R1 for Advanced Studio)");
                ui_dirty = 1;
            } else if (nav & ORBIS_PAD_BUTTON_DOWN) {
                sm_focus = (sm_focus + 1) % ROW_COUNT;
                toast[0] = 0; ui_dirty = 1;
            } else if (nav & ORBIS_PAD_BUTTON_UP) {
                sm_focus = (sm_focus + ROW_COUNT - 1) % ROW_COUNT;
                toast[0] = 0; ui_dirty = 1;
            } else if (nav & (ORBIS_PAD_BUTTON_LEFT | ORBIS_PAD_BUTTON_RIGHT)) {
                int dir = (nav & ORBIS_PAD_BUTTON_RIGHT) ? 1 : -1;
                toast[0] = 0;
                switch (sm_focus) {
                case ROW_LORA:
                    g_num_loras = lora_list_installed(g_loras, LORA_MAX_INSTALLED);
                    http_srv_lock();
                    if (g_job.lora[0]) {
                        float ns = floorf((g_job.lora_scale + (float)dir * 0.1f) * 10.0f + 0.5f) * 0.1f;
                        if (ns <= 0.05f && dir < 0) {
                            g_job.lora[0] = 0;
                            snprintf(toast, sizeof(toast), "LoRA: Off");
                        } else {
                            if (ns > 2.0f) ns = 2.0f;
                            g_job.lora_scale = ns;
                            snprintf(toast, sizeof(toast), "LoRA %s strength: %.2f", g_job.lora, (double)ns);
                        }
                    } else if (g_num_loras > 0) {
                        int pick = dir > 0 ? 0 : (g_num_loras - 1);
                        snprintf(g_job.lora, sizeof(g_job.lora), "%s", g_loras[pick].name);
                        if (g_job.lora_scale < 0.1f) g_job.lora_scale = 0.80f;
                        snprintf(toast, sizeof(toast), "LoRA: %s (%.2f)", g_job.lora, (double)g_job.lora_scale);
                    }
                    http_srv_unlock();
                    break;
                case ROW_PRESET:
                    if (g_num_presets > 0) {
                        preset_idx = (preset_idx + g_num_presets + dir) % g_num_presets;
                        apply_tv_preset(preset_idx);
                        snprintf(toast, sizeof(toast), "Preset: %s", g_presets[preset_idx].name);
                    }
                    break;
                case ROW_CHAR:
                    if (g_num_chars > 0) {
                        int prev_c = char_idx;
                        int n = g_num_chars + 1;
                        int cur = (char_idx + 1 + n + dir) % n;
                        char_idx = cur - 1;
                        apply_tv_char_selection(prev_c, char_idx);
                    }
                    break;
                case ROW_STYLE:
                    if (g_num_styles > 0) {
                        int n = g_num_styles + 1;
                        int cur = (style_idx + 1 + n + dir) % n;
                        style_idx = cur - 1;
                    }
                    break;
                case ROW_RES:
                    res_idx = (res_idx + RES_COUNT + dir) % RES_COUNT;
                    http_srv_lock();
                    g_job.width = RES_LIST[res_idx][0];
                    g_job.height = RES_LIST[res_idx][1];
                    http_srv_unlock();
                    break;
                case ROW_STEPS:
                    http_srv_lock();
                    {
                        int ns = (int)g_job.steps + dir;
                        if (ns < SM_STEPS_MIN) ns = SM_STEPS_MIN;
                        if (ns > SM_STEPS_MAX) ns = SM_STEPS_MAX;
                        g_job.steps = (uint32_t)ns;
                    }
                    http_srv_unlock();
                    break;
                case ROW_CFG:
                    http_srv_lock();
                    g_job.cfg = sm_step_cfg(g_job.cfg, dir);
                    http_srv_unlock();
                    break;
                case ROW_SAMPLER:
                    http_srv_lock();
                    g_job.sampler = (g_job.sampler + 4 + dir) % 4;
                    http_srv_unlock();
                    break;
                case ROW_UPSCALE:
                    http_srv_lock();
                    g_job.upscale = (g_job.upscale + 4 + dir) % 4;
                    http_srv_unlock();
                    break;
                case ROW_POWER:
                    http_srv_lock();
                    {
                        // Left = cooler (higher eco_mode), Right = faster (lower eco_mode)
                        int ne = g_job.eco_mode - dir;
                        if (ne < 0) ne = 3;
                        if (ne > 3) ne = 0;
                        g_job.eco_mode = ne;
                        g_st.eco_mode = ne;
                    }
                    http_srv_unlock();
                    break;
                case ROW_SEED:
                    http_srv_lock();
                    if (g_job.seed == 0) g_job.seed = sm_last_seed ? sm_last_seed : 42ull;
                    else g_job.seed = 0;
                    http_srv_unlock();
                    break;
                case ROW_BATCH:
                    http_srv_lock();
                    {
                        uint32_t cur_b = g_job.batch_count ? g_job.batch_count : 1u;
                        int bi = 0;
                        for (int k = 0; k < BATCH_COUNT; k++) {
                            if (BATCH_LIST[k] <= cur_b) bi = k;
                        }
                        bi = (bi + BATCH_COUNT + dir) % BATCH_COUNT;
                        g_job.batch_count = BATCH_LIST[bi];
                    }
                    http_srv_unlock();
                    break;
                }
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_CROSS) {
                toast[0] = 0;
                switch (sm_focus) {
                case ROW_MODEL:
                    scan_models();
                    http_srv_lock();
                    sm_dd_models_n = g_st.num_models;
                    sm_dd_cur = -1;
                    for (int i = 0; i < sm_dd_models_n; i++) {
                        snprintf(sm_dd_models[i], sizeof(sm_dd_models[i]), "%s", g_st.models[i]);
                        if (strcmp(sm_dd_models[i], g_st.model) == 0) sm_dd_cur = i;
                    }
                    http_srv_unlock();
                    sm_dd_row = ROW_MODEL;
                    sm_dd_n = sm_dd_models_n + 1;
                    sm_dd_sel = sm_dd_cur >= 0 ? sm_dd_cur : 0;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_LORA:
                    g_num_loras = lora_list_installed(g_loras, LORA_MAX_INSTALLED);
                    sm_dd_row = ROW_LORA;
                    sm_dd_n = g_num_loras + 1;
                    sm_dd_cur = 0;
                    http_srv_lock();
                    for (int i = 0; i < g_num_loras; i++) {
                        if (strcmp(g_loras[i].name, g_job.lora) == 0) { sm_dd_cur = i + 1; break; }
                    }
                    http_srv_unlock();
                    sm_dd_sel = sm_dd_cur;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_PRESET:
                    sm_dd_row = ROW_PRESET;
                    sm_dd_n = g_num_presets + 1;
                    sm_dd_cur = (preset_idx >= 0) ? (preset_idx + 1) : 0;
                    sm_dd_sel = sm_dd_cur;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_CHAR:
                    sm_dd_row = ROW_CHAR;
                    sm_dd_n = g_num_chars + 1;
                    sm_dd_cur = char_idx + 1;
                    sm_dd_sel = char_idx + 1;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_STYLE:
                    sm_dd_row = ROW_STYLE;
                    sm_dd_n = g_num_styles + 1;
                    sm_dd_cur = style_idx + 1;
                    sm_dd_sel = style_idx + 1;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_PROMPT:
                    if (edit_text_osk(L"Positive Prompt", g_job.prompt, sizeof(g_job.prompt), 0))
                        snprintf(toast, sizeof(toast), "Prompt updated");
                    break;
                case ROW_NEG:
                    if (edit_text_osk(L"Negative Prompt", g_job.negative, sizeof(g_job.negative), 0))
                        snprintf(toast, sizeof(toast), "Negative prompt updated");
                    break;
                case ROW_RES:
                    sm_dd_row = ROW_RES;
                    sm_dd_n = RES_COUNT;
                    sm_dd_cur = res_idx;
                    sm_dd_sel = res_idx;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_STEPS:
                    http_srv_lock(); sm_saved_steps = g_job.steps; http_srv_unlock();
                    sm_adjust = 1;
                    break;
                case ROW_CFG:
                    http_srv_lock(); sm_saved_cfg = g_job.cfg; http_srv_unlock();
                    sm_adjust = 1;
                    break;
                case ROW_SAMPLER:
                    sm_dd_row = ROW_SAMPLER;
                    sm_dd_n = 4;
                    http_srv_lock(); sm_dd_cur = g_job.sampler; http_srv_unlock();
                    sm_dd_sel = (sm_dd_cur >= 0 && sm_dd_cur <= 3) ? sm_dd_cur : 0;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_UPSCALE:
                    sm_dd_row = ROW_UPSCALE;
                    sm_dd_n = 4;
                    http_srv_lock(); sm_dd_cur = g_job.upscale; http_srv_unlock();
                    sm_dd_sel = (sm_dd_cur >= 0 && sm_dd_cur <= 3) ? sm_dd_cur : 0;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_POWER:
                    sm_dd_row = ROW_POWER;
                    sm_dd_n = 4;
                    http_srv_lock(); sm_dd_cur = 3 - g_job.eco_mode; http_srv_unlock();
                    if (sm_dd_cur < 0 || sm_dd_cur > 3) sm_dd_cur = 1;
                    sm_dd_sel = sm_dd_cur;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_SEED:
                    sm_dd_row = ROW_SEED;
                    sm_dd_n = 3;
                    http_srv_lock(); sm_dd_cur = g_job.seed ? 1 : 0; http_srv_unlock();
                    sm_dd_sel = sm_dd_cur;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_BATCH:
                    sm_dd_row = ROW_BATCH;
                    {
                        uint32_t cur_b = 1;
                        int ns_o = 0;
                        http_srv_lock();
                        cur_b = g_job.batch_count ? g_job.batch_count : 1u;
                        ns_o = g_job.nsfw_allow;
                        http_srv_unlock();
                        int tot_o = tv_batch_opts_count(ns_o);
                        sm_dd_n = BATCH_COUNT + tot_o;
                        sm_dd_cur = tot_o;
                        for (int i = 0; i < BATCH_COUNT; i++) {
                            if (BATCH_LIST[i] == cur_b) { sm_dd_cur = i + tot_o; break; }
                        }
                    }
                    sm_dd_sel = sm_dd_cur;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_GEN: {
                    int open_mgr = 0, neo_m = 1;
                    http_srv_lock();
                    neo_m = g_st.neo;
                    if (g_st.ready && !g_st.busy) {
                        g_job.cancel_req = 0;
                        g_tv_gen = 1;
                        g_job.pending = 1;
                    } else if (g_st.busy) {
                        snprintf(toast, sizeof(toast), "GPU is busy — press O to cancel current job");
                    } else {
                        open_mgr = 1;
                    }
                    http_srv_unlock();
                    if (open_mgr) {
                        open_starter_models_mgr(neo_m);
                        snprintf(toast, sizeof(toast), "Pick a ★ STARTER Checkpoint and press X to download!");
                    }
                    break;
                }
                case ROW_CREDITS:
                    g_in_credits = 1; toast[0] = 0;
                    break;
                }
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_L3) {
                g_in_credits = 1; toast[0] = 0; ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) {
                http_srv_lock();
                if (g_st.ready && !g_st.busy) {
                    g_job.cancel_req = 0;
                    g_tv_gen = 1;
                    g_job.pending = 1;
                } else if (g_st.busy) {
                    snprintf(toast, sizeof(toast), "GPU is busy — press O to cancel");
                } else {
                    snprintf(toast, sizeof(toast), "No complete model loaded - press OPTIONS");
                }
                http_srv_unlock();
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_SQUARE) {
                g_in_gallery = 1; g_gal_full = 0; g_gal_idx = 0; toast[0] = 0;
                gal_refresh();
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_OPTIONS) {
                g_in_models = 1; g_mdl_sel = 0; g_mdl_del_arm = 0; toast[0] = 0;
                models_mgr_refresh();
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                if (cur_busy) {
                    http_srv_lock(); g_job.cancel_req = 1; http_srv_unlock();
                    snprintf(toast, sizeof(toast), "Cancelling current job...");
                } else {
                    g_in_credits = 1; toast[0] = 0;
                }
                ui_dirty = 1;
            }
        }

        // Non-blocking 60 FPS VSync draw when busy/downloading/dirty; calm 60 Hz pad poll when idle!
        {
            int is_busy = 0;
            http_srv_lock();
            is_busy = g_st.busy;
            http_srv_unlock();
            ModelDlStatus dl_chk;
            model_dl_get_status(&dl_chk);
            if (is_busy || dl_chk.active || ui_dirty || ++idle_ticks >= 30) {
                draw_ui();
                idle_ticks = 0;
            } else {
                sceKernelUsleep(16000);
            }
        }
    }
    return 0;
}
