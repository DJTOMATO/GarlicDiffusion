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

#include "gallery.h"
#include "garlic_ui.h"
#include "http_srv.h"
#include "model_dl.h"
#include "png_write.h"
#include "sdxl.h"
#include "studio_menu.h"

#define SCREEN_W 1920
#define SCREEN_H 1080
#define ASSETS_DIR "/app0/assets"
#define MODELS_ROOT "/data/garlic-diffusion/models"
#define OUT_ROOT "/data/garlic-diffusion/out"
#define PRESETS_ROOT "/data/garlic-diffusion/presets"
#define STYLES_ROOT "/data/garlic-diffusion/styles"
#define CHARS_ROOT "/data/garlic-diffusion/characters"
#define REPO_CFG "/data/garlic-diffusion/repo.cfg"
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
static uint8_t* preview_rgb = NULL;
static uint32_t preview_w = 0, preview_h = 0;
static char toast[128] = "";

// Dynamic User Presets, Art Styles, Character Presets + TV Gallery + TV Model Manager state
static UserPreset g_presets[PRESETS_MAX_ITEMS];
static int g_num_presets = 0;
static StyleItem g_styles[STYLES_MAX_ITEMS];
static int g_num_styles = 0, style_idx = -1;
static CharacterItem g_chars[CHARS_MAX_ITEMS];
static int g_num_chars = 0, char_idx = -1;
static int g_tv_gen = 0;
static GalleryItem g_gal[GALLERY_MAX_ITEMS];
static int g_gal_count = 0, g_gal_idx = 0;
static int g_in_gallery = 0, g_gal_full = 0, g_del_arm = 0;
static uint8_t* g_gal_rgb = NULL;
static uint32_t g_gal_w = 0, g_gal_h = 0;

static int g_in_models = 0, g_in_credits = 0, g_mdl_sel = 0, g_mdl_del_arm = 0, g_last_dl_fin = 0;
static InstalledModel g_inst[MODEL_DL_MAX_INSTALLED];
static int g_inst_cnt = 0;
static CatalogModel g_cat[MODEL_DL_MAX_CATALOG];
static int g_cat_cnt = 0;

static void format_a1111_params(char* out, size_t max, const char* prompt, const char* neg,
                                uint32_t steps, const char* sampler, float cfg, uint64_t seed,
                                uint32_t w, uint32_t h, const char* model, int upscaled) {
    const char* up_str = upscaled == 3 ? ", Hires upscaler: RealESRGAN Anime v3 (4K UHD)"
                       : (upscaled > 0 ? ", Hires upscaler: RealESRGAN Anime v3" : "");
    snprintf(out, max,
             "%s\nNegative prompt: %s\nSteps: %u, Sampler: %s, CFG scale: %.1f, Seed: %llu, Size: %ux%u, Model: %s, Clip skip: 2%s, Generator: GarlicDiffusion PS4 (GPLv3)",
             prompt ? prompt : "", neg ? neg : "", steps, (sampler && sampler[0]) ? sampler : "Euler a",
             (double)cfg, (unsigned long long)seed, w, h, (model && model[0]) ? model : "GarlicDiffusion",
             up_str);
}

static void models_mgr_refresh(void) {
    g_inst_cnt = model_dl_list_installed(g_inst, MODEL_DL_MAX_INSTALLED);
    g_cat_cnt = model_dl_get_catalog(g_cat, MODEL_DL_MAX_CATALOG);
    int total = g_inst_cnt + g_cat_cnt;
    if (g_mdl_sel >= total) g_mdl_sel = total > 0 ? total - 1 : 0;
    g_mdl_del_arm = 0;
}

static const uint32_t RES_LIST[][2] = {
    {512, 512}, {512, 768}, {768, 512}, {768, 768}, {768, 1024},
    {896, 512}, {1024, 576}, {1152, 640}, {1280, 704},
    {832, 1216}, {1216, 832}, {1024, 1024}
};
static const int RES_COUNT = (int)(sizeof(RES_LIST) / sizeof(RES_LIST[0]));
static int preset_idx = 0, res_idx = 4, model_idx = 0;

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

static const uint32_t BATCH_LIST[] = {1, 2, 3, 4, 5, 8, 10, 15, 20};
static const int BATCH_COUNT = (int)(sizeof(BATCH_LIST) / sizeof(BATCH_LIST[0]));

static void sm_dd_item_label(void* ctx, int row, int i, char* out, size_t cap) {
    (void)ctx;
    switch (row) {
    case ROW_MODEL:
        if (i < sm_dd_models_n) snprintf(out, cap, "%s", sm_dd_models[i]);
        else snprintf(out, cap, "+ Download / Manage Models (Hugging Face)...");
        break;
    case ROW_PRESET:
        if (i == 0) snprintf(out, cap, "+ Save Current Prompt as Preset...");
        else if (i - 1 >= 0 && i - 1 < g_num_presets) snprintf(out, cap, "%s", g_presets[i - 1].name);
        break;
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
    case ROW_POWER:
        // Ordered coolest -> loudest: 0=Silent(eco 3), 1=Quiet(eco 2), 2=Balanced(eco 1), 3=Turbo(eco 0)
        sm_power_label(3 - i, eng.neo, out, cap);
        break;
    case ROW_SEED:
        if (i == 0) snprintf(out, cap, "Random each time");
        else if (i == 1) {
            if (sm_last_seed) snprintf(out, cap, "Lock last seed (%llu)", (unsigned long long)sm_last_seed);
            else snprintf(out, cap, "Lock seed 42");
        } else {
            snprintf(out, cap, "Enter custom seed number...");
        }
        break;
    case ROW_BATCH:
        if (i >= 0 && i < BATCH_COUNT) {
            if (BATCH_LIST[i] == 1) snprintf(out, cap, "1 image  ·  single");
            else snprintf(out, cap, "%u images  ·  random seed each", BATCH_LIST[i]);
        }
        break;
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
    g_gal_count = gallery_list(g_gal, GALLERY_MAX_ITEMS);
    if (g_gal_idx >= g_gal_count) g_gal_idx = g_gal_count > 0 ? g_gal_count - 1 : 0;
    if (g_in_gallery) gal_load_current();
}

static void draw_ui(void) {
    http_srv_lock();
    char cur_m[64], stg[128], err[256], pr[1024];
    float prog = g_st.progress;
    int busy = g_st.busy, eco = g_job.eco_mode, ups = g_job.upscale;
    uint32_t vram_u = (uint32_t)(eng.next >> 20), vram_t = (uint32_t)(eng.mem_size >> 20);
    double lsec = g_st.last_sec, tf = g_st.eff_tflops;
    uint32_t w = g_job.width, h = g_job.height, steps = g_job.steps;
    float cfg = g_job.cfg;
    snprintf(cur_m, sizeof(cur_m), "%s", g_st.model);
    snprintf(stg, sizeof(stg), "%s", g_st.stage);
    snprintf(err, sizeof(err), "%s", g_st.error);
    snprintf(pr, sizeof(pr), "%s", g_job.prompt);
    http_srv_unlock();

    if (eco < 0 || eco > 3) eco = 2;
    if (ups < 0 || ups > 3) ups = 0;

    // ---- 1. CREDITS, SUPPORT & QR CODES SCREEN (Garlic LLM style) ----
    if (g_in_credits) {
        g_draw_bg(back);
        g_draw_header(back, "Credits, Licenses & Support", cur_m, busy, vram_u, vram_t, http_srv_address());

        // Top lineage & license banner card
        g_rrect(back, 56, 112, 1808, 174, 18, C_SURFACE, 235);
        g_rrect_line(back, 56, 112, 1808, 174, 18, 1, C_LINE_HI, 255);
        g_rrect(back, 72, 128, 4, 142, 2, C_ACCENT_HI, 220);
        g_draw_str(back, &g_fbold, 94, 126, "GarlicDiffusion v1.10  ·  Open-Source GNU GPLv3  ·  Inspired by Garlic LLM", 0xFFFFFF);
        g_draw_str(back, &g_fsmall, 94, 158,
                   "Built upon Leonardo Alegria's Garlic LLM (garlicllm.dev) bare-metal PS4 GCN PM4 compute & OpenOrbis architecture.",
                   C_TEXT_DIM);
        g_draw_str(back, &g_fsmall, 94, 186,
                   "Models & components keep their own licenses: SDXL / SD 1.5 (OpenRAIL), WAI-Illustrious (Fair AI 1.0-SD), DMD2 (MIT),",
                   C_TEXT_DIM);
        g_draw_str(back, &g_fsmall, 94, 212,
                   "Real-ESRGAN Anime v3 (BSD-3-Clause), OpenOrbis Toolchain (GPLv3), Selawik Font (SIL OFL 1.1).",
                   C_TEXT_DIM);
        g_draw_str(back, &g_fsmall, 94, 244,
                   "DISCLAIMER: Provided 'AS IS' without warranty. Use Quiet/Silent mode on aging PS4 hardware and ensure good ventilation.",
                   0xF2A65A);

        // 4 Scannable QR Code Cards across the center
        struct {
            const char* title;
            const char* sub1;
            const char* sub2;
            const uint8_t* mat;
            int sz;
            uint32_t accent;
        } qcards[4] = {
            { "Support on Ko-fi", "ko-fi.com/djtomato", "Fund PS4 AI kernels & models", QR_KOFI, QR_KOFI_SIZE, C_TRIANGLE },
            { "Support Discord", "discord.gg/Gbyq3kpGjc", "Request new models & bot setup", QR_DISCORD, QR_DISCORD_SIZE, C_CROSS },
            { "GitHub Repository", "DJTOMATO/GarlicDiffusion", "Source code, PKG & issues", QR_GITHUB, QR_GITHUB_SIZE, C_ACCENT_HI },
            { "Hugging Face Models", "DJTomato/ps4-diffusion-models", "10 pre-converted PSD1 models", QR_HF, QR_HF_SIZE, C_SQUARE },
        };
        for (int i = 0; i < 4; i++) {
            int cx = 56 + i * 458, cy = 306, cw = 434, ch = 664;
            g_rrect(back, cx, cy, cw, ch, 18, C_SURFACE, 235);
            g_rrect_line(back, cx, cy, cw, ch, 18, 1, C_LINE_HI, 255);
            g_rrect(back, cx + 20, cy + 20, cw - 40, 4, 2, qcards[i].accent, 220);
            g_draw_str(back, &g_fbold, cx + 28, cy + 38, qcards[i].title, 0xFFFFFF);
            g_draw_str(back, &g_fsmall, cx + 28, cy + 72, qcards[i].sub1, qcards[i].accent);
            g_draw_str(back, &g_fsmall, cx + 28, cy + 98, qcards[i].sub2, C_TEXT_MUTED);

            int scale = (qcards[i].sz <= 25) ? 11 : 10;
            int qpix = (qcards[i].sz + 4) * scale;
            int qx = cx + (cw - qpix) / 2;
            int qy = cy + 148 + (460 - qpix) / 2;
            g_draw_qr(back, qcards[i].mat, qcards[i].sz, qx, qy, scale, 0x0B1C3A, 0xFFFFFF);
            g_draw_str(back, &g_fsmall, cx + 28, cy + 618, "Scan with phone camera", C_TEXT_DIM);
        }

        const GHint fhints[] = {
            { GBTN_CIRCLE, "Back to Studio" },
            { GBTN_SQUARE, "Gallery" },
            { GBTN_UPDOWN, "Model Manager" },
        };
        g_draw_footer(back, fhints, 3, "Not affiliated with Sony Interactive Entertainment · GPLv3");
        present();
        return;
    }

    // ---- 2. MODEL MANAGER & HUGGING FACE DOWNLOADER (Garlic LLM style) ----
    if (g_in_models) {
        g_draw_bg(back);
        char repo[128], rsub[180];
        model_dl_get_repo(repo, sizeof(repo));
        snprintf(rsub, sizeof(rsub), "HF Repo: %s", repo[0] ? repo : "DJTomato/ps4-diffusion-models");
        g_draw_header(back, rsub, cur_m, busy, vram_u, vram_t, http_srv_address());

        ModelDlStatus dl;
        model_dl_get_status(&dl);
        int y0 = 110;
        if (dl.active || dl.error[0] || dl.stage[0]) {
            g_rrect(back, 56, y0, 1808, 74, 16, C_SURFACE, 240);
            g_rrect_line(back, 56, y0, 1808, 74, 16, 1, C_LINE_HI, 255);
            if (dl.active) {
                float pct = dl.bytes_total ? (100.0f * (float)dl.bytes_done / (float)dl.bytes_total) : 0.0f;
                char dmsg[220];
                snprintf(dmsg, sizeof(dmsg), "Downloading %s [%d/%d] %s  ·  %llu / %llu MiB (%.0f%%)  ·  %.1f MB/s",
                         dl.model_id, dl.file_idx, dl.file_total, dl.cur_file,
                         (unsigned long long)(dl.bytes_done >> 20), (unsigned long long)(dl.bytes_total >> 20),
                         (double)pct, dl.speed_mbs);
                g_draw_str(back, &g_fsmall, 80, y0 + 14, dmsg, C_ACCENT_HI);
                g_rrect(back, 80, y0 + 46, 1760, 12, 6, 0x1B3156, 255);
                int bw = (int)(1760.0f * (pct / 100.0f));
                if (bw > 1760) bw = 1760;
                if (bw > 0) g_rrect(back, 80, y0 + 46, bw < 12 ? 12 : bw, 12, 6, C_GREEN, 255);
            } else {
                g_draw_str(back, &g_fsmall, 80, y0 + 26, dl.error[0] ? dl.error : dl.stage, dl.error[0] ? C_RED : C_GREEN);
            }
            y0 += 88;
        }

        g_draw_str(back, &g_fbold, 56, y0, "Installed Models on PS4 (/data/garlic-diffusion/models)", 0xFFFFFF);
        y0 += 34;
        int idx = 0;
        for (int i = 0; i < g_inst_cnt && y0 < 860; i++, idx++) {
            int sel = (idx == g_mdl_sel);
            int act = (strcmp(g_inst[i].name, cur_m) == 0);
            g_rrect(back, 56, y0, 1808, 44, 12, sel ? C_SURFACE_HI : C_SURFACE, sel ? 255 : 210);
            g_rrect_line(back, 56, y0, 1808, 44, 12, 1, sel ? C_ACCENT_HI : C_LINE, 255);
            if (act || sel) g_circle(back, 80.0f, (float)(y0 + 22), 5.0f, 0.0f, act ? C_GREEN : C_ACCENT_HI, 255);
            char row[240];
            snprintf(row, sizeof(row), "%s   [%s · %u MiB]   %s%s",
                     g_inst[i].name, g_inst[i].arch, g_inst[i].size_mb,
                     act ? "ACTIVE · " : "", g_inst[i].detail);
            g_draw_str(back, &g_fsmall, 100, y0 + 11, row, act ? C_GREEN : (g_inst[i].complete ? C_TEXT : C_RED));
            y0 += 50;
        }
        y0 += 12;
        g_draw_str(back, &g_fbold, 56, y0, "Downloadable Hugging Face Catalog (1-Click HTTPS)", 0xFFFFFF);
        y0 += 34;
        for (int i = 0; i < g_cat_cnt && y0 < 935; i++, idx++) {
            int sel = (idx == g_mdl_sel);
            g_rrect(back, 56, y0, 1808, 44, 12, sel ? C_SURFACE_HI : C_PANEL, sel ? 255 : 210);
            g_rrect_line(back, 56, y0, 1808, 44, 12, 1, sel ? C_ACCENT_HI : C_LINE, 255);
            char row[240];
            snprintf(row, sizeof(row), "%s   [%s · ~%u MiB]   %s",
                     g_cat[i].id, g_cat[i].arch, g_cat[i].size_mb, g_cat[i].desc);
            g_draw_str(back, &g_fsmall, 84, y0 + 11, row, sel ? 0xFFFFFF : C_TEXT_DIM);
            y0 += 50;
        }

        if (toast[0]) g_draw_str(back, &g_fsmall, 56, 956, toast, g_mdl_del_arm ? C_RED : C_GREEN);
        const GHint mhints[] = {
            { GBTN_UPDOWN, "Select" },
            { GBTN_CROSS, "Load / Download" },
            { GBTN_TRIANGLE, "Cancel Download" },
            { GBTN_OPTIONS, "Delete (x2)" },
            { GBTN_CIRCLE, "Back" },
        };
        g_draw_footer(back, mhints, 5, "Models stream directly over PS4 HTTPS");
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
        g_draw_header(back, "Saved Artwork Gallery", cur_m, busy, vram_u, vram_t, http_srv_address());
        if (g_gal_count <= 0) {
            g_rrect(back, 56, 120, 620, 220, 18, C_SURFACE, 235);
            g_rrect_line(back, 56, 120, 620, 220, 18, 1, C_LINE_HI, 255);
            g_draw_str(back, &g_fbold, 84, 154, "No saved images yet", 0xFFFFFF);
            g_draw_str(back, &g_fsmall, 84, 196, "Press Circle or Square to return to Studio and generate artwork.", C_TEXT_DIM);
            const GHint ehints[] = {{ GBTN_CIRCLE, "Back to Studio" }};
            g_draw_footer(back, ehints, 1, "");
            present();
            return;
        }
        const GalleryItem* it = &g_gal[g_gal_idx];
        g_rrect(back, 56, 112, 612, 868, 18, C_SURFACE, 235);
        g_rrect_line(back, 56, 112, 612, 868, 18, 1, C_LINE_HI, 255);

        char hdr[96];
        snprintf(hdr, sizeof(hdr), "Artwork %d / %d   ·   %ux%u%s", g_gal_idx + 1, g_gal_count,
                 it->width, it->height, it->upscaled == 3 ? " 4K" : (it->upscaled ? " HD" : ""));
        g_draw_str(back, &g_fbold, 82, 134, hdr, 0xFFFFFF);

        char info[160];
        snprintf(info, sizeof(info), "%s  ·  %u steps  ·  CFG %.1f  ·  %s",
                 it->model[0] ? it->model : "Model", it->steps, (double)it->cfg, it->sampler);
        g_draw_str(back, &g_fsmall, 82, 172, info, C_ACCENT_HI);
        snprintf(info, sizeof(info), "Seed: %llu   ·   Render time: %.1fs", (unsigned long long)it->seed, it->sec);
        g_draw_str(back, &g_fsmall, 82, 200, info, C_TEXT_DIM);

        // Prompt inside Garlic LLM blue user bubble
        int rad_b[4] = {22, 22, 8, 22};
        g_rrect4(back, 78, 240, 568, 280, rad_b, 0, C_USER, 245);
        g_draw_wrapped(back, &g_fsmall, 98, 258, 528, 26, 0xFFFFFF, it->prompt[0] ? it->prompt : "(no prompt metadata)", 9);

        if (toast[0]) g_draw_wrapped(back, &g_fsmall, 82, 544, 560, 26, g_del_arm ? C_RED : C_GREEN, toast, 2);

        g_draw_str(back, &g_fsmall, 82, 620, "All gallery PNGs contain embedded Automatic1111 parameters.", C_TEXT_MUTED);
        g_draw_str(back, &g_fsmall, 82, 648, "Download or inspect full resolution at http://", C_TEXT_MUTED);
        g_draw_str(back, &g_fsmall, 82, 674, http_srv_address(), C_GREEN);

        // Right image showcase card
        g_rrect(back, 692, 112, 1172, 868, 18, C_PANEL, 245);
        g_rrect_line(back, 692, 112, 1172, 868, 18, 1, C_LINE_HI, 255);
        if (g_gal_rgb && g_gal_w && g_gal_h) {
            blit_rgb_fit(g_gal_rgb, g_gal_w, g_gal_h, 708, 128, 1140, 836);
        } else {
            g_draw_str(back, &g_fsmall, 1120, 530, "Preview unavailable for this PNG", C_TEXT_MUTED);
        }

        const GHint ghints[] = {
            { GBTN_LEFTRIGHT, "Browse" },
            { GBTN_R1, "Full Screen" },
            { GBTN_CROSS, "Use Prompt" },
            { GBTN_TRIANGLE, "Recreate" },
            { GBTN_L3, "Upscale 1080p" },
            { GBTN_OPTIONS, "Delete (x2)" },
            { GBTN_CIRCLE, "Back" },
        };
        g_draw_footer(back, ghints, 7, "");
        present();
        return;
    }

    // ---- 4. MAIN STUDIO SCREEN (Dropdowns, Sliders & D-Pad/X/O Navigation) ----
    {
        char neg_buf[512];
        int smp_v = 0, ready_v = 0;
        uint32_t batch_v = 1;
        uint64_t seed_v = 0;
        http_srv_lock();
        snprintf(neg_buf, sizeof(neg_buf), "%s", g_job.negative);
        smp_v = g_job.sampler;
        seed_v = g_job.seed;
        batch_v = g_job.batch_count ? g_job.batch_count : 1u;
        ready_v = g_st.ready;
        http_srv_unlock();

        StudioView sv;
        memset(&sv, 0, sizeof(sv));
        sv.model = cur_m;
        sv.neo = eng.neo;
        sv.busy = busy;
        sv.ready = ready_v;
        sv.has_upscaler = eng.has_upscaler;
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
        sv.batch = batch_v;
        sv.seed = seed_v; sv.last_seed = sm_last_seed;
        sv.last_sec = lsec; sv.last_tf = tf;
        sv.preview = preview_rgb; sv.pw = preview_w; sv.ph = preview_h;
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

static void on_progress(void* user, float frac, const char* msg) {
    (void)user;
    http_srv_lock();
    g_st.progress = frac;
    if (g_st.batch_total > 1) {
        snprintf(g_st.stage, sizeof(g_st.stage), "[%u/%u] %s", g_st.batch_cur, g_st.batch_total, msg);
    } else {
        snprintf(g_st.stage, sizeof(g_st.stage), "%s", msg);
    }
    http_srv_unlock();
    // Poll Circle button for cancel while generating
    OrbisPadData pd;
    if (pad_fd >= 0 && scePadReadState(pad_fd, &pd) == 0 && (pd.buttons & ORBIS_PAD_BUTTON_CIRCLE)) {
        http_srv_lock();
        g_job.cancel_req = 1;
        http_srv_unlock();
    }
    draw_ui();
}

static int on_cancel(void* user) {
    (void)user;
    OrbisPadData pd;
    if (pad_fd >= 0 && scePadReadState(pad_fd, &pd) == 0 && (pd.buttons & ORBIS_PAD_BUTTON_CIRCLE)) {
        http_srv_lock();
        g_job.cancel_req = 1;
        http_srv_unlock();
    }
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

static void apply_model_preset(const char* model_dir, const char* name, int is_sd15) {
    ModelPreset pr;
    sdxl_load_preset(model_dir, name, is_sd15, &pr);
    http_srv_lock();
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

int main(void) {
    __libc.threads_minus_1 = 1;
    if (rename(LOG_FILE, "/data/garlic_diffusion_log_prev.txt")) unlink(LOG_FILE);
    glog("GarlicDiffusion v1.10 starting (GPLv3 · PS4 Base/Pro + SD1.5/SDXL + 1080p/4K + Presets/Styles/Characters)");
    log_console_info();
    gallery_init(OUT_ROOT, PRESETS_ROOT);
    styles_init(STYLES_ROOT);
    characters_init(CHARS_ROOT);
    model_dl_init(MODELS_ROOT, REPO_CFG);
    g_num_presets = presets_list(g_presets, PRESETS_MAX_ITEMS);
    g_num_styles = styles_list(g_styles, STYLES_MAX_ITEMS);
    g_num_chars = characters_list(g_chars, CHARS_MAX_ITEMS);
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
    g_job.seed = 0;
    g_job.batch_count = 1;
    g_job.eco_mode = 2; // Default: Quiet (~18 CUs checkerboard + breathers, cool & safe!)
    snprintf(g_st.stage, sizeof(g_st.stage), "Starting HTTP server & loading SDXL weights...");

    http_srv_start(&g_job, &g_st);
    draw_ui();

    // Phase 1: Initialize GPU, kernels, self-test, upscaler, and CU power benchmark ONCE
    if (sdxl_init(&eng, ASSETS_DIR, NULL, on_progress, NULL)) {
        http_srv_lock();
        snprintf(g_st.error, sizeof(g_st.error), "GPU init failed: %s", eng.error);
        http_srv_unlock();
        glog("sdxl_init (GPU) failed: %s", eng.error);
        for (;;) { draw_ui(); sceKernelUsleep(250000); }
    }
    http_srv_lock();
    g_st.gpu_ready = 1;
    g_st.neo = eng.neo;
    g_st.has_upscaler = eng.has_upscaler;
    g_st.gpu_used_mb = (uint32_t)(eng.next >> 20);
    g_st.gpu_total_mb = (uint32_t)(eng.mem_size >> 20);
    if (!eng.neo) g_job.eco_mode = 1; // Base PS4 (18 CUs): Balanced 14CU is cool & responsive!
    g_st.eco_mode = g_job.eco_mode;
    g_st.bench_gemm_tflops = eng.stats.bench_gemm_tflops;
    g_st.bench_attn_ms = eng.stats.bench_attn_ms;
    memcpy(g_st.bench_mode_tf, eng.stats.bench_mode_tf, sizeof(g_st.bench_mode_tf));
    http_srv_unlock();

    // Phase 2: Scan models & auto-fallback across candidates so an incomplete/uploading model NEVER hangs boot!
    int n_found = scan_models();
    int loaded_ok = 0, skipped_cnt = 0;
    char skipped_msg[256] = {0};
    if (n_found > 0) {
        int order[DIFF_MAX_MODELS];
        int used[DIFF_MAX_MODELS] = {0};
        int oc = 0;
        // Priority: DMD2 -> 8step -> waiNSFWIllustrious_v140 -> remaining models
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
        int do_rescan = g_job.rescan_req;
        http_srv_unlock();
        if (pdirty) g_num_presets = presets_list(g_presets, PRESETS_MAX_ITEMS);
        if (sdirty) { g_num_styles = styles_list(g_styles, STYLES_MAX_ITEMS); if (style_idx >= g_num_styles) style_idx = -1; }
        if (cdirty) { g_num_chars = characters_list(g_chars, CHARS_MAX_ITEMS); if (char_idx >= g_num_chars) char_idx = -1; }
        if (do_rescan) {
            int nm = scan_models();
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
                if (g_in_models) models_mgr_refresh();
                snprintf(toast, sizeof(toast), "Downloaded %s!", dl.model_id);
                ui_dirty = 1;
            }
            if (g_in_models && dl.active) ui_dirty = 1;
        }

        if (g_in_credits) {
            if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                g_in_credits = 0; ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_SQUARE) {
                g_in_credits = 0; g_in_gallery = 1; g_gal_full = 0; g_gal_idx = 0; gal_refresh(); ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_OPTIONS) {
                g_in_credits = 0; g_in_models = 1; g_mdl_sel = 0; models_mgr_refresh(); ui_dirty = 1;
            }
        } else if (g_in_models) {
            int total = g_inst_cnt + g_cat_cnt;
            if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                g_in_models = 0; g_mdl_del_arm = 0; toast[0] = 0; ui_dirty = 1;
            } else if ((nav & ORBIS_PAD_BUTTON_DOWN) && total > 0) {
                g_mdl_sel = (g_mdl_sel + 1) % total; g_mdl_del_arm = 0; ui_dirty = 1;
            } else if ((nav & ORBIS_PAD_BUTTON_UP) && total > 0) {
                g_mdl_sel = (g_mdl_sel + total - 1) % total; g_mdl_del_arm = 0; ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) {
                model_dl_cancel(); snprintf(toast, sizeof(toast), "Cancelling download..."); ui_dirty = 1;
            } else if ((pressed & ORBIS_PAD_BUTTON_CROSS) && total > 0) {
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
            } else if ((pressed & ORBIS_PAD_BUTTON_OPTIONS) && g_mdl_sel < g_inst_cnt) {
                if (!g_mdl_del_arm) {
                    g_mdl_del_arm = 1;
                    snprintf(toast, sizeof(toast), "Press [OPTIONS] again to DELETE %s", g_inst[g_mdl_sel].name);
                } else {
                    http_srv_lock();
                    char act[64]; snprintf(act, sizeof(act), "%s", g_st.model);
                    http_srv_unlock();
                    int rc = model_dl_delete(g_inst[g_mdl_sel].name, act);
                    g_mdl_del_arm = 0;
                    if (rc == 0) { scan_models(); models_mgr_refresh(); snprintf(toast, sizeof(toast), "Deleted model"); }
                    else if (rc == -2) snprintf(toast, sizeof(toast), "Switch away from active model before deleting");
                    else snprintf(toast, sizeof(toast), "Delete failed");
                }
                ui_dirty = 1;
            }
        } else if (g_in_gallery) {
            if (g_gal_full) {
                if (pressed) { g_gal_full = 0; ui_dirty = 1; }
            } else if (pressed || nav) {
                if (pressed & (ORBIS_PAD_BUTTON_CIRCLE | ORBIS_PAD_BUTTON_SQUARE)) {
                    g_in_gallery = 0; g_del_arm = 0; toast[0] = 0; ui_dirty = 1;
                } else if (nav & ORBIS_PAD_BUTTON_RIGHT) {
                    if (g_gal_count > 0) { g_gal_idx = (g_gal_idx + 1) % g_gal_count; toast[0] = 0; gal_load_current(); ui_dirty = 1; }
                } else if (nav & ORBIS_PAD_BUTTON_LEFT) {
                    if (g_gal_count > 0) { g_gal_idx = (g_gal_idx + g_gal_count - 1) % g_gal_count; toast[0] = 0; gal_load_current(); ui_dirty = 1; }
                } else if (pressed & ORBIS_PAD_BUTTON_R1) {
                    if (g_gal_rgb) { g_gal_full = 1; ui_dirty = 1; }
                } else if ((pressed & ORBIS_PAD_BUTTON_CROSS) && g_gal_count > 0) {
                    const GalleryItem* it = &g_gal[g_gal_idx];
                    http_srv_lock();
                    if (it->prompt[0]) snprintf(g_job.prompt, sizeof(g_job.prompt), "%s", it->prompt);
                    if (it->negative[0]) snprintf(g_job.negative, sizeof(g_job.negative), "%s", it->negative);
                    if (it->steps) g_job.steps = it->steps;
                    if (it->cfg > 0) g_job.cfg = it->cfg;
                    g_job.sampler = strstr(it->sampler, "DPM") ? 3 : ((strcmp(it->sampler, "LCM") == 0) ? 2 : (strcmp(it->sampler, "Euler") == 0 ? 1 : 0));
                    http_srv_unlock();
                    snprintf(toast, sizeof(toast), "Loaded prompt from %s", it->id);
                    g_in_gallery = 0; ui_dirty = 1;
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
                    http_srv_unlock();
                    g_in_gallery = 0; toast[0] = 0; ui_dirty = 1;
                } else if ((pressed & ORBIS_PAD_BUTTON_L3) && g_gal_count > 0 && eng.has_upscaler) {
                    http_srv_lock();
                    snprintf(g_job.gal_id, sizeof(g_job.gal_id), "%s", g_gal[g_gal_idx].id);
                    g_job.gal_op = 1;
                    g_job.cancel_req = 0;
                    http_srv_unlock();
                    g_in_gallery = 0; toast[0] = 0; ui_dirty = 1;
                } else if ((pressed & ORBIS_PAD_BUTTON_OPTIONS) && g_gal_count > 0) {
                    if (!g_del_arm) {
                        g_del_arm = 1;
                        snprintf(toast, sizeof(toast), "Press [OPTIONS] again to DELETE %s", g_gal[g_gal_idx].id);
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
                case ROW_PRESET:
                    if (sel == 0) {
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
                                g_num_presets = presets_list(g_presets, PRESETS_MAX_ITEMS);
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
                        http_srv_lock();
                        snprintf(g_job.prompt, sizeof(g_job.prompt), "%s", g_presets[preset_idx].prompt);
                        if (g_presets[preset_idx].negative[0])
                            snprintf(g_job.negative, sizeof(g_job.negative), "%s", g_presets[preset_idx].negative);
                        http_srv_unlock();
                        snprintf(toast, sizeof(toast), "Preset: %s", g_presets[preset_idx].name);
                    }
                    break;
                case ROW_CHAR:
                    char_idx = sel - 1;
                    if (char_idx >= g_num_chars) char_idx = -1;
                    if (char_idx >= 0) snprintf(toast, sizeof(toast), "Character: %s (%s)", g_chars[char_idx].name, g_chars[char_idx].series);
                    else snprintf(toast, sizeof(toast), "Character overlay: Off");
                    break;
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
                case ROW_BATCH:
                    if (sel >= 0 && sel < BATCH_COUNT) {
                        http_srv_lock(); g_job.batch_count = BATCH_LIST[sel]; http_srv_unlock();
                    }
                    break;
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
        } else {
            // ---- MAIN STUDIO SETTINGS LIST ----
            if (nav & ORBIS_PAD_BUTTON_DOWN) {
                sm_focus = (sm_focus + 1) % ROW_COUNT;
                toast[0] = 0; ui_dirty = 1;
            } else if (nav & ORBIS_PAD_BUTTON_UP) {
                sm_focus = (sm_focus + ROW_COUNT - 1) % ROW_COUNT;
                toast[0] = 0; ui_dirty = 1;
            } else if (nav & (ORBIS_PAD_BUTTON_LEFT | ORBIS_PAD_BUTTON_RIGHT)) {
                int dir = (nav & ORBIS_PAD_BUTTON_RIGHT) ? 1 : -1;
                toast[0] = 0;
                switch (sm_focus) {
                case ROW_PRESET:
                    if (g_num_presets > 0) {
                        preset_idx = (preset_idx + g_num_presets + dir) % g_num_presets;
                        http_srv_lock();
                        snprintf(g_job.prompt, sizeof(g_job.prompt), "%s", g_presets[preset_idx].prompt);
                        if (g_presets[preset_idx].negative[0])
                            snprintf(g_job.negative, sizeof(g_job.negative), "%s", g_presets[preset_idx].negative);
                        http_srv_unlock();
                        snprintf(toast, sizeof(toast), "Preset: %s", g_presets[preset_idx].name);
                    }
                    break;
                case ROW_CHAR:
                    if (g_num_chars > 0) {
                        int n = g_num_chars + 1;
                        int cur = (char_idx + 1 + n + dir) % n;
                        char_idx = cur - 1;
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
                        int nb = (int)(g_job.batch_count ? g_job.batch_count : 1u) + dir;
                        if (nb < 1) nb = 20;
                        if (nb > 20) nb = 1;
                        g_job.batch_count = (uint32_t)nb;
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
                    sm_dd_n = BATCH_COUNT;
                    {
                        uint32_t cur_b = 1;
                        http_srv_lock(); cur_b = g_job.batch_count ? g_job.batch_count : 1u; http_srv_unlock();
                        sm_dd_cur = 0;
                        for (int i = 0; i < BATCH_COUNT; i++) {
                            if (BATCH_LIST[i] == cur_b) { sm_dd_cur = i; break; }
                        }
                    }
                    sm_dd_sel = sm_dd_cur;
                    sm_dd_top = 0; sm_dd_clamp();
                    sm_dd_open = 1;
                    break;
                case ROW_GEN:
                    http_srv_lock();
                    if (g_st.ready) {
                        g_job.cancel_req = 0;
                        g_tv_gen = 1;
                        g_job.pending = 1;
                    } else {
                        snprintf(toast, sizeof(toast), "No complete model loaded - select Model or press OPTIONS");
                    }
                    http_srv_unlock();
                    break;
                }
                ui_dirty = 1;
            } else if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) {
                http_srv_lock();
                if (g_st.ready) {
                    g_job.cancel_req = 0;
                    g_tv_gen = 1;
                    g_job.pending = 1;
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
                g_in_credits = 1; toast[0] = 0; ui_dirty = 1;
            }
        }

        // Check for queued Gallery Upscale / Resize job (from Web UI or TV Gallery)
        int gop = 0;
        char gid[64] = {0};
        uint32_t gw = 0, gh = 0;
        int geco = 2;
        http_srv_lock();
        if (g_job.gal_op && !g_st.busy) {
            gop = g_job.gal_op;
            snprintf(gid, sizeof(gid), "%s", g_job.gal_id);
            gw = g_job.gal_w; gh = g_job.gal_h;
            geco = g_job.eco_mode;
            g_job.gal_op = 0;
            g_st.busy = 1;
            g_st.error[0] = 0;
            g_st.progress = 0.0f;
        }
        http_srv_unlock();
        if (gop) {
            GalleryItem src_meta;
            size_t psz = 0;
            uint8_t* spng = (gallery_get(gid, &src_meta) == 0) ? gallery_load_png(gid, &psz) : NULL;
            uint32_t sw = 0, sh = 0;
            uint8_t* srgb = spng ? png_decode_rgb(spng, psz, &sw, &sh) : NULL;
            free(spng);
            if (srgb) {
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
                }
                free(srgb);
                if (ok == 0 && drgb) {
                    size_t nsz = 0;
                    src_meta.width = dw; src_meta.height = dh;
                    if (gop >= 1 && gop <= 3) src_meta.upscaled = gop;
                    char pmeta[2048];
                    format_a1111_params(pmeta, sizeof(pmeta), src_meta.prompt, src_meta.negative,
                                        src_meta.steps, src_meta.sampler, src_meta.cfg, src_meta.seed,
                                        dw, dh, src_meta.model, src_meta.upscaled);
                    uint8_t* npng = png_encode_rgb_meta(drgb, dw, dh, pmeta, &nsz);
                    if (npng) {
                        char nid[64] = {0};
                        gallery_save(npng, nsz, &src_meta, nid, sizeof(nid));
                        free(preview_rgb);
                        preview_rgb = drgb; preview_w = dw; preview_h = dh;
                        http_srv_lock();
                        free(g_st.png_data);
                        g_st.png_data = npng; g_st.png_size = nsz;
                        g_st.width = dw; g_st.height = dh;
                        g_st.upscale_sec = eng.stats.upscale_ms / 1000.0;
                        g_st.image_gen_id++;
                        snprintf(g_st.last_gallery_id, sizeof(g_st.last_gallery_id), "%s", nid);
                        g_st.progress = 1.0f;
                        snprintf(g_st.stage, sizeof(g_st.stage), "Saved %s (%ux%u)", nid, dw, dh);
                        g_st.busy = 0;
                        http_srv_unlock();
                        glog("gallery op=%d %s -> %s (%ux%u)", gop, gid, nid, dw, dh);
                    } else {
                        free(drgb);
                        http_srv_lock(); g_st.busy = 0; http_srv_unlock();
                    }
                } else {
                    free(drgb);
                    http_srv_lock();
                    snprintf(g_st.stage, sizeof(g_st.stage), ok == 1 ? "Cancelled" : "Gallery op failed");
                    g_st.busy = 0;
                    http_srv_unlock();
                }
            } else {
                http_srv_lock();
                snprintf(g_st.error, sizeof(g_st.error), "Cannot decode %s.png", gid);
                g_st.busy = 0;
                http_srv_unlock();
            }
            ui_dirty = 1;
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
                g_st.error[0] = 0;
                snprintf(g_st.model, sizeof(g_st.model), "%s", eng.model_name);
                snprintf(g_st.stage, sizeof(g_st.stage), "Switched to %s (%ux%u, %u stp, CFG %.1f)",
                         eng.model_name, g_job.width, g_job.height, g_job.steps, (double)g_job.cfg);
                g_st.progress = 1.0f;
                g_st.busy = 0;
                http_srv_unlock();
                toast[0] = 0;
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
            ui_dirty = 1;
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
                if (char_idx >= 0 && char_idx < g_num_chars && g_chars[char_idx].prompt[0]) {
                    char merged[1024];
                    snprintf(merged, sizeof(merged), "%s, %s", g_chars[char_idx].prompt, tmp_p);
                    snprintf(tmp_p, sizeof(tmp_p), "%s", merged);
                    if (g_chars[char_idx].negative[0]) {
                        char mn[512];
                        snprintf(mn, sizeof(mn), "%s, %s", tmp_n, g_chars[char_idx].negative);
                        snprintf(tmp_n, sizeof(tmp_n), "%s", mn);
                    }
                }
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
                snprintf(run_job.prompt, sizeof(run_job.prompt), "%s", tmp_p);
                snprintf(run_job.negative, sizeof(run_job.negative), "%s", tmp_n);
            }
            uint32_t rw = (run_job.width + 32u) & ~63u;
            uint32_t rh = (run_job.height + 32u) & ~63u;
            if (rw < 256u) rw = 256u;
            if (rh < 256u) rh = 256u;
            if (rw > 1280u) rw = 1280u;
            if (rh > 1280u) rh = 1280u;
            uint64_t max_lat = (eng.is_sd15 || !eng.neo) ? 12288ull : 16384ull;
            while ((uint64_t)(rw / 8u) * (rh / 8u) > max_lat) {
                if (rw >= rh) rw -= 64u;
                else rh -= 64u;
            }
            uint32_t rsteps = run_job.steps < 1u ? 20u : (run_job.steps > 64u ? 64u : run_job.steps);
            float rcfg = run_job.cfg <= 0.0f ? 7.0f : run_job.cfg;
            uint32_t btotal = run_job.batch_count;
            if (btotal < 1u) btotal = 1u;
            if (btotal > 20u) btotal = 20u;
            for (uint32_t bi = 0; bi < btotal; bi++) {
                http_srv_lock();
                int cancelled = g_job.cancel_req;
                g_st.batch_cur = bi + 1u;
                g_st.batch_total = btotal;
                g_st.progress = 0.0f;
                http_srv_unlock();
                if (cancelled) break;

                uint64_t seed = (bi == 0 && run_job.seed)
                    ? run_job.seed
                    : ((sceKernelReadTsc() ^ ((uint64_t)(bi + 1u) * 2654435761ull)) & 0xFFFFFFFFull);
                if (seed == 0) seed = 1;
                sm_last_seed = seed;
                SdxlParams p = {run_job.prompt, run_job.negative, rw, rh, rsteps, rcfg, seed, run_job.eco_mode, run_job.sampler};
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
                    format_a1111_params(pmeta, sizeof(pmeta), p.prompt, p.negative, p.steps, smp_str,
                                        p.cfg_scale, seed, out_w, out_h, eng.model_name, did_up);
                    size_t png_sz = 0;
                    uint8_t* png = png_encode_rgb_meta(rgb, out_w, out_h, pmeta, &png_sz);
                    char saved_id[64] = {0};
                    if (png && png_sz) {
                        GalleryItem gm;
                        memset(&gm, 0, sizeof(gm));
                        snprintf(gm.model, sizeof(gm.model), "%s", eng.model_name);
                        snprintf(gm.prompt, sizeof(gm.prompt), "%s", p.prompt);
                        snprintf(gm.negative, sizeof(gm.negative), "%s", p.negative);
                        snprintf(gm.sampler, sizeof(gm.sampler), "%s", smp_str);
                        gm.width = out_w; gm.height = out_h; gm.steps = p.steps;
                        gm.cfg = p.cfg_scale; gm.seed = seed; gm.eco = p.eco_mode;
                        gm.upscaled = did_up;
                        gm.sec = eng.last_gen_sec;
                        if (gallery_save(png, png_sz, &gm, saved_id, sizeof(saved_id)) == 0) {
                            glog("gallery auto-saved %s (%ux%u)", saved_id, out_w, out_h);
                        }
                    }
                    free(preview_rgb);
                    preview_rgb = rgb;
                    preview_w = out_w;
                    preview_h = out_h;
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
                    draw_ui();
                } else {
                    free(rgb);
                    http_srv_lock();
                    if (rc == 1) snprintf(g_st.stage, sizeof(g_st.stage), btotal > 1u ? "Batch cancelled at [%u/%u]" : "Cancelled", bi + 1u, btotal);
                    else snprintf(g_st.error, sizeof(g_st.error), "%s", eng.error[0] ? eng.error : "generation error");
                    http_srv_unlock();
                    glog("gen ended rc=%d (%s)", rc, eng.error);
                    break;
                }
            }
            http_srv_lock();
            g_st.batch_cur = 0;
            g_st.batch_total = 0;
            g_st.busy = 0;
            http_srv_unlock();
            ui_dirty = 1;
        }

        if (ui_dirty || ++idle_ticks >= 60) {
            draw_ui();
            idle_ticks = 0;
        } else {
            sceKernelUsleep(16000); // Sleep peacefully while idle (~0.3% CPU!)
        }
    }
    return 0;
}
