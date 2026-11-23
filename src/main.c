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

#include "font8x8.h"
#include "gallery.h"
#include "http_srv.h"
#include "model_dl.h"
#include "png_write.h"
#include "sdxl.h"

#define SCREEN_W 1920
#define SCREEN_H 1080
#define ASSETS_DIR "/app0/assets"
#define MODELS_ROOT "/data/garlic-diffusion/models"
#define OUT_ROOT "/data/garlic-diffusion/out"
#define PRESETS_ROOT "/data/garlic-diffusion/presets"
#define STYLES_ROOT "/data/garlic-diffusion/styles"
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

// ---- Simple Crisp Software Renderer -------------------------------------------------------------
static void rect(int x, int y, int w, int h, uint32_t rgb) {
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > SCREEN_W) w = SCREEN_W - x;
    if (y + h > SCREEN_H) h = SCREEN_H - y;
    uint32_t c = 0x80000000u | rgb;
    for (int j = 0; j < h; j++) {
        uint32_t* row = back + (size_t)(y + j) * SCREEN_W + x;
        for (int i = 0; i < w; i++) row[i] = c;
    }
}

static void draw_text(int x, int y, int scale, uint32_t rgb, const char* s) {
    uint32_t c = 0x80000000u | rgb;
    int cx = x;
    for (; s && *s; s++) {
        unsigned char ch = (unsigned char)*s;
        if (ch == '\n') { cx = x; y += 10 * scale; continue; }
        if (ch < 32 || ch > 126) ch = '?';
        const uint8_t* g = FONT8X8[ch - 32];
        for (int r = 0; r < 8; r++) {
            uint8_t bits = g[r];
            for (int col = 0; col < 8; col++) {
                if (!(bits & (0x80u >> col))) continue;
                for (int sy = 0; sy < scale; sy++) {
                    int py = y + r * scale + sy;
                    if (py < 0 || py >= SCREEN_H) continue;
                    for (int sx = 0; sx < scale; sx++) {
                        int px = cx + col * scale + sx;
                        if (px >= 0 && px < SCREEN_W) back[(size_t)py * SCREEN_W + px] = c;
                    }
                }
            }
        }
        cx += 8 * scale;
    }
}

static void draw_wrapped(int x, int y, int max_w, int scale, uint32_t rgb, const char* s, int max_lines) {
    int cols = max_w / (8 * scale);
    if (cols < 8) cols = 8;
    char line[128];
    int lines = 0;
    while (s && *s && lines < max_lines) {
        int n = 0;
        while (s[n] && s[n] != '\n' && n < cols && n < (int)sizeof(line) - 1) n++;
        if (s[n] && s[n] != '\n' && s[n] != ' ') {
            int sp = n;
            while (sp > 0 && s[sp] != ' ') sp--;
            if (sp > 0) n = sp;
        }
        memcpy(line, s, (size_t)n);
        line[n] = 0;
        draw_text(x, y + lines * (11 * scale), scale, rgb, line);
        s += n;
        while (*s == ' ' || *s == '\n') s++;
        lines++;
    }
}

// ---- App State ----------------------------------------------------------------------------------
static SdxlEngine eng;
static DiffJob g_job;
static DiffStatus g_st;
static uint8_t* preview_rgb = NULL;
static uint32_t preview_w = 0, preview_h = 0;
static char toast[128] = "";

// Dynamic User Presets + TV Gallery + TV Model Manager state
static UserPreset g_presets[PRESETS_MAX_ITEMS];
static int g_num_presets = 0;
static GalleryItem g_gal[GALLERY_MAX_ITEMS];
static int g_gal_count = 0, g_gal_idx = 0;
static int g_in_gallery = 0, g_gal_full = 0, g_del_arm = 0;
static uint8_t* g_gal_rgb = NULL;
static uint32_t g_gal_w = 0, g_gal_h = 0;

static int g_in_models = 0, g_mdl_sel = 0, g_mdl_del_arm = 0, g_last_dl_fin = 0;
static InstalledModel g_inst[MODEL_DL_MAX_INSTALLED];
static int g_inst_cnt = 0;
static CatalogModel g_cat[MODEL_DL_MAX_CATALOG];
static int g_cat_cnt = 0;

static void format_a1111_params(char* out, size_t max, const char* prompt, const char* neg,
                                uint32_t steps, const char* sampler, float cfg, uint64_t seed,
                                uint32_t w, uint32_t h, const char* model, int upscaled) {
    snprintf(out, max,
             "%s\nNegative prompt: %s\nSteps: %u, Sampler: %s, CFG scale: %.1f, Seed: %llu, Size: %ux%u, Model: %s, Clip skip: 2%s, Generator: Garlic Diffusion PS4",
             prompt ? prompt : "", neg ? neg : "", steps, (sampler && sampler[0]) ? sampler : "Euler a",
             (double)cfg, (unsigned long long)seed, w, h, (model && model[0]) ? model : "GarlicDiffusion",
             upscaled ? ", Hires upscaler: RealESRGAN Anime v3" : "");
}

static void models_mgr_refresh(void) {
    g_inst_cnt = model_dl_list_installed(g_inst, MODEL_DL_MAX_INSTALLED);
    g_cat_cnt = model_dl_get_catalog(g_cat, MODEL_DL_MAX_CATALOG);
    int total = g_inst_cnt + g_cat_cnt;
    if (g_mdl_sel >= total) g_mdl_sel = total > 0 ? total - 1 : 0;
    g_mdl_del_arm = 0;
}

static const uint32_t RES_LIST[][2] = {{512, 512}, {512, 768}, {768, 512}, {768, 768}, {768, 1024}, {896, 512}, {832, 1216}, {1024, 1024}};
static const uint32_t STEP_LIST[] = {4, 6, 8, 10, 12, 16, 20, 25, 30};
static int preset_idx = 0, res_idx = 3, model_idx = 0;

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
    if (g_in_models) {
        rect(0, 0, SCREEN_W, SCREEN_H, 0x080D18);
        rect(0, 0, SCREEN_W, 92, 0x101829);
        rect(0, 92, SCREEN_W, 2, 0x1F2E4A);
        draw_text(40, 26, 3, 0x5FB2FF, "MODEL MANAGER & DOWNLOADER");
        char repo[128], rline[200];
        model_dl_get_repo(repo, sizeof(repo));
        snprintf(rline, sizeof(rline), "HF Repo: %s  (Configure on Web UI -> Models tab)",
                 repo[0] ? repo : "(not set - open Web UI -> Models tab)");
        draw_text(740, 38, 2, repo[0] ? 0x45D6A0 : 0xF59ADB, rline);

        ModelDlStatus dl;
        model_dl_get_status(&dl);
        int y0 = 114;
        if (dl.active || dl.error[0] || dl.stage[0]) {
            rect(40, y0, SCREEN_W - 80, 74, 0x131D32);
            char dmsg[220];
            if (dl.active) {
                float pct = dl.bytes_total ? (100.0f * (float)dl.bytes_done / (float)dl.bytes_total) : 0.0f;
                snprintf(dmsg, sizeof(dmsg), "Downloading %s [%d/%d] %s: %llu / %llu MiB (%.0f%%) %.1f MB/s",
                         dl.model_id, dl.file_idx, dl.file_total, dl.cur_file,
                         (unsigned long long)(dl.bytes_done >> 20), (unsigned long long)(dl.bytes_total >> 20),
                         (double)pct, dl.speed_mbs);
                draw_text(60, y0 + 12, 2, 0x5FB2FF, dmsg);
                rect(60, y0 + 44, SCREEN_W - 120, 16, 0x1F2E4A);
                int bw = (int)((float)(SCREEN_W - 120) * (pct / 100.0f));
                if (bw > SCREEN_W - 120) bw = SCREEN_W - 120;
                rect(60, y0 + 44, bw, 16, 0x45D6A0);
            } else {
                draw_text(60, y0 + 26, 2, dl.error[0] ? 0xFF7A85 : 0x45D6A0, dl.error[0] ? dl.error : dl.stage);
            }
            y0 += 92;
        }

        draw_text(40, y0, 2, 0x8BA2C4, "INSTALLED MODELS ON PS4 (/data/garlic-diffusion/models):");
        y0 += 30;
        http_srv_lock();
        char cur_m[64];
        snprintf(cur_m, sizeof(cur_m), "%s", g_st.model);
        http_srv_unlock();

        int idx = 0;
        for (int i = 0; i < g_inst_cnt && y0 < 880; i++, idx++) {
            int sel = (idx == g_mdl_sel);
            rect(40, y0, SCREEN_W - 80, 42, sel ? 0x1D3357 : 0x101829);
            char row[220];
            int act = (strcmp(g_inst[i].name, cur_m) == 0);
            snprintf(row, sizeof(row), "%s %-30s [%4s] %4u MiB  %s%s",
                     sel ? ">" : " ", g_inst[i].name, g_inst[i].arch, g_inst[i].size_mb,
                     act ? "*ACTIVE* " : "", g_inst[i].detail);
            draw_text(56, y0 + 12, 2, act ? 0x45D6A0 : (g_inst[i].complete ? 0xE8EFFA : 0xFF7A85), row);
            y0 += 46;
        }
        y0 += 14;
        draw_text(40, y0, 2, 0x8BA2C4, "DOWNLOADABLE CATALOG (Hugging Face):");
        y0 += 30;
        for (int i = 0; i < g_cat_cnt && y0 < 930; i++, idx++) {
            int sel = (idx == g_mdl_sel);
            rect(40, y0, SCREEN_W - 80, 42, sel ? 0x1D3357 : 0x0D1424);
            char row[220];
            snprintf(row, sizeof(row), "%s %-30s [%4s] ~%4u MiB  %s",
                     sel ? ">" : " ", g_cat[i].id, g_cat[i].arch, g_cat[i].size_mb, g_cat[i].desc);
            draw_text(56, y0 + 12, 2, sel ? 0x66B8FF : 0xA8BCD9, row);
            y0 += 46;
        }

        if (toast[0]) draw_text(40, 962, 2, g_mdl_del_arm ? 0xFF7A85 : 0x45D6A0, toast);
        rect(40, 1002, SCREEN_W - 80, 2, 0x1F2E4A);
        draw_text(40, 1020, 2, 0xE8EFFA, "[Up/Down] Select   [X] Load / Download   [Triangle] Cancel Download   [OPTIONS x2] Delete   [Circle] Back");
        present();
        return;
    }

    if (g_in_gallery) {
        rect(0, 0, SCREEN_W, SCREEN_H, 0x05080F);
        if (g_gal_full && g_gal_rgb) {
            blit_rgb_fit(g_gal_rgb, g_gal_w, g_gal_h, 0, 0, SCREEN_W, SCREEN_H);
            present();
            return;
        }
        rect(0, 0, 620, SCREEN_H, 0x101829);
        rect(620, 0, 2, SCREEN_H, 0x1F2E4A);
        draw_text(32, 28, 3, 0x5FB2FF, "GALLERY VIEWER");
        if (g_gal_count <= 0) {
            draw_text(32, 96, 2, 0x8BA2C4, "No saved images yet.");
            draw_text(32, 136, 2, 0x45D6A0, "Press [Circle] or [Square] to return.");
            present();
            return;
        }
        const GalleryItem* it = &g_gal[g_gal_idx];
        char hdr[96];
        snprintf(hdr, sizeof(hdr), "Image %d / %d  (%ux%u%s)", g_gal_idx + 1, g_gal_count,
                 it->width, it->height, it->upscaled ? " HD" : "");
        draw_text(32, 72, 2, 0x45D6A0, hdr);
        char info[128];
        snprintf(info, sizeof(info), "%s | %ustp CFG%.1f %s", it->model[0] ? it->model : "?",
                 it->steps, (double)it->cfg, it->sampler);
        draw_wrapped(32, 108, 560, 2, 0xE8EFFA, info, 2);
        snprintf(info, sizeof(info), "Seed: %llu  (%.1fs)", (unsigned long long)it->seed, it->sec);
        draw_text(32, 160, 2, 0x8BA2C4, info);
        rect(32, 198, 556, 2, 0x1F2E4A);
        draw_text(32, 214, 2, 0x8BA2C4, "PROMPT:");
        rect(32, 244, 556, 280, 0x0B1220);
        draw_wrapped(44, 258, 532, 2, 0xE8EFFA, it->prompt[0] ? it->prompt : "(no metadata)", 11);
        if (toast[0]) draw_wrapped(32, 548, 556, 2, g_del_arm ? 0xFF6B76 : 0x45D6A0, toast, 2);

        rect(32, 650, 556, 2, 0x1F2E4A);
        draw_text(32, 672, 2, 0xE8EFFA, "[Left / Right] Browse Images");
        draw_text(32, 708, 2, 0x66B2FF, "[R1]           Full-Screen 100% View");
        draw_text(32, 744, 2, 0x8AB4FF, "[X]            Load Prompt to Studio");
        draw_text(32, 780, 2, 0x45D6A0, "[Triangle]     Recreate (New Seed)");
        draw_text(32, 816, 2, 0xF59ADB, "[L3]           AI Upscale 1080p");
        draw_text(32, 852, 2, 0xFF7A85, "[OPTIONS x2]   Delete Image");
        draw_text(32, 888, 2, 0xA3B5D2, "[Circle/Sq]    Back to Studio");

        if (g_gal_rgb && g_gal_w && g_gal_h) {
            blit_rgb_fit(g_gal_rgb, g_gal_w, g_gal_h, 640, 20, 1260, 1040);
        } else {
            draw_text(980, 520, 2, 0x617594, "Preview unavailable for this PNG");
        }
        present();
        return;
    }

    rect(0, 0, SCREEN_W, SCREEN_H, 0x0A101D);
    rect(0, 0, 660, SCREEN_H, 0x101829);
    rect(660, 0, 2, SCREEN_H, 0x1F2E4A);

    draw_text(36, 32, 3, 0x5FB2FF, "GARLIC DIFFUSION");
    int cur_temp = sdxl_read_temp_c(NULL, NULL);
    char sub[128];
    if (cur_temp > 0)
        snprintf(sub, sizeof(sub), "%s | %u MiB | %dC", eng.neo ? "PS4 Pro Neo" : "PS4 Base", (unsigned)(eng.next >> 20), cur_temp);
    else
        snprintf(sub, sizeof(sub), "%s | %u / %u MiB GPU", eng.neo ? "PS4 Pro Neo" : "PS4 Base",
                 (unsigned)(eng.next >> 20), (unsigned)(eng.mem_size >> 20));
    draw_text(36, 72, 2, cur_temp >= 83 ? 0xFF7A85 : 0x8BA2C4, sub);

    char url[96];
    snprintf(url, sizeof(url), "Web/A1111: http://%s", http_srv_address());
    draw_text(36, 106, 2, 0x45D6A0, url);

    http_srv_lock();
    g_st.temp_c = cur_temp;
    char mdl[64], stg[128], err[256], pr[1024];
    float prog = g_st.progress;
    int busy = g_st.busy, ready = g_st.ready, eco = g_job.eco_mode, ups = g_job.upscale, nmod = g_st.num_models;
    double lsec = g_st.last_sec, tf = g_st.eff_tflops, rest = g_st.cooldown_sec;
    int pk = g_st.temp_peak_c;
    uint32_t w = g_job.width, h = g_job.height, steps = g_job.steps;
    float cfg = g_job.cfg;
    snprintf(mdl, sizeof(mdl), "%s", g_st.model);
    snprintf(stg, sizeof(stg), "%s", g_st.stage);
    snprintf(err, sizeof(err), "%s", g_st.error);
    snprintf(pr, sizeof(pr), "%s", g_job.prompt);
    http_srv_unlock();
    if (eco < 0 || eco > 3) eco = 2;
    if (ups < 0 || ups > 2) ups = 0;
    static const char* UPS_SHORT[3] = {"1x", "1080p", "2xHD"};

    rect(32, 142, 596, 2, 0x1F2E4A);
    draw_text(36, 158, 2, 0x8BA2C4, "MODEL:");
    draw_text(148, 158, 2, mdl[0] ? 0xE8EFFA : 0xFF7A85, mdl[0] ? mdl : (ready ? "loading..." : "(none - press OPTIONS / Web UI)"));

    char cfgline[128];
    snprintf(cfgline, sizeof(cfgline), "%ux%u->%s | %ustp | CFG%.1f | %s",
             w, h, UPS_SHORT[ups], steps, (double)cfg, sdxl_eco_name_ex(eco, eng.neo));
    draw_text(36, 194, 2, 0xE8EFFA, cfgline);

    draw_text(36, 238, 2, 0x8BA2C4, "PROMPT (X to edit, D-Pad presets):");
    rect(32, 268, 596, 210, 0x0B1220);
    draw_wrapped(44, 282, 572, 2, 0xE8EFFA, pr, 8);

    // Progress bar
    rect(32, 510, 596, 20, 0x1B2740);
    int fw = (int)(596.0f * (prog < 0 ? 0 : prog > 1 ? 1 : prog));
    rect(32, 510, fw, 20, busy ? 0x3FA9F5 : (ready ? 0x3DDC97 : 0xF59ADB));
    draw_wrapped(36, 544, 590, 2, err[0] ? 0xFF6B76 : 0xD2E1F7, err[0] ? err : stg, 3);
    if (lsec > 0.0) {
        char tm[112];
        if (pk > 0) snprintf(tm, sizeof(tm), "Last: %.1fs | %.2f TF | Rest %.1fs | %dC", lsec, tf, rest, pk);
        else snprintf(tm, sizeof(tm), "Last: %.1fs | %.2f TFLOPS | Rest %.1fs", lsec, tf, rest);
        draw_text(36, 622, 2, 0x8BA2C4, tm);
    }
    if (toast[0]) draw_text(36, 660, 2, 0x45D6A0, toast);

    // Controls
    rect(32, 706, 596, 2, 0x1F2E4A);
    draw_text(36, 722, 2, 0x8AB4FF, "[X]        Edit Prompt (OSK / USB KB)");
    draw_text(36, 754, 2, 0x45D6A0, "[Triangle] Generate Image");
    draw_text(36, 786, 2, 0xFF7A85, "[Circle]   Cancel Generation");
    draw_text(36, 818, 2, 0xF59ADB, "[Square]   Open Image Gallery");
    draw_text(36, 850, 2, 0xA3B5D2, "[L1 / R1]  Resolution");
    draw_text(36, 882, 2, 0xA3B5D2, "[L2 / R2]  Steps (4..30)");
    draw_text(36, 914, 2, 0xA3B5D2, "[L3 / R3]  Upscale(1080p) / Power Mode");
    draw_text(36, 946, 2, 0xA3B5D2, "[Left/Right] Presets   [Up] Model Manager");
    if (nmod > 1 || !ready) draw_text(36, 978, 2, 0x66B2FF, "[OPTIONS]  Switch / Reload Model");

    // Right panel: image preview
    int px = 690, py = 30, pw = 1200, ph = 1020;
    rect(px, py, pw, ph, 0x070B14);
    if (preview_rgb && preview_w > 0 && preview_h > 0) {
        blit_rgb_fit(preview_rgb, preview_w, preview_h, px, py, pw, ph);
    } else {
        draw_text(px + 330, py + 490, 2, 0x4A5E7D, "Press [Triangle] or open Web UI to generate");
    }
    present();
}

static void on_progress(void* user, float frac, const char* msg) {
    (void)user;
    http_srv_lock();
    g_st.progress = frac;
    snprintf(g_st.stage, sizeof(g_st.stage), "%s", msg);
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

static void edit_prompt_osk(void) {
    static wchar_t buf[513];
    memset(buf, 0, sizeof(buf));
    for (int i = 0; g_job.prompt[i] && i < 512; i++) buf[i] = (uint8_t)g_job.prompt[i];
    OrbisImeDialogSetting s;
    memset(&s, 0, sizeof(s));
    s.userId = user_id;
    s.type = ORBIS_TYPE_DEFAULT;
    s.enterLabel = ORBIS_BUTTON_LABEL_SEND;
    s.maxTextLength = 512;
    s.inputTextBuffer = buf;
    s.posx = SCREEN_W / 2;
    s.posy = SCREEN_H / 2;
    s.horizontalAlignment = ORBIS_H_CENTER;
    s.verticalAlignment = ORBIS_V_CENTER;
    s.title = L"Garlic Diffusion Prompt";
    if (sceImeDialogInit(&s, NULL) < 0) return;
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
    if (res.endstatus == ORBIS_DIALOG_OK && buf[0]) {
        http_srv_lock();
        int n = 0;
        for (int i = 0; buf[i] && n < (int)sizeof(g_job.prompt) - 1; i++)
            g_job.prompt[n++] = (buf[i] < 128) ? (char)buf[i] : ' ';
        g_job.prompt[n] = 0;
        http_srv_unlock();
    }
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
    glog("Garlic Diffusion v1.70 starting (PS4 Base/Pro + SD1.5/SDXL + 1080p + Gallery + Models + Styles)");
    log_console_info();
    gallery_init(OUT_ROOT, PRESETS_ROOT);
    styles_init(STYLES_ROOT);
    model_dl_init(MODELS_ROOT, REPO_CFG);
    g_num_presets = presets_list(g_presets, PRESETS_MAX_ITEMS);
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
    int idle_ticks = 0;
    for (;;) {
        int ui_dirty = 0;
        OrbisPadData pd;
        uint32_t btn = 0;
        if (pad_fd >= 0 && scePadReadState(pad_fd, &pd) == 0) btn = pd.buttons;
        uint32_t pressed = btn & ~prev_btn;
        prev_btn = btn;

        // Reload user presets or rescan models if requested by Web UI
        http_srv_lock();
        int pdirty = g_job.presets_dirty;
        g_job.presets_dirty = 0;
        int do_rescan = g_job.rescan_req;
        http_srv_unlock();
        if (pdirty) g_num_presets = presets_list(g_presets, PRESETS_MAX_ITEMS);
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

        if (g_in_models) {
            int total = g_inst_cnt + g_cat_cnt;
            if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
                g_in_models = 0; g_mdl_del_arm = 0; toast[0] = 0; ui_dirty = 1;
            } else if ((pressed & ORBIS_PAD_BUTTON_DOWN) && total > 0) {
                g_mdl_sel = (g_mdl_sel + 1) % total; g_mdl_del_arm = 0; ui_dirty = 1;
            } else if ((pressed & ORBIS_PAD_BUTTON_UP) && total > 0) {
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
            } else if (pressed) {
                if (pressed & (ORBIS_PAD_BUTTON_CIRCLE | ORBIS_PAD_BUTTON_SQUARE)) {
                    g_in_gallery = 0; g_del_arm = 0; toast[0] = 0; ui_dirty = 1;
                } else if (pressed & ORBIS_PAD_BUTTON_RIGHT) {
                    if (g_gal_count > 0) { g_gal_idx = (g_gal_idx + 1) % g_gal_count; toast[0] = 0; gal_load_current(); ui_dirty = 1; }
                } else if (pressed & ORBIS_PAD_BUTTON_LEFT) {
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
                    g_job.sampler = (strcmp(it->sampler, "LCM") == 0) ? 2 : (strcmp(it->sampler, "Euler") == 0 ? 1 : 0);
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
                    g_job.sampler = (strcmp(it->sampler, "LCM") == 0) ? 2 : (strcmp(it->sampler, "Euler") == 0 ? 1 : 0);
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
        } else {
            if (pressed & ORBIS_PAD_BUTTON_CROSS) { edit_prompt_osk(); ui_dirty = 1; }
            if ((pressed & ORBIS_PAD_BUTTON_RIGHT) && g_num_presets > 0) {
                preset_idx = (preset_idx + 1) % g_num_presets;
                http_srv_lock();
                snprintf(g_job.prompt, sizeof(g_job.prompt), "%s", g_presets[preset_idx].prompt);
                if (g_presets[preset_idx].negative[0])
                    snprintf(g_job.negative, sizeof(g_job.negative), "%s", g_presets[preset_idx].negative);
                http_srv_unlock();
                snprintf(toast, sizeof(toast), "Preset: %s", g_presets[preset_idx].name);
                ui_dirty = 1;
            }
            if ((pressed & ORBIS_PAD_BUTTON_LEFT) && g_num_presets > 0) {
                preset_idx = (preset_idx + g_num_presets - 1) % g_num_presets;
                http_srv_lock();
                snprintf(g_job.prompt, sizeof(g_job.prompt), "%s", g_presets[preset_idx].prompt);
                if (g_presets[preset_idx].negative[0])
                    snprintf(g_job.negative, sizeof(g_job.negative), "%s", g_presets[preset_idx].negative);
                http_srv_unlock();
                snprintf(toast, sizeof(toast), "Preset: %s", g_presets[preset_idx].name);
                ui_dirty = 1;
            }
            if (pressed & ORBIS_PAD_BUTTON_R1) {
                res_idx = (res_idx + 1) % 8;
                http_srv_lock();
                g_job.width = RES_LIST[res_idx][0];
                g_job.height = RES_LIST[res_idx][1];
                http_srv_unlock();
                ui_dirty = 1;
            }
            if (pressed & ORBIS_PAD_BUTTON_L1) {
                res_idx = (res_idx + 7) % 8;
                http_srv_lock();
                g_job.width = RES_LIST[res_idx][0];
                g_job.height = RES_LIST[res_idx][1];
                http_srv_unlock();
                ui_dirty = 1;
            }
            if (pressed & ORBIS_PAD_BUTTON_R2) {
                http_srv_lock();
                for (int i = 0; i < (int)(sizeof(STEP_LIST) / sizeof(STEP_LIST[0])); i++) {
                    if (STEP_LIST[i] > g_job.steps) { g_job.steps = STEP_LIST[i]; break; }
                }
                http_srv_unlock();
                ui_dirty = 1;
            }
            if (pressed & ORBIS_PAD_BUTTON_L2) {
                http_srv_lock();
                for (int i = (int)(sizeof(STEP_LIST) / sizeof(STEP_LIST[0])) - 1; i >= 0; i--) {
                    if (STEP_LIST[i] < g_job.steps) { g_job.steps = STEP_LIST[i]; break; }
                }
                http_srv_unlock();
                ui_dirty = 1;
            }
            if (pressed & ORBIS_PAD_BUTTON_L3) {
                http_srv_lock();
                g_job.upscale = (g_job.upscale + 1) % 3;
                http_srv_unlock();
                ui_dirty = 1;
            }
            if (pressed & ORBIS_PAD_BUTTON_R3) {
                http_srv_lock();
                g_job.eco_mode = (g_job.eco_mode + 1) % 4;
                g_st.eco_mode = g_job.eco_mode;
                http_srv_unlock();
                ui_dirty = 1;
            }
            if (pressed & ORBIS_PAD_BUTTON_OPTIONS) {
                scan_models();
                http_srv_lock();
                if (g_st.num_models > 0) {
                    model_idx = (model_idx + 1) % g_st.num_models;
                    snprintf(g_job.switch_model, sizeof(g_job.switch_model), "%s", g_st.models[model_idx]);
                    g_job.switch_req = 1;
                }
                http_srv_unlock();
                ui_dirty = 1;
            }
            if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) {
                http_srv_lock();
                if (g_st.ready) {
                    g_job.cancel_req = 0;
                    g_job.pending = 1;
                } else {
                    snprintf(toast, sizeof(toast), "No complete model loaded - press [OPTIONS]");
                }
                http_srv_unlock();
                ui_dirty = 1;
            }
            if (pressed & ORBIS_PAD_BUTTON_SQUARE) {
                g_in_gallery = 1; g_gal_full = 0; g_gal_idx = 0; toast[0] = 0;
                gal_refresh();
                ui_dirty = 1;
            }
            if (pressed & ORBIS_PAD_BUTTON_UP) {
                g_in_models = 1; g_mdl_sel = 0; g_mdl_del_arm = 0; toast[0] = 0;
                models_mgr_refresh();
                ui_dirty = 1;
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
                if (gop == 1 || gop == 2) {
                    ok = sdxl_upscale_rgb(&eng, srgb, sw, sh, &drgb, &dw, &dh, gop, geco, on_progress, on_cancel, NULL);
                } else if (gop == 3) {
                    on_progress(NULL, 0.5f, "Resizing image...");
                    drgb = rgb_resize(srgb, sw, sh, gw, gh);
                    dw = gw; dh = gh;
                    ok = drgb ? 0 : -1;
                }
                free(srgb);
                if (ok == 0 && drgb) {
                    size_t nsz = 0;
                    src_meta.width = dw; src_meta.height = dh;
                    if (gop == 1 || gop == 2) src_meta.upscaled = gop;
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
            uint64_t seed = run_job.seed ? run_job.seed : (sceKernelReadTsc() & 0xFFFFFFFFull);
            SdxlParams p = {run_job.prompt, run_job.negative, rw, rh, rsteps, rcfg, seed, run_job.eco_mode, run_job.sampler};
            uint8_t* rgb = (uint8_t*)malloc((size_t)p.width * p.height * 3);
            static const char* SMP_LOG2[3] = {"EulerA", "Euler", "LCM"};
            glog("gen start %s %ux%u steps=%u cfg=%.1f %s %s seed=%llu",
                 eng.is_sd15 ? "SD1.5" : "SDXL", p.width, p.height, p.steps, (double)p.cfg_scale,
                 SMP_LOG2[(p.sampler >= 0 && p.sampler <= 2) ? p.sampler : 0], sdxl_eco_name_ex(p.eco_mode, eng.neo), (unsigned long long)seed);
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
                static const char* SMP_META[3] = {"Euler a", "Euler", "LCM"};
                const char* smp_str = SMP_META[(p.sampler >= 0 && p.sampler <= 2) ? p.sampler : 0];
                int did_up = (out_w != p.width || out_h != p.height) ? run_job.upscale : 0;
                char pmeta[2048];
                format_a1111_params(pmeta, sizeof(pmeta), p.prompt, p.negative, p.steps, smp_str,
                                    p.cfg_scale, seed, out_w, out_h, eng.model_name, did_up);
                size_t png_sz = 0;
                uint8_t* png = png_encode_rgb_meta(rgb, out_w, out_h, pmeta, &png_sz);
                // Auto-save every generated image + metadata into persistent Gallery!
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
                snprintf(g_st.stage, sizeof(g_st.stage), "Done %ux%u in %.1fs (%.2f TF | %s)",
                         out_w, out_h, eng.last_gen_sec, eng.stats.effective_tflops, sdxl_eco_name_ex(p.eco_mode, eng.neo));
                g_st.busy = 0;
                http_srv_unlock();
                glog("gen done %ux%u in %.1fs (%zu bytes PNG)", out_w, out_h, eng.last_gen_sec, png_sz);
            } else {
                free(rgb);
                http_srv_lock();
                if (rc == 1) snprintf(g_st.stage, sizeof(g_st.stage), "Cancelled");
                else snprintf(g_st.error, sizeof(g_st.error), "%s", eng.error[0] ? eng.error : "generation error");
                g_st.busy = 0;
                http_srv_unlock();
                glog("gen ended rc=%d (%s)", rc, eng.error);
            }
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
