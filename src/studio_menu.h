// Garlic Diffusion: TV Studio settings menu (pure drawing code, no PS4 dependencies so it can be
// rendered and checked on a PC).  Navigation model:
//   Up/Down = pick a setting, Left/Right = change it, X = open dropdown / edit / adjust, O = cancel.
#ifndef STUDIO_MENU_H
#define STUDIO_MENU_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "garlic_ui.h"

enum {
    ROW_MODEL = 0, ROW_PRESET, ROW_CHAR, ROW_STYLE, ROW_PROMPT, ROW_NEG, ROW_RES,
    ROW_STEPS, ROW_CFG, ROW_SAMPLER, ROW_UPSCALE, ROW_POWER, ROW_SEED, ROW_GEN, ROW_COUNT
};
enum { K_DROP = 0, K_CHOICE, K_SLIDER, K_TEXT, K_ACTION };

static const uint8_t SM_KIND[ROW_COUNT] = {
    K_DROP, K_DROP, K_CHOICE, K_CHOICE, K_TEXT, K_TEXT, K_CHOICE,
    K_SLIDER, K_SLIDER, K_CHOICE, K_CHOICE, K_CHOICE, K_CHOICE, K_ACTION
};
static const char* const SM_LABEL[ROW_COUNT] = {
    "Model", "Preset", "Character", "Art Style", "Prompt", "Negative", "Resolution",
    "Steps", "CFG Scale", "Sampler", "Upscale", "Power Mode", "Seed", "Generate"
};
static const char* const SM_SAMPLER[4] = { "Euler a", "Euler", "LCM (DMD2 models)", "DPM++ 2M Karras" };
static const char* const SM_UPSCALE[4] = {
    "Off (native resolution)", "1080p HD (Real-ESRGAN)", "2x HD (Real-ESRGAN)", "4K UHD (Real-ESRGAN)"
};

#define SM_PX 56
#define SM_PW 760
#define SM_ROW0 110
#define SM_ROW_H 48
#define SM_DD_VIS 9
#define SM_STEPS_MIN 4
#define SM_STEPS_MAX 40
#define SM_CFG_MIN 1.0f
#define SM_CFG_MAX 15.0f

typedef struct {
    const char* model;
    int neo, busy, ready, has_upscaler;
    uint32_t vram_u, vram_t;
    float prog;
    const char* stage;
    const char* err;
    const char* toast;
    const char* addr;
    const char* preset_name;                      // NULL = none selected
    const char* char_name;                        // NULL = Off
    const char* char_series;
    const char* style_name;                       // NULL = Off
    const char* prompt;
    const char* neg;
    uint32_t w, h, steps;
    float cfg;
    int sampler, upscale, eco;
    uint64_t seed, last_seed;
    double last_sec, last_tf;
    const uint8_t* preview;                       // RGB, preview_w x preview_h
    uint32_t pw, ph;
    int focus, adjust;
    int dd_open, dd_row, dd_sel, dd_top, dd_n, dd_cur;
    void (*dd_label)(void* ctx, int row, int i, char* out, size_t cap);
    void* dd_ctx;
} StudioView;

static const char* sm_power_note(int eco) {
    switch (eco) {
        case 3: return "coolest, quietest";
        case 2: return "default, calm fan";
        case 1: return "faster, warmer";
        default: return "fastest, LOUD";
    }
}

static void sm_power_label(int eco, int neo, char* out, size_t cap) {
    static const char* const pro[4]  = { "Turbo (36CU)", "Balanced (25CU)", "Quiet (20CU)", "Silent (14CU)" };
    static const char* const base[4] = { "Turbo (18CU)", "Balanced (14CU)", "Quiet (11CU)", "Silent (7CU)" };
    if (eco < 0 || eco > 3) eco = 2;
    snprintf(out, cap, "%s  ·  %s", (neo ? pro : base)[eco], sm_power_note(eco));
}

static void sm_res_label(uint32_t w, uint32_t h, char* out, size_t cap) {
    const char* shape = (w == h) ? "square" : (w < h ? "portrait" : "landscape");
    snprintf(out, cap, "%u x %u   ·   %.2f MP %s", w, h, (double)w * (double)h / 1e6, shape);
}

// Cut `s` so it fits in max_w pixels, adding "..." when truncated.
static void sm_fit(const GFont* f, const char* s, int max_w, char* out, size_t cap) {
    if (!s) s = "";
    size_t n = strlen(s);
    if (n > 220) n = 220;
    if (n > cap - 5) n = cap - 5;
    memcpy(out, s, n);
    out[n] = 0;
    if (g_str_width(f, out) <= max_w) return;
    int dots = g_str_width(f, "...");
    while (n > 0) {
        n--;
        while (n > 0 && (((uint8_t)out[n]) & 0xC0) == 0x80) n--;
        out[n] = 0;
        if (g_str_width(f, out) + dots <= max_w) break;
    }
    strcat(out, "...");
}

static void sm_chev(uint32_t* back, int cx, int cy, int dir, uint32_t col) {
    const float s = 6.0f;
    if (dir != 0) {
        float d = (float)dir * 3.0f;
        g_line(back, (float)cx - d, (float)cy - s, (float)cx + d, (float)cy, 2.6f, col, 255);
        g_line(back, (float)cx + d, (float)cy, (float)cx - d, (float)cy + s, 2.6f, col, 255);
    } else {
        g_line(back, (float)cx - s, (float)cy - 3.0f, (float)cx, (float)cy + 3.0f, 2.6f, col, 255);
        g_line(back, (float)cx, (float)cy + 3.0f, (float)cx + s, (float)cy - 3.0f, 2.6f, col, 255);
    }
}

static void sm_row_value(const StudioView* v, int row, char* out, size_t cap, float* frac) {
    char tmp[256];
    *frac = 0.0f;
    switch (row) {
    case ROW_MODEL:
        snprintf(out, cap, "%s", (v->model && v->model[0]) ? v->model : "No model loaded - press X");
        break;
    case ROW_PRESET:
        snprintf(out, cap, "%s", v->preset_name ? v->preset_name : "Choose a saved prompt...");
        break;
    case ROW_CHAR:
        if (v->char_name) snprintf(out, cap, "%s  (%s)", v->char_name, v->char_series ? v->char_series : "");
        else snprintf(out, cap, "Off");
        break;
    case ROW_STYLE:
        snprintf(out, cap, "%s", v->style_name ? v->style_name : "Off");
        break;
    case ROW_PROMPT:
        sm_fit(&g_fsmall, v->prompt, 470, out, cap);
        break;
    case ROW_NEG:
        sm_fit(&g_fsmall, v->neg, 470, out, cap);
        break;
    case ROW_RES:
        sm_res_label(v->w, v->h, tmp, sizeof(tmp));
        snprintf(out, cap, "%s", tmp);
        break;
    case ROW_STEPS:
        snprintf(out, cap, "%u", v->steps);
        *frac = (float)v->steps / (float)SM_STEPS_MAX;
        break;
    case ROW_CFG:
        snprintf(out, cap, "%.1f", (double)v->cfg);
        *frac = (v->cfg - SM_CFG_MIN) / (SM_CFG_MAX - SM_CFG_MIN);
        break;
    case ROW_SAMPLER:
        snprintf(out, cap, "%s", SM_SAMPLER[(v->sampler >= 0 && v->sampler <= 3) ? v->sampler : 0]);
        break;
    case ROW_UPSCALE:
        snprintf(out, cap, "%s", SM_UPSCALE[(v->upscale >= 0 && v->upscale <= 3) ? v->upscale : 0]);
        break;
    case ROW_POWER:
        sm_power_label(v->eco, v->neo, out, cap);
        break;
    case ROW_SEED:
        if (v->seed) snprintf(out, cap, "Locked  ·  %llu", (unsigned long long)v->seed);
        else if (v->last_seed) snprintf(out, cap, "Random each time  ·  last %llu", (unsigned long long)v->last_seed);
        else snprintf(out, cap, "Random each time");
        break;
    default:
        out[0] = 0;
        break;
    }
}

static void sm_draw_row(uint32_t* back, const StudioView* v, int row) {
    const int x = SM_PX, w = SM_PW, h = SM_ROW_H - 6;
    const int y = SM_ROW0 + row * SM_ROW_H;
    const int kind = SM_KIND[row];
    const int foc = (v->focus == row);
    const int adj = foc && v->adjust && kind == K_SLIDER;
    const int ty = y + (h - g_fsmall.line) / 2;
    const int cy = y + h / 2;
    char val[300];
    float frac = 0.0f;
    sm_row_value(v, row, val, sizeof(val), &frac);

    if (kind == K_ACTION) {
        uint32_t fill = !v->ready ? 0x1B3156 : (v->busy ? 0x1B3156 : C_ACCENT);
        g_rrect(back, x, y, w, h, 14, fill, foc ? 255 : 215);
        g_rrect_line(back, x, y, w, h, 14, foc ? 2 : 1, foc ? 0xFFFFFF : C_LINE_HI, 255);
        const char* t = v->busy ? "Generating...  press O to cancel" : (v->ready ? "Generate Image" : "Load a model first");
        int tw = g_str_width(&g_fbold, t) + (v->ready && !v->busy ? 44 : 0);
        int tx = x + (w - tw) / 2;
        if (v->ready && !v->busy) tx += g_draw_btn(back, GBTN_TRIANGLE, tx, cy) + 12;
        g_draw_str(back, &g_fbold, tx, cy - g_fbold.line / 2, t, 0xFFFFFF);
        return;
    }

    uint32_t line = foc ? (adj ? C_GREEN : C_ACCENT_HI) : C_LINE;
    g_rrect(back, x, y, w, h, 14, foc ? C_SURFACE_HI : C_SURFACE, foc ? 255 : 205);
    g_rrect_line(back, x, y, w, h, 14, foc ? 2 : 1, line, 255);
    if (foc) g_rrect(back, x + 8, y + 10, 4, h - 20, 2, adj ? C_GREEN : C_ACCENT_HI, 255);

    g_draw_str(back, &g_fsmall, x + 26, ty, SM_LABEL[row], foc ? 0xFFFFFF : C_TEXT_DIM);

    const int vx = x + 236;
    const int vw = w - 236 - 22;
    uint32_t vcol = foc ? 0xFFFFFF : C_TEXT;
    if (kind == K_SLIDER) {
        const int tw = vw - 96;
        g_rrect(back, vx, cy - 4, tw, 8, 4, 0x1B3156, 255);
        int fw = (int)((float)tw * g_clamp01(frac));
        if (fw < 8) fw = 8;
        g_rrect(back, vx, cy - 4, fw, 8, 4, adj ? C_GREEN : C_ACCENT_HI, 255);
        g_circle(back, (float)(vx + fw), (float)cy, adj ? 12.0f : 9.0f, 0.0f, 0xFFFFFF, 255);
        int nw = g_str_width(&g_fbold, val);
        g_draw_str(back, &g_fbold, x + w - 24 - nw, cy - g_fbold.line / 2, val, adj ? C_GREEN : vcol);
        if (foc) {
            sm_chev(back, vx - 16, cy, -1, adj ? C_GREEN : C_ACCENT_HI);
            sm_chev(back, vx + tw + 18, cy, +1, adj ? C_GREEN : C_ACCENT_HI);
        }
        return;
    }

    char fit[300];
    int tx = vx;
    if (foc && (kind == K_CHOICE)) {
        sm_chev(back, vx - 6, cy, -1, C_ACCENT_HI);
        sm_chev(back, x + w - 28, cy, +1, C_ACCENT_HI);
        tx = vx + 14;
    } else if (foc && kind == K_DROP) {
        sm_chev(back, x + w - 30, cy, 0, C_ACCENT_HI);
    }
    int max_w = vw - (kind == K_TEXT ? 0 : 36);
    sm_fit(&g_fsmall, val, max_w, fit, sizeof(fit));
    int is_off = (!strcmp(val, "Off") || !strncmp(val, "Random each", 11));
    g_draw_str(back, &g_fsmall, tx, ty, fit, is_off && !foc ? C_TEXT_MUTED : vcol);
}

static const char* sm_hint(const StudioView* v, int row, char* buf, size_t cap) {
    switch (row) {
    case ROW_MODEL:   return "Checkpoint on the GPU. Press X to pick another (loads in a few seconds). OPTIONS opens the model downloader.";
    case ROW_PRESET:  return "Saved prompt presets. Press X to choose one.";
    case ROW_CHAR:    return "Adds a character's tags in front of your prompt. Left/Right cycles, X shows the full list.";
    case ROW_STYLE:   return "Adds an art style (cel shading, Shinkai, ...) after your prompt. Left/Right cycles, X shows the list.";
    case ROW_PROMPT:  return v->prompt && v->prompt[0] ? v->prompt : "(empty) Press X to type a prompt.";
    case ROW_NEG:     return v->neg && v->neg[0] ? v->neg : "(empty) Press X to type a negative prompt.";
    case ROW_RES:     return "Bigger = more detail in eyes and hands but slower. SDXL is native at 832x1216 / 1024x1024, SD 1.5 at 512x768.";
    case ROW_STEPS:   return "More steps = more detail but slower. Distilled DMD2 / 8-step models: 6-8. Full models: 20-28.";
    case ROW_CFG:     return "Prompt strength. At 1.0 the negative prompt is IGNORED (fastest). DMD2 models: try 1.4 to enable it. Others: 3.5-7.";
    case ROW_SAMPLER: return "DPM++ 2M Karras: smooth 2nd-order sampler for 20+ step models. LCM is for DMD2 models only.";
    case ROW_UPSCALE: return "Real-ESRGAN Anime v3 runs on the GPU after rendering: 1080p, 2x or 4K UHD.";
    case ROW_POWER:   return "GPU compute units and fan noise. Quiet is the default; avoid Turbo for long sessions.";
    case ROW_SEED:    return "Random each time, or lock the last used seed to re-render the same composition.";
    case ROW_GEN:
        if (v->last_sec > 0.0) {
            snprintf(buf, cap, "Press X or Triangle to generate.   Last render: %.1fs  ·  %.2f TFLOPS", v->last_sec, v->last_tf);
            return buf;
        }
        return "Press X or Triangle to generate.";
    }
    return "";
}

static void sm_blit_fit(uint32_t* back, const uint8_t* rgb, uint32_t rw, uint32_t rh, int px, int py, int pw, int ph) {
    float sx = (float)pw / (float)rw, sy = (float)ph / (float)rh;
    float sc = sx < sy ? sx : sy;
    int dw = (int)((float)rw * sc), dh = (int)((float)rh * sc);
    int ox = px + (pw - dw) / 2, oy = py + (ph - dh) / 2;
    for (int y = 0; y < dh; y++) {
        int src_y = (int)((float)y / sc);
        if (src_y >= (int)rh) src_y = (int)rh - 1;
        uint32_t* dst = back + (size_t)(oy + y) * 1920 + ox;
        const uint8_t* srow = rgb + (size_t)src_y * rw * 3;
        for (int x = 0; x < dw; x++) {
            int src_x = (int)((float)x / sc);
            if (src_x >= (int)rw) src_x = (int)rw - 1;
            const uint8_t* p = srow + src_x * 3;
            dst[x] = 0x80000000u | ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2];
        }
    }
}

static void sm_draw_dropdown(uint32_t* back, const StudioView* v) {
    g_fill(back, 0, 0, 1920, 1080, 0x020611, 178);
    const int vis = v->dd_n < SM_DD_VIS ? v->dd_n : SM_DD_VIS;
    const int pw = 820, ph = 82 + vis * 52 + 18;
    const int px = (1920 - pw) / 2, py = (1080 - ph) / 2 - 16;
    g_rrect(back, px, py, pw, ph, 22, C_PANEL, 252);
    g_rrect_line(back, px, py, pw, ph, 22, 2, C_ACCENT_HI, 255);
    g_draw_str(back, &g_fbold, px + 32, py + 22, SM_LABEL[v->dd_row], 0xFFFFFF);
    char cnt[32];
    snprintf(cnt, sizeof(cnt), "%d / %d", v->dd_sel + 1, v->dd_n);
    g_draw_str(back, &g_fsmall, px + pw - 32 - g_str_width(&g_fsmall, cnt), py + 26, cnt, C_TEXT_MUTED);

    for (int k = 0; k < vis; k++) {
        int i = v->dd_top + k;
        if (i >= v->dd_n) break;
        int iy = py + 70 + k * 52;
        int sel = (i == v->dd_sel);
        g_rrect(back, px + 18, iy, pw - 52, 46, 12, sel ? C_SURFACE_HI : C_SURFACE, sel ? 255 : 150);
        if (sel) g_rrect_line(back, px + 18, iy, pw - 52, 46, 12, 2, C_ACCENT_HI, 255);
        if (i == v->dd_cur) g_circle(back, (float)(px + 44), (float)(iy + 23), 6.0f, 0.0f, C_GREEN, 255);
        char lab[160], fit[160];
        lab[0] = 0;
        if (v->dd_label) v->dd_label(v->dd_ctx, v->dd_row, i, lab, sizeof(lab));
        sm_fit(&g_fsmall, lab, pw - 52 - 80, fit, sizeof(fit));
        g_draw_str(back, &g_fsmall, px + 66, iy + (46 - g_fsmall.line) / 2, fit, sel ? 0xFFFFFF : C_TEXT);
    }
    if (v->dd_n > vis) {
        int tx = px + pw - 22, ty0 = py + 70, th = vis * 52 - 6;
        g_rrect(back, tx, ty0, 6, th, 3, 0x1B3156, 255);
        int thumb = th * vis / v->dd_n;
        if (thumb < 24) thumb = 24;
        int tpos = (th - thumb) * v->dd_top / (v->dd_n - vis);
        g_rrect(back, tx, ty0 + tpos, 6, thumb, 3, C_ACCENT_HI, 255);
    }
}

static void sm_draw_studio(uint32_t* back, const StudioView* v) {
    g_draw_bg(back);
    g_draw_header(back, v->neo ? "PS4 Pro Neo" : "PS4 Base",
                  (v->model && v->model[0]) ? v->model : "No Model", v->busy, v->vram_u, v->vram_t, v->addr);

    for (int r = 0; r < ROW_COUNT; r++) sm_draw_row(back, v, r);

    // Info box: progress + status / help text for the focused setting
    const int ix = SM_PX, iy = SM_ROW0 + ROW_COUNT * SM_ROW_H + 10, iw = SM_PW, ih = 980 - iy;
    g_rrect(back, ix, iy, iw, ih, 16, C_PANEL, 235);
    g_rrect_line(back, ix, iy, iw, ih, 16, 1, C_LINE_HI, 255);
    g_rrect(back, ix + 20, iy + 16, iw - 40, 10, 5, 0x1B3156, 255);
    int pfw = (int)((float)(iw - 40) * g_clamp01(v->prog));
    if (pfw > 0) g_rrect(back, ix + 20, iy + 16, pfw < 10 ? 10 : pfw, 10, 5, v->busy ? C_ACCENT_HI : C_GREEN, 255);
    char hbuf[200];
    const char* txt;
    uint32_t tcol = C_TEXT_DIM;
    if (v->err && v->err[0]) { txt = v->err; tcol = C_RED; }
    else if (v->busy) { txt = v->stage; tcol = C_TEXT; }
    else if (v->toast && v->toast[0]) { txt = v->toast; tcol = C_GREEN; }
    else {
        txt = sm_hint(v, v->focus, hbuf, sizeof(hbuf));
        if (v->focus == ROW_PROMPT || v->focus == ROW_NEG) tcol = C_TEXT;
    }
    g_draw_wrapped(back, &g_fsmall, ix + 20, iy + 38, iw - 40, 28, tcol, txt ? txt : "", 5);

    // Artwork panel
    const int px = SM_PX + SM_PW + 24, py = 112, pw = 1920 - 56 - px, ph = 868;
    g_rrect(back, px, py, pw, ph, 20, C_PANEL, 240);
    g_rrect_line(back, px, py, pw, ph, 20, 1, C_LINE_HI, 255);
    if (v->preview && v->pw > 0 && v->ph > 0) {
        sm_blit_fit(back, v->preview, v->pw, v->ph, px + 14, py + 14, pw - 28, ph - 28);
    } else {
        int ecx = px + pw / 2, ecy = py + ph / 2 - 40;
        g_circle(back, (float)ecx, (float)ecy, 88.0f, 0.0f, C_ACCENT, 38);
        g_circle(back, (float)ecx, (float)ecy, 64.0f, 0.0f, C_ACCENT_HI, 28);
        g_draw_logo48(back, ecx - 24, ecy - 24);
        int w1 = g_str_width(&g_fbig, "Garlic"), w2 = g_str_width(&g_fbig, "Diffusion");
        int tx = ecx - (w1 + g_fbig.space + w2) / 2;
        tx = g_draw_str(back, &g_fbig, tx, ecy + 60, "Garlic", 0xFFFFFF);
        g_draw_str(back, &g_fbig, tx + g_fbig.space, ecy + 60, "Diffusion", C_ACCENT_HI);
        const char* s1 = "Native Stable Diffusion 1.5, SDXL & Real-ESRGAN on PS4 GPU. Offline.";
        const char* s2 = "Pick your settings on the left, then press Triangle to generate.";
        g_draw_str(back, &g_fsmall, ecx - g_str_width(&g_fsmall, s1) / 2, ecy + 114, s1, C_TEXT);
        g_draw_str(back, &g_fsmall, ecx - g_str_width(&g_fsmall, s2) / 2, ecy + 144, s2, C_TEXT_MUTED);
    }

    // Footer hints depend on the current mode
    const char* right = "ko-fi.com/djtomato  ·  discord.gg/Gbyq3kpGjc";
    if (v->dd_open) {
        sm_draw_dropdown(back, v);
        const GHint h[] = { { GBTN_UPDOWN, "Choose" }, { GBTN_L1, "Page up" }, { GBTN_R1, "Page down" },
                            { GBTN_CROSS, "Select" }, { GBTN_CIRCLE, "Cancel" } };
        g_draw_footer(back, h, 5, "");
    } else if (v->adjust) {
        const GHint h[] = { { GBTN_LEFTRIGHT, "Adjust" }, { GBTN_CROSS, "Confirm" }, { GBTN_CIRCLE, "Revert" } };
        g_draw_footer(back, h, 3, "");
    } else if (v->busy) {
        const GHint h[] = { { GBTN_CIRCLE, "Cancel" } };
        g_draw_footer(back, h, 1, right);
    } else {
        const GHint h[] = { { GBTN_UPDOWN, "Select" }, { GBTN_LEFTRIGHT, "Change" }, { GBTN_CROSS, "Open / Edit" },
                            { GBTN_TRIANGLE, "Generate" }, { GBTN_SQUARE, "Gallery" },
                            { GBTN_OPTIONS, "Models" }, { GBTN_CIRCLE, "Credits" } };
        g_draw_footer(back, h, 7, right);
    }
}

#endif
