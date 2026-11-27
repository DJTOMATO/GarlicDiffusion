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
    ROW_MODEL = 0, ROW_LORA, ROW_PRESET, ROW_CHAR, ROW_STYLE, ROW_PROMPT, ROW_NEG, ROW_RES,
    ROW_STEPS, ROW_CFG, ROW_SAMPLER, ROW_UPSCALE, ROW_POWER, ROW_SEED, ROW_BATCH, ROW_GEN, ROW_CREDITS, ROW_COUNT
};
enum {
    EROW_CHAR = 0, EROW_SCENE = 1, EROW_FRAMING = 2, EROW_QUALITY = 3, EROW_MODEL = 4, EROW_GEN = 5, EROW_CREDITS = 6, EROW_COUNT = 7
};
enum { K_DROP = 0, K_CHOICE, K_SLIDER, K_TEXT, K_ACTION };

static const uint8_t SM_KIND[ROW_COUNT] = {
    K_DROP, K_CHOICE, K_DROP, K_CHOICE, K_CHOICE, K_TEXT, K_TEXT, K_CHOICE,
    K_SLIDER, K_SLIDER, K_CHOICE, K_CHOICE, K_CHOICE, K_CHOICE, K_CHOICE, K_ACTION, K_DROP
};
static const char* const SM_LABEL[ROW_COUNT] = {
    "Model", "LoRA", "Preset", "Character", "Art Style", "Prompt", "Negative", "Resolution",
    "Steps", "CFG Scale", "Sampler", "Upscale", "Power Mode", "Seed", "Batch", "Generate", "Credits & Updates"
};
static const char* const SM_EASY_LABEL[EROW_COUNT] = {
    "1. Character", "2. Scene / Preset", "3. Framing", "4. Speed / Quality", "5. Checkpoint", "Generate", "Credits & Updates"
};
static const char* const SM_EASY_FRAMING[4] = {
    "Portrait (Close-Up Face & Eyes)",
    "Upper Body (Cowboy Shot & Hands)",
    "Full Body (Head to Toe Standing)",
    "Wide 16:9 (Cinematic Wallpaper)"
};
static const char* const SM_EASY_QUALITY[3] = {
    "Fast Mode  ·  Quickest render (Speed Profile)",
    "Best Quality  ·  Model Sweet-Spot + Anatomy Shield",
    "Best + 1080p HD  ·  Sweet-Spot + Real-ESRGAN Upscale"
};
static const char* const SM_SAMPLER[4] = { "Euler a", "Euler", "LCM (DMD2 models)", "DPM++ 2M Karras" };
static const char* const SM_UPSCALE[4] = {
    "Off (native resolution)", "1080p HD (Real-ESRGAN)", "2x HD (Real-ESRGAN)", "4K UHD (Real-ESRGAN)"
};

#define SM_PX 56
#define SM_PW 760
#define SM_ROW0 106
#define SM_ROW_H 40
#define SM_EASY_ROW_H 76
#define SM_DD_VIS 9
#define SM_STEPS_MIN 4
#define SM_STEPS_MAX 40
#define SM_CFG_MIN 1.0f
#define SM_CFG_MAX 15.0f

typedef struct {
    const char* model;
    int neo, is_sd15, busy, ready, has_upscaler;
    int ui_mode;                                  // 0 = Easy Mode, 1 = Advanced Studio
    int easy_framing;                             // 0=Portrait, 1=Upper, 2=Full Body, 3=Wide 16:9
    int easy_quality;                             // 0=Fast, 1=Best Quality, 2=Best + 1080p HD
    const char* update_ver;                       // non-empty if a newer GitHub version is available
    uint32_t vram_u, vram_t;
    float prog;
    const char* stage;
    const char* err;
    const char* toast;
    const char* addr;
    const char* lora_name;                        // NULL or "" = Off
    float lora_scale;
    const char* preset_name;                      // NULL = none selected
    const char* char_name;                        // NULL = Off
    const char* char_series;
    const char* style_name;                       // NULL = Off
    const char* prompt;
    const char* neg;
    uint32_t w, h, steps;
    float cfg;
    int sampler, upscale, eco, api_eco;
    uint32_t batch, rand_mask;
    int shutdown_after, nsfw_allow, auto_quality, auto_face_fix;
    const char* batch_folder;
    uint64_t seed, last_seed;
    double last_sec, last_tf;
    float eta_sec, elapsed_sec;
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

static const char* sm_api_eco_short(int api_eco) {
    switch (api_eco) {
        case 3: return "Silent (whisper quiet)";
        case 2: return "Quiet";
        case 1: return "Balanced";
        case 0: return "Turbo";
        default: return "Follow Studio";
    }
}

static void sm_power_label(int eco, int neo, char* out, size_t cap) {
    static const char* const pro[4]  = { "Turbo (36CU)", "Balanced (25CU)", "Quiet (16CU)", "Silent (12CU)" };
    static const char* const base[4] = { "Turbo (18CU)", "Balanced (13CU)", "Quiet (8CU)", "Silent (6CU)" };
    if (eco < 0 || eco > 3) eco = 2;
    snprintf(out, cap, "%s  ·  %s", (neo ? pro : base)[eco], sm_power_note(eco));
}

static void sm_fmt_eta(float sec, char* out, size_t cap) {
    if (sec <= 0.4f) { out[0] = 0; return; }
    int s = (int)(sec + 0.5f);
    if (s < 60) snprintf(out, cap, "ETA ~%ds", s);
    else if (s < 3600) snprintf(out, cap, "ETA ~%dm %02ds", s / 60, s % 60);
    else snprintf(out, cap, "ETA ~%dh %02dm", s / 3600, (s % 3600) / 60);
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
    case ROW_LORA:
        if (v->lora_name && v->lora_name[0])
            snprintf(out, cap, "%s   ·   %.2f", v->lora_name, (double)v->lora_scale);
        else
            snprintf(out, cap, "Off");
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
    case ROW_POWER: {
        char pl[96];
        sm_power_label(v->eco, v->neo, pl, sizeof(pl));
        snprintf(out, cap, "%s  ·  API: %s", pl, sm_api_eco_short(v->api_eco));
        break;
    }
    case ROW_SEED:
        if (v->seed) snprintf(out, cap, "Locked  ·  %llu", (unsigned long long)v->seed);
        else if (v->last_seed) snprintf(out, cap, "Random each time  ·  last %llu", (unsigned long long)v->last_seed);
        else snprintf(out, cap, "Random each time");
        break;
    case ROW_BATCH: {
        char flags[192] = {0};
        size_t fp = 0;
        if (v->rand_mask) {
            char cats[96] = {0};
            size_t cp = 0;
            static const char* const CN[9] = {"Pose", "Expr", "Scene", "Light", "Outfit", "Detail", "Cam", "Ecchi", "Cog"};
            for (int b = 0; b < 9; b++) {
                if (v->rand_mask & (1u << b))
                    cp += (size_t)snprintf(cats + cp, sizeof(cats) - cp, "%s%s", cp ? "+" : "", CN[b]);
            }
            fp += (size_t)snprintf(flags + fp, sizeof(flags) - fp, "  ·  🎲 %s", cats[0] ? cats : "ON");
        }
        if (v->batch_folder && v->batch_folder[0])
            fp += (size_t)snprintf(flags + fp, sizeof(flags) - fp, "  ·  /%s", v->batch_folder);
        if (v->shutdown_after) fp += (size_t)snprintf(flags + fp, sizeof(flags) - fp, "  ·  SHUTDOWN");
        if (v->batch <= 1)
            snprintf(out, cap, "1 image%s", flags[0] ? flags : "  ·  single (X = options)");
        else
            snprintf(out, cap, "%u images%s", v->batch, flags[0] ? flags : "  ·  random seed each");
        break;
    }
    case ROW_CREDITS:
        if (v->update_ver && v->update_ver[0])
            snprintf(out, cap, "Update v%s Available!  ·  Press X or L3", v->update_ver);
        else
            snprintf(out, cap, "Licenses, OTA Updater & Content Filter  (Press X or L3)");
        break;
    default:
        out[0] = 0;
        break;
    }
}

static void sm_easy_row_value(const StudioView* v, int erow, char* out, size_t cap) {
    switch (erow) {
    case EROW_CHAR:
        if (v->char_name && v->char_name[0])
            snprintf(out, cap, "%s   (%s)", v->char_name, (v->char_series && v->char_series[0]) ? v->char_series : "Character");
        else
            snprintf(out, cap, "Any / Custom Character   (Left/Right or X to pick)");
        break;
    case EROW_SCENE:
        if (v->preset_name && v->preset_name[0])
            snprintf(out, cap, "%s", v->preset_name);
        else
            snprintf(out, cap, "Default / Custom Scene   (X to pick, L2 Surprise, R2 Type)");
        break;
    case EROW_FRAMING:
        snprintf(out, cap, "%s", SM_EASY_FRAMING[(v->easy_framing >= 0 && v->easy_framing <= 3) ? v->easy_framing : 2]);
        break;
    case EROW_QUALITY:
        snprintf(out, cap, "%s", SM_EASY_QUALITY[(v->easy_quality >= 0 && v->easy_quality <= 2) ? v->easy_quality : 1]);
        break;
    case EROW_MODEL:
        snprintf(out, cap, "%s   ·   %ux%u  ·  %u stp  ·  CFG %.1f",
                 (v->model && v->model[0]) ? v->model : "No model loaded - press X",
                 v->w, v->h, v->steps, (double)v->cfg);
        break;
    case EROW_CREDITS:
        if (v->update_ver && v->update_ver[0])
            snprintf(out, cap, "Update v%s Ready on GitHub!  ·  Press X or L3", v->update_ver);
        else
            snprintf(out, cap, "Credits, Licenses, OTA Updates & Content Filter  (X / L3)");
        break;
    default:
        out[0] = 0;
        break;
    }
}

static void sm_draw_easy_panel(uint32_t* back, const StudioView* v, int* out_next_y) {
    const int px = SM_PX, pw = SM_PW;
    int y = SM_ROW0;

    // Web-style Easy Studio outer card header
    g_rrect(back, px, y, pw, 668, 18, 0x0A162D, 240);
    g_rrect_line(back, px, y, pw, 668, 18, 1, 0x1E3866, 255);
    g_draw_str(back, &g_fsmall, px + 20, y + 12, "Easy Studio", C_GREEN);
    const char* sw = "R1: Advanced Studio ->";
    g_draw_str(back, &g_fsmall, px + pw - 20 - g_str_width(&g_fsmall, sw), y + 12, sw, C_ACCENT_HI);

    const int ix = px + 18, iw = pw - 36;
    y += 46;

    // Helper for compact input-box rows (1. Character, 2. Scenario, 5. Checkpoint)
    static const char* const FR_PILLS[4] = { "Portrait", "Upper Body", "Full Body", "Wide 16:9" };
    static const char* const QL_PILLS_XL[3] = { "Testing Draft", "Best Quality", "Best + 1080p" };
    static const char* const QL_PILLS_15[2] = { "Testing Draft", "Best + 1080p" };

    // Row 0: 1. Character
    {
        int foc = (v->focus == EROW_CHAR);
        g_draw_str(back, &g_fsmall, ix + 2, y, "1. Character  (Left/Right or X to browse)", foc ? 0xFFFFFF : C_TEXT_DIM);
        y += 32;
        g_rrect(back, ix, y, iw, 38, 10, foc ? C_SURFACE_HI : 0x071226, 250);
        g_rrect_line(back, ix, y, iw, 38, 10, foc ? 2 : 1, foc ? C_ACCENT_HI : C_LINE, 255);
        char val[240], fit[240];
        sm_easy_row_value(v, EROW_CHAR, val, sizeof(val));
        sm_fit(&g_fsmall, val, iw - 68, fit, sizeof(fit));
        g_draw_str(back, &g_fsmall, ix + 14, y + (38 - g_fsmall.line) / 2, fit, foc ? 0xFFFFFF : C_TEXT);
        if (foc) { sm_chev(back, ix + iw - 36, y + 19, -1, C_ACCENT_HI); sm_chev(back, ix + iw - 16, y + 19, +1, C_ACCENT_HI); }
        y += 50;
    }

    // Row 1: 2. Scenario / Action
    {
        int foc = (v->focus == EROW_SCENE);
        g_draw_str(back, &g_fsmall, ix + 2, y, "2. Scenario / Action  (X Pick · L2 Surprise · R2 Type)", foc ? 0xFFFFFF : C_TEXT_DIM);
        y += 32;
        g_rrect(back, ix, y, iw, 38, 10, foc ? C_SURFACE_HI : 0x071226, 250);
        g_rrect_line(back, ix, y, iw, 38, 10, foc ? 2 : 1, foc ? C_ACCENT_HI : C_LINE, 255);
        char val[240], fit[240];
        sm_easy_row_value(v, EROW_SCENE, val, sizeof(val));
        sm_fit(&g_fsmall, val, iw - 68, fit, sizeof(fit));
        g_draw_str(back, &g_fsmall, ix + 14, y + (38 - g_fsmall.line) / 2, fit, foc ? 0xFFFFFF : C_TEXT);
        if (foc) { sm_chev(back, ix + iw - 36, y + 19, -1, C_ACCENT_HI); sm_chev(back, ix + iw - 16, y + 19, +1, C_ACCENT_HI); }
        y += 50;
    }

    // Row 2: 3. Camera Framing (segmented web-style pills!)
    {
        int foc = (v->focus == EROW_FRAMING);
        int n_fr = v->is_sd15 ? 3 : 4;
        int cur_f = (v->easy_framing >= 0 && v->easy_framing < n_fr) ? v->easy_framing : 1;
        g_draw_str(back, &g_fsmall, ix + 2, y, "3. Camera Framing  (Left/Right)", foc ? 0xFFFFFF : C_TEXT_DIM);
        y += 32;
        int gap = 10;
        int bw = (iw - gap * (n_fr - 1)) / n_fr;
        for (int i = 0; i < n_fr; i++) {
            int bx = ix + i * (bw + gap);
            int act = (i == cur_f);
            uint32_t bg = act ? C_ACCENT : (foc ? 0x132648 : 0x0A1830);
            uint32_t bd = (act && foc) ? 0xFFFFFF : (act ? C_ACCENT_HI : (foc ? C_LINE_HI : C_LINE));
            g_rrect(back, bx, y, bw, 36, 10, bg, 250);
            g_rrect_line(back, bx, y, bw, 36, 10, (act || foc) ? 2 : 1, bd, 255);
            int tw = g_str_width(&g_fsmall, FR_PILLS[i]);
            g_draw_str(back, &g_fsmall, bx + (bw - tw) / 2, y + (36 - g_fsmall.line) / 2, FR_PILLS[i], act ? 0xFFFFFF : C_TEXT_DIM);
        }
        y += 48;
    }

    // Row 3: 4. Speed vs Quality (segmented web-style pills!)
    {
        int foc = (v->focus == EROW_QUALITY);
        int n_q = v->is_sd15 ? 2 : 3;
        int cur_q = (v->easy_quality >= 0 && v->easy_quality < n_q) ? v->easy_quality : 1;
        g_draw_str(back, &g_fsmall, ix + 2, y, "4. Speed vs Quality  (Left/Right)", foc ? 0xFFFFFF : C_TEXT_DIM);
        y += 32;
        int gap = 10;
        int bw = (iw - gap * (n_q - 1)) / n_q;
        for (int i = 0; i < n_q; i++) {
            int bx = ix + i * (bw + gap);
            int act = (i == cur_q);
            const char* lbl = v->is_sd15 ? QL_PILLS_15[i] : QL_PILLS_XL[i];
            uint32_t bg = act ? (i == 0 ? 0x1D5E46 : C_ACCENT) : (foc ? 0x132648 : 0x0A1830);
            uint32_t bd = (act && foc) ? 0xFFFFFF : (act ? (i == 0 ? C_GREEN : C_ACCENT_HI) : (foc ? C_LINE_HI : C_LINE));
            g_rrect(back, bx, y, bw, 36, 10, bg, 250);
            g_rrect_line(back, bx, y, bw, 36, 10, (act || foc) ? 2 : 1, bd, 255);
            int tw = g_str_width(&g_fsmall, lbl);
            g_draw_str(back, &g_fsmall, bx + (bw - tw) / 2, y + (36 - g_fsmall.line) / 2, lbl, act ? 0xFFFFFF : C_TEXT_DIM);
        }
        y += 48;
    }

    // Row 4: 5. Checkpoint Model
    {
        int foc = (v->focus == EROW_MODEL);
        g_draw_str(back, &g_fsmall, ix + 2, y, "5. Checkpoint Model  (X to switch / OPTIONS to download)", foc ? 0xFFFFFF : C_TEXT_DIM);
        y += 32;
        g_rrect(back, ix, y, iw, 38, 10, foc ? C_SURFACE_HI : 0x071226, 250);
        g_rrect_line(back, ix, y, iw, 38, 10, foc ? 2 : 1, foc ? C_ACCENT_HI : C_LINE, 255);
        char val[240], fit[240];
        snprintf(val, sizeof(val), "%s", (v->model && v->model[0]) ? v->model : "No model loaded - press X");
        sm_fit(&g_fsmall, val, iw - 52, fit, sizeof(fit));
        g_draw_str(back, &g_fsmall, ix + 14, y + (38 - g_fsmall.line) / 2, fit, foc ? 0xFFFFFF : C_TEXT);
        if (foc) sm_chev(back, ix + iw - 20, y + 19, 0, C_ACCENT_HI);
        y += 50;
    }

    // Auto-Tuned 2-Row Summary Box (fits every setting cleanly with zero truncation!)
    {
        static const char* const SMP_SHORT[4] = { "Euler a", "Euler", "LCM", "DPM++ 2M Karras" };
        g_rrect(back, ix, y, iw, 60, 10, 0x061022, 245);
        g_rrect_line(back, ix, y, iw, 60, 10, 1, 0x1A3158, 255);
        char row1[180], row2[180], f1[180], f2[180];
        const char* smp_n = SMP_SHORT[(v->sampler >= 0 && v->sampler <= 3) ? v->sampler : 0];
        const char* up_n = (v->upscale == 1) ? "1080p RealESRGAN" : (v->upscale == 3 ? "4K UHD" : "Off (Native)");
        snprintf(row1, sizeof(row1), "Auto-Tuned: %ux%u  ·  %u steps  ·  CFG %.1f  ·  %s",
                 v->w, v->h, v->steps, (double)v->cfg, smp_n);
        snprintf(row2, sizeof(row2), "Upscale: %s  ·  Shield: %s  ·  FaceFix: %s",
                 up_n, v->auto_quality ? "ON" : "OFF", v->auto_face_fix ? "ON" : "OFF");
        sm_fit(&g_fsmall, row1, iw - 24, f1, sizeof(f1));
        sm_fit(&g_fsmall, row2, iw - 24, f2, sizeof(f2));
        g_draw_str(back, &g_fsmall, ix + 12, y + 6, f1, 0x9CC4F8);
        g_draw_str(back, &g_fsmall, ix + 12, y + 32, f2, 0x6EF3BD);
        y += 70;
    }

    // Row 5: Generate CTA Button
    {
        int foc = (v->focus == EROW_GEN);
        uint32_t fill = !v->ready ? 0x1B3156 : (v->busy ? 0x1B3156 : C_ACCENT);
        g_rrect(back, ix, y, iw, 48, 12, fill, foc ? 255 : 230);
        g_rrect_line(back, ix, y, iw, 48, 12, foc ? 2 : 1, foc ? 0xFFFFFF : C_LINE_HI, 255);
        char gen_lbl[96], eta_b[32];
        if (v->busy) {
            sm_fmt_eta(v->eta_sec, eta_b, sizeof(eta_b));
            if (eta_b[0]) snprintf(gen_lbl, sizeof(gen_lbl), "Working...  ·  %s  ·  press O to cancel", eta_b);
            else snprintf(gen_lbl, sizeof(gen_lbl), "Working...  ·  press O to cancel");
        } else if (!v->ready) {
            snprintf(gen_lbl, sizeof(gen_lbl), "Open Starter Model Downloader (press X)");
        } else {
            snprintf(gen_lbl, sizeof(gen_lbl), "Generate Anime Artwork");
        }
        int tw = g_str_width(&g_fsmall, gen_lbl) + (v->ready && !v->busy ? 38 : 0);
        int tx = ix + (iw - tw) / 2;
        if (v->ready && !v->busy) tx += g_draw_btn(back, GBTN_TRIANGLE, tx, y + 24) + 10;
        g_draw_str(back, &g_fsmall, tx, y + (48 - g_fsmall.line) / 2, gen_lbl, 0xFFFFFF);
        y += 58;
    }

    // Row 6: Full-width Credits & Updates bar (fits cleanly with zero truncation)
    {
        int foc = (v->focus == EROW_CREDITS);
        g_rrect(back, ix, y, iw, 42, 10, foc ? C_SURFACE_HI : 0x09162D, 240);
        g_rrect_line(back, ix, y, iw, 42, 10, foc ? 2 : 1, foc ? C_ACCENT_HI : C_LINE, 255);
        char val[220], fit[220];
        sm_easy_row_value(v, EROW_CREDITS, val, sizeof(val));
        sm_fit(&g_fsmall, val, iw - 28, fit, sizeof(fit));
        g_draw_str(back, &g_fsmall, ix + 14, y + (42 - g_fsmall.line) / 2, fit, foc ? 0xFFFFFF : C_TEXT_DIM);
    }

    if (out_next_y) *out_next_y = SM_ROW0 + 680;
}

static void sm_draw_welcome(uint32_t* back, int sel, int neo, const char* model, int busy,
                            uint32_t vram_u, uint32_t vram_t, const char* addr) {
    g_draw_bg(back);
    g_draw_header(back, neo ? "PS4 Pro Neo  ·  Welcome" : "PS4 Base  ·  Welcome",
                  (model && model[0]) ? model : "Loading...", busy, vram_u, vram_t, addr);

    g_draw_str(back, &g_fbig, 56, 132, "Welcome to Garlic Diffusion", 0xFFFFFF);
    g_draw_str(back, &g_fsmall, 56, 184,
               "Choose how you want to create anime artwork. You can switch between Easy and Advanced anytime with R1!",
               C_TEXT_DIM);

    // Card 0: Easy Mode
    const int cw = 876, ch = 594, cy = 232;
    const int c0x = 56, c1x = 988;
    int s0 = (sel == 0), s1 = (sel == 1);

    g_rrect(back, c0x, cy, cw, ch, 24, s0 ? C_SURFACE_HI : C_SURFACE, s0 ? 252 : 215);
    g_rrect_line(back, c0x, cy, cw, ch, 24, s0 ? 3 : 1, s0 ? C_GREEN : C_LINE_HI, 255);
    g_rrect(back, c0x + 34, cy + 26, 184, 34, 17, 0x0F3A2A, 245);
    g_rrect_line(back, c0x + 34, cy + 26, 184, 34, 17, 1, C_GREEN, 255);
    g_draw_str(back, &g_fbold, c0x + 54, cy + 30, "RECOMMENDED", C_GREEN);
    g_draw_str(back, &g_fbig, c0x + 34, cy + 78, "Easy Mode", 0xFFFFFF);
    g_draw_str(back, &g_fsmall, c0x + 34, cy + 130, "For fast, effortless anime art with zero technical setup:", C_ACCENT_HI);
    static const char* const E_BULLETS[5] = {
        "1. Pick a Character (290+ anime heroes & heroines) or type a name",
        "2. Pick a Scene / Action preset — or press L2 for a Surprise Scene",
        "3. Choose Framing: Portrait, Upper Body, Full Body, or 16:9 Wallpaper",
        "4. Choose Speed vs Best Quality vs 1080p HD Upscale",
        "5. Optimal CFG, Steps, Sampler & Hand/Eye Shield are auto-locked!"
    };
    for (int i = 0; i < 5; i++) {
        int by = cy + 178 + i * 64;
        char bfit[180];
        sm_fit(&g_fsmall, E_BULLETS[i], cw - 108, bfit, sizeof(bfit));
        g_rrect(back, c0x + 34, by, cw - 68, 50, 12, C_PANEL, 225);
        g_draw_str(back, &g_fsmall, c0x + 54, by + 13, bfit, C_TEXT);
    }
    if (s0) {
        g_rrect(back, c0x + 34, cy + ch - 72, cw - 68, 48, 14, C_GREEN, 245);
        const char* b0 = "Press X to Start in Easy Mode";
        g_draw_str(back, &g_fbold, c0x + (cw - g_str_width(&g_fbold, b0)) / 2, cy + ch - 60, b0, 0x041A10);
    }

    // Card 1: Advanced Studio
    g_rrect(back, c1x, cy, cw, ch, 24, s1 ? C_SURFACE_HI : C_SURFACE, s1 ? 252 : 215);
    g_rrect_line(back, c1x, cy, cw, ch, 24, s1 ? 3 : 1, s1 ? C_ACCENT_HI : C_LINE_HI, 255);
    g_rrect(back, c1x + 34, cy + 26, 170, 34, 17, 0x10294E, 245);
    g_rrect_line(back, c1x + 34, cy + 26, 170, 34, 17, 1, C_ACCENT_HI, 255);
    g_draw_str(back, &g_fbold, c1x + 54, cy + 30, "PRO STUDIO", C_ACCENT_HI);
    g_draw_str(back, &g_fbig, c1x + 34, cy + 78, "Advanced Studio", 0xFFFFFF);
    g_draw_str(back, &g_fsmall, c1x + 34, cy + 130, "Full control over every parameter on PS4 GPU:", C_ACCENT_HI);
    static const char* const A_BULLETS[5] = {
        "1. Full Positive & Negative Prompt editors + inline {a|b|c} wildcards",
        "2. Native GPU .safetensors LoRA stacking & strength sliders",
        "3. Custom Resolution (up to 1024x1344), Steps, CFG & Sampler control",
        "4. Batch 1-100 runs with 500-word Randomizer & Gallery Folders",
        "5. Real-ESRGAN 1080p / 2x / 4K UHD Upscaling & Power/CU tuning"
    };
    for (int i = 0; i < 5; i++) {
        int by = cy + 178 + i * 64;
        char bfit[180];
        sm_fit(&g_fsmall, A_BULLETS[i], cw - 108, bfit, sizeof(bfit));
        g_rrect(back, c1x + 34, by, cw - 68, 50, 12, C_PANEL, 225);
        g_draw_str(back, &g_fsmall, c1x + 54, by + 13, bfit, C_TEXT);
    }
    if (s1) {
        g_rrect(back, c1x + 34, cy + ch - 72, cw - 68, 48, 14, C_ACCENT, 250);
        const char* b1 = "Press X to Start in Advanced Studio";
        g_draw_str(back, &g_fbold, c1x + (cw - g_str_width(&g_fbold, b1)) / 2, cy + ch - 60, b1, 0xFFFFFF);
    }

    // Friendly First-Time & Trial-and-Error Expectation Banner
    g_rrect(back, 56, 844, 1808, 126, 18, C_PANEL, 240);
    g_rrect_line(back, 56, 844, 1808, 126, 18, 1, C_LINE_HI, 255);
    g_draw_str(back, &g_fbold, 84, 858, "First Time Here? Built-In Model & LoRA Downloader + How AI Seeds Work:", 0xF2A65A);
    {
        char w1[200], w2[220];
        sm_fit(&g_fsmall, "No models yet? Our Guided Starter Manager opens automatically so you can download your first checkpoint in 1 click!", 1740, w1, sizeof(w1));
        sm_fit(&g_fsmall, "Note: Every render uses a random seed, so results vary! Try a few renders or a small batch, star your favorites, and upscale winners.", 1740, w2, sizeof(w2));
        g_draw_str(back, &g_fsmall, 84, 892, w1, C_TEXT);
        g_draw_str(back, &g_fsmall, 84, 922, w2, C_TEXT_DIM);
    }

    const GHint wh[] = {
        { GBTN_LEFTRIGHT, "Choose Mode" },
        { GBTN_CROSS, "Confirm & Start" },
    };
    g_draw_footer(back, wh, 2, "Web UI also ready at http://PS4-IP:8081");
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
        char gen_lbl[96], eta_b[32];
        if (v->busy) {
            sm_fmt_eta(v->eta_sec, eta_b, sizeof(eta_b));
            if (eta_b[0]) snprintf(gen_lbl, sizeof(gen_lbl), "Generating...  ·  %s  ·  press O to cancel", eta_b);
            else snprintf(gen_lbl, sizeof(gen_lbl), "Generating...  press O to cancel");
        }
        else if (!v->ready) snprintf(gen_lbl, sizeof(gen_lbl), "Download / Load a Starter Model (press X)");
        else if (v->batch > 1 && v->shutdown_after) snprintf(gen_lbl, sizeof(gen_lbl), "Generate %u Images  ·  Auto-Shutdown", v->batch);
        else if (v->batch > 1) snprintf(gen_lbl, sizeof(gen_lbl), "Generate %u Images", v->batch);
        else if (v->shutdown_after) snprintf(gen_lbl, sizeof(gen_lbl), "Generate Image  ·  Auto-Shutdown");
        else snprintf(gen_lbl, sizeof(gen_lbl), "Generate Image");
        const char* t = gen_lbl;
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

static const char* sm_easy_hint(const StudioView* v, int erow, char* buf, size_t cap) {
    switch (erow) {
    case EROW_CHAR:
        return "Pick from 290+ anime characters (Left/Right or X), or leave on Any/Custom.";
    case EROW_SCENE:
        return "Press X to pick a Scene Preset, Left/Right to cycle, L2 for a Surprise Scene, or R2 to type a custom scene!";
    case EROW_FRAMING:
        return "Controls camera distance & aspect ratio: Portrait, Upper Body, Full Body, or 16:9 Widescreen.";
    case EROW_QUALITY:
        return "Automatically locks optimal CFG, Steps, Sampler & Hand/Eye Anatomy Shield for the active checkpoint.";
    case EROW_MODEL:
        return "Switch AI checkpoint on PS4 GPU (press X). Press OPTIONS to download new models from Hugging Face.";
    case EROW_CREDITS:
        return "Open Credits, Open-Source Licenses, GitHub OTA Updater & Content Filter settings (works anytime!).";
    case EROW_GEN:
        if (v->last_sec > 0.0) {
            snprintf(buf, cap, "Press X or Triangle to generate!   Last render: %.1fs  ·  %.2f TFLOPS  ·  Press R1 for Advanced Studio", v->last_sec, v->last_tf);
            return buf;
        }
        return "Press X or Triangle to generate!   (Press R1 anytime to switch to Advanced Studio)";
    }
    return "";
}

static const char* sm_hint(const StudioView* v, int row, char* buf, size_t cap) {
    switch (row) {
    case ROW_MODEL:   return "Checkpoint on the GPU. Switching auto-applies optimal CFG, Steps & Sampler. OPTIONS opens Model & LoRA Downloader.";
    case ROW_LORA:    return "Fuses a .safetensors LoRA from /data/garlic-diffusion/loras onto GPU weights. Press X to pick a LoRA; Left/Right adjusts strength.";
    case ROW_PRESET:  return "Saved prompt & adaptive Speed presets (preserves model optimal CFG/Sampler). Press X to choose one.";
    case ROW_CHAR:    return "Inserts or swaps character tags directly into Prompt (auto-deduplicated). Left/Right cycles, X shows full list.";
    case ROW_STYLE:   return "Adds an art style (cel shading, Shinkai, ...) after your prompt. Left/Right cycles, X shows the list.";
    case ROW_PROMPT:  return v->prompt && v->prompt[0] ? v->prompt : "(empty) Press X to type a prompt.";
    case ROW_NEG:     return v->neg && v->neg[0] ? v->neg : "(empty) Press X to type a negative prompt.";
    case ROW_RES:     return "Bigger = more detail in eyes and hands. Illustrious/SDXL shines at 1024x1344 (PS4 Pro), SD 1.5 at 512x768.";
    case ROW_STEPS:   return "More steps = more detail but slower. Distilled DMD2 / 8-step models: 6-8. Full models: 20-24.";
    case ROW_CFG:     return "Prompt strength. DMD2 models shine at CFG 1.4 + LCM. 8-step: 3.5 + Euler. Full models: 5.5-6.5 + DPM++ 2M Karras.";
    case ROW_SAMPLER: return "DPM++ 2M Karras: smooth 2nd-order sampler for 20+ step models. LCM is for DMD2 models.";
    case ROW_UPSCALE: return "Real-ESRGAN Anime v3 runs on the GPU after rendering: 1080p, 2x or 4K UHD.";
    case ROW_POWER:   return "GPU compute units and fan noise. Quiet is the default; avoid Turbo for long sessions.";
    case ROW_SEED:    return "Random each time, or lock the last used seed to re-render the same composition.";
    case ROW_BATCH:   return "Generate 1-100 images in a row. Press X for 500-word Randomizer, Target Folder & Auto-Shutdown.";
    case ROW_CREDITS: return "Open Credits, Licenses, GitHub OTA Updater & Content Filter (press X or L3 anytime).";
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
    const int is_easy = (v->ui_mode == 0);
    char hsub[96];
    if (v->update_ver && v->update_ver[0]) {
        snprintf(hsub, sizeof(hsub), "%s  ·  %s  ·  Update v%s!",
                 v->neo ? "PS4 Pro" : "PS4 Base", is_easy ? "Easy Mode" : "Advanced", v->update_ver);
    } else {
        snprintf(hsub, sizeof(hsub), "%s  ·  %s",
                 v->neo ? "PS4 Pro Neo" : "PS4 Base", is_easy ? "Easy Mode [R1=Adv]" : "Advanced [R1=Easy]");
    }
    g_draw_header(back, hsub,
                  (v->model && v->model[0]) ? v->model : "No Model", v->busy, v->vram_u, v->vram_t, v->addr);

    int iy;
    if (is_easy) {
        sm_draw_easy_panel(back, v, &iy);
    } else {
        for (int r = 0; r < ROW_COUNT; r++) sm_draw_row(back, v, r);
        iy = SM_ROW0 + ROW_COUNT * SM_ROW_H + 10;
    }

    // Info box: progress + live ETA + status / help text for the focused setting
    const int ix = SM_PX, iw = SM_PW, ih = 980 - iy;
    g_rrect(back, ix, iy, iw, ih, 16, C_PANEL, 235);
    g_rrect_line(back, ix, iy, iw, ih, 16, 1, C_LINE_HI, 255);
    int bar_w = iw - 40;
    char eta_pill[64] = {0};
    if (v->busy) {
        char eb[32];
        sm_fmt_eta(v->eta_sec, eb, sizeof(eb));
        int pct = (int)(g_clamp01(v->prog) * 100.0f + 0.5f);
        if (eb[0]) snprintf(eta_pill, sizeof(eta_pill), "%d%%  ·  %s", pct, eb);
        else snprintf(eta_pill, sizeof(eta_pill), "%d%%", pct);
        int epw = g_str_width(&g_fsmall, eta_pill);
        bar_w = iw - 56 - epw;
        g_draw_str(back, &g_fsmall, ix + iw - 20 - epw, iy + 8, eta_pill, C_GREEN);
    }
    g_rrect(back, ix + 20, iy + 16, bar_w, 10, 5, 0x1B3156, 255);
    int pfw = (int)((float)bar_w * g_clamp01(v->prog));
    if (pfw > 0) g_rrect(back, ix + 20, iy + 16, pfw < 10 ? 10 : pfw, 10, 5, v->busy ? C_ACCENT_HI : C_GREEN, 255);
    char hbuf[220];
    const char* txt;
    uint32_t tcol = C_TEXT_DIM;
    if (v->err && v->err[0]) { txt = v->err; tcol = C_RED; }
    else if (v->busy) { txt = v->stage; tcol = C_TEXT; }
    else if (v->toast && v->toast[0]) { txt = v->toast; tcol = C_GREEN; }
    else if (is_easy) {
        txt = sm_easy_hint(v, v->focus, hbuf, sizeof(hbuf));
        if (is_easy && v->focus == EROW_SCENE && v->prompt && v->prompt[0]) {
            snprintf(hbuf, sizeof(hbuf), "Prompt: %s", v->prompt);
            txt = hbuf;
            tcol = C_TEXT;
        }
    } else {
        txt = sm_hint(v, v->focus, hbuf, sizeof(hbuf));
        if (v->focus == ROW_PROMPT || v->focus == ROW_NEG) tcol = C_TEXT;
    }
    int max_ln = (ih - 46) / 28;
    if (max_ln < 2) max_ln = 2;
    g_draw_wrapped(back, &g_fsmall, ix + 20, iy + 36, iw - 40, 28, tcol, txt ? txt : "", max_ln);

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
        const char* s1 = (!v->ready && !v->busy)
            ? "No checkpoint loaded yet! Press OPTIONS (or X on Generate) for Starter Setup."
            : "Native Stable Diffusion 1.5, SDXL & Real-ESRGAN on PS4 GPU.";
        const char* s2 = "Tip: Every random seed is unique — try a few renders or a small batch";
        const char* s3 = "to cherry-pick the cleanest pose, hands & eyes, then upscale your winner!";
        char fs1[180], fs2[180], fs3[180];
        sm_fit(&g_fsmall, s1, pw - 56, fs1, sizeof(fs1));
        sm_fit(&g_fsmall, s2, pw - 56, fs2, sizeof(fs2));
        sm_fit(&g_fsmall, s3, pw - 56, fs3, sizeof(fs3));
        g_draw_str(back, &g_fsmall, ecx - g_str_width(&g_fsmall, fs1) / 2, ecy + 112, fs1, (!v->ready && !v->busy) ? 0xF2A65A : C_TEXT);
        g_draw_str(back, &g_fsmall, ecx - g_str_width(&g_fsmall, fs2) / 2, ecy + 150, fs2, C_TEXT_DIM);
        g_draw_str(back, &g_fsmall, ecx - g_str_width(&g_fsmall, fs3) / 2, ecy + 178, fs3, C_TEXT_MUTED);
    }

    // Footer hints depend on the current mode — L3 Credits works anytime (even while busy!)
    const char* right = (v->update_ver && v->update_ver[0])
        ? "Update Ready! Press L3 for Credits"
        : "ko-fi.com/djtomato · Support: Discord / GitHub (L3)";
    if (v->dd_open) {
        sm_draw_dropdown(back, v);
        const GHint h[] = { { GBTN_UPDOWN, "Choose" }, { GBTN_L1, "Page up" }, { GBTN_R1, "Page down" },
                            { GBTN_CROSS, "Select" }, { GBTN_CIRCLE, "Cancel" } };
        g_draw_footer(back, h, 5, "");
    } else if (v->adjust) {
        const GHint h[] = { { GBTN_LEFTRIGHT, "Adjust" }, { GBTN_CROSS, "Confirm" }, { GBTN_CIRCLE, "Revert" } };
        g_draw_footer(back, h, 3, "");
    } else if (v->busy) {
        const GHint h[] = { { GBTN_CIRCLE, "Cancel Job" }, { GBTN_UPDOWN, "Browse" },
                            { GBTN_SQUARE, "Gallery" }, { GBTN_R1, is_easy ? "Advanced" : "Easy" },
                            { GBTN_OPTIONS, "Models" }, { GBTN_L3, "Credits" } };
        g_draw_footer(back, h, 6, right);
    } else if (is_easy) {
        const GHint h[] = { { GBTN_UPDOWN, "Select" }, { GBTN_LEFTRIGHT, "Change" }, { GBTN_CROSS, "Open" },
                            { GBTN_L2, "Surprise" }, { GBTN_TRIANGLE, "Generate" }, { GBTN_SQUARE, "Gallery" },
                            { GBTN_R1, "Advanced" }, { GBTN_L3, "Credits" } };
        g_draw_footer(back, h, 8, right);
    } else {
        const GHint h[] = { { GBTN_UPDOWN, "Select" }, { GBTN_LEFTRIGHT, "Change" }, { GBTN_CROSS, "Open" },
                            { GBTN_TRIANGLE, "Generate" }, { GBTN_SQUARE, "Gallery" }, { GBTN_R1, "Easy" },
                            { GBTN_OPTIONS, "Models" }, { GBTN_L3, "Credits" } };
        g_draw_footer(back, h, 8, right);
    }
}

#endif
