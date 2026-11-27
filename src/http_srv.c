// Garlic Diffusion HTTP server on PS4 (port 8081):
//   GET  /                     -> built-in dark-mode Web UI
//   GET  /status               -> JSON status + progress
//   GET  /image.png            -> latest generated PNG
//   POST /generate             -> queue async generation (JSON)
//   POST /cancel               -> cancel active generation
//   POST /v1/images/generations -> OpenAI-compatible synchronous b64_json image generation

#include "http_srv.h"
#include "gallery.h"
#include "model_dl.h"

#include <ctype.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <orbis/Net.h>
#include <orbis/Sysmodule.h>
#include <orbis/libkernel.h>

#define SOL_SOCKET_PS4 0xffff
#define SO_REUSEADDR_PS4 0x0004
#define SO_NOSIGPIPE_PS4 0x0800
#define IPPROTO_TCP_PS4 6
#define IPPROTO_UDP_PS4 17
#define SIGPIPE_PS4 13

int _sigaction(int sig, const void* act, void* old);
void glog(const char* fmt, ...);

typedef struct {
    uint8_t len, family;
    uint16_t port;
    uint32_t addr;
    char zero[8];
} SockAddrIn;

static OrbisPthreadMutex srv_mu;
static OrbisPthreadMutex sync_gen_mu;
static OrbisPthread srv_th;
static DiffJob* g_job;
static DiffStatus* g_st;
static char g_addr[32] = ":8081";
static char* g_web_ui = NULL;
static size_t g_web_ui_len = 0;

static void load_web_ui(void) {
    int fd = open("/app0/assets/index.html", O_RDONLY);
    if (fd < 0) return;
    off_t sz = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    if (sz > 0 && sz < (2 << 20)) {
        g_web_ui = (char*)malloc((size_t)sz + 1);
        if (g_web_ui) {
            size_t done = 0;
            while (done < (size_t)sz) {
                ssize_t r = read(fd, g_web_ui + done, (size_t)sz - done);
                if (r <= 0) break;
                done += (size_t)r;
            }
            g_web_ui[done] = 0;
            g_web_ui_len = done;
        }
    }
    close(fd);
}

static void save_ui_mode_file(int mode) {
    if (mode != 0 && mode != 1) return;
    int fd = open("/data/garlic-diffusion/ui_mode.cfg", O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd >= 0) {
        char buf[32];
        int n = snprintf(buf, sizeof(buf), "ui_mode=%d\n", mode);
        if (n > 0) (void)write(fd, buf, (size_t)n);
        close(fd);
    }
}

void http_srv_lock(void) { scePthreadMutexLock(&srv_mu); }
void http_srv_unlock(void) { scePthreadMutexUnlock(&srv_mu); }
const char* http_srv_address(void) { return g_addr; }

static int send_all(int fd, const void* buf, size_t n) {
    const char* p = (const char*)buf;
    while (n) {
        int w = sceNetSend(fd, p, n, 0);
        if (w <= 0) return -1;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

static void send_resp_ex(int fd, int code, const char* reason, const char* ctype,
                         const char* cache_hdr, const void* body, size_t len) {
    char hdr[600];
    int h = snprintf(hdr, sizeof(hdr),
                     "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                     "Cache-Control: %s\r\n"
                     "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
                     "Access-Control-Allow-Headers: Content-Type, Authorization\r\nConnection: close\r\n\r\n",
                     code, reason, ctype, len, cache_hdr);
    if (send_all(fd, hdr, (size_t)h) == 0 && len > 0) send_all(fd, body, len);
}

static void send_resp(int fd, int code, const char* reason, const char* ctype, const void* body, size_t len) {
    send_resp_ex(fd, code, reason, ctype, "no-store", body, len);
}

static void send_zip_resp(int fd, const char* filename, const void* body, size_t len) {
    char hdr[640];
    int h = snprintf(hdr, sizeof(hdr),
                     "HTTP/1.1 200 OK\r\nContent-Type: application/zip\r\nContent-Length: %zu\r\n"
                     "Content-Disposition: attachment; filename=\"%s\"\r\n"
                     "Cache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                     len, (filename && filename[0]) ? filename : "garlic_gallery.zip");
    if (send_all(fd, hdr, (size_t)h) == 0 && len > 0) send_all(fd, body, len);
}

// Tiny JSON string/number extractors
static int json_get_str(const char* json, const char* key, char* out, size_t max) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (*p != '"') return 0;
    p++;
    size_t n = 0;
    while (*p && *p != '"' && n + 1 < max) {
        if (*p == '\\' && p[1]) {
            p++;
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

static int json_get_double(const char* json, const char* key, double* out) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    if (!*p || (*p != '-' && !isdigit((unsigned char)*p))) return 0;
    *out = strtod(p, NULL);
    return 1;
}

static void json_escape(const char* s, char* out, size_t max) {
    size_t n = 0;
    for (; s && *s && n + 2 < max; s++) {
        if (*s == '"' || *s == '\\') { out[n++] = '\\'; out[n++] = *s; }
        else if ((unsigned char)*s >= 32) out[n++] = *s;
    }
    out[n] = 0;
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int url_get_param(const char* qs, const char* key, char* out, size_t max) {
    if (!qs || !key || !out || !max) return 0;
    size_t kl = strlen(key);
    const char* p = qs;
    while (*p) {
        if (strncmp(p, key, kl) == 0 && p[kl] == '=') {
            p += kl + 1;
            size_t n = 0;
            while (*p && *p != '&' && n + 1 < max) {
                if (*p == '+') { out[n++] = ' '; p++; }
                else if (*p == '%' && hex_val(p[1]) >= 0 && hex_val(p[2]) >= 0) {
                    out[n++] = (char)((hex_val(p[1]) << 4) | hex_val(p[2]));
                    p += 3;
                } else {
                    out[n++] = *p++;
                }
            }
            out[n] = 0;
            return 1;
        }
        while (*p && *p != '&') p++;
        if (*p == '&') p++;
    }
    return 0;
}

static int json_get_str_array(const char* json, const char* key, char out[][64], int max_items) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p && *p != '[' && *p != '}') p++;
    if (*p != '[') return 0;
    p++;
    int cnt = 0;
    while (*p && *p != ']' && cnt < max_items) {
        while (*p && (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
        if (*p != '"') break;
        p++;
        size_t k = 0;
        while (*p && *p != '"' && k + 1 < 64) {
            if (*p == '\\' && p[1]) { p++; out[cnt][k++] = *p++; }
            else out[cnt][k++] = *p++;
        }
        out[cnt][k] = 0;
        if (*p == '"') p++;
        if (k > 0) cnt++;
    }
    return cnt;
}

static char* b64_encode(const uint8_t* in, size_t len, size_t* out_len) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t ol = ((len + 2) / 3) * 4;
    char* out = (char*)malloc(ol + 1);
    if (!out) return NULL;
    size_t j = 0;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = ((uint32_t)in[i] << 16) | ((i + 1 < len ? (uint32_t)in[i + 1] : 0u) << 8) |
                     (i + 2 < len ? (uint32_t)in[i + 2] : 0u);
        out[j++] = tbl[(v >> 18) & 63];
        out[j++] = tbl[(v >> 12) & 63];
        out[j++] = (i + 1 < len) ? tbl[(v >> 6) & 63] : '=';
        out[j++] = (i + 2 < len) ? tbl[v & 63] : '=';
    }
    out[j] = 0;
    *out_len = j;
    return out;
}

static const char FALLBACK_UI[] =
"<!DOCTYPE html><html><body style='background:#0b0f18;color:#e8effa;font-family:sans-serif;padding:24px'>"
"<h2>Garlic Diffusion PS4</h2><p>Web UI asset missing (/app0/assets/index.html).</p></body></html>";

static void apply_job_json(const char* body, DiffJob* j) {
    char tmp[1024];
    if (json_get_str(body, "prompt", tmp, sizeof(tmp))) snprintf(j->prompt, sizeof(j->prompt), "%s", tmp);
    if (json_get_str(body, "negative_prompt", tmp, sizeof(tmp)) || json_get_str(body, "negative", tmp, sizeof(tmp)))
        snprintf(j->negative, sizeof(j->negative), "%s", tmp);
    if (json_get_str(body, "lora", tmp, sizeof(tmp))) {
        if (!strchr(tmp, '/') && !strchr(tmp, '\\') && !strstr(tmp, ".."))
            snprintf(j->lora, sizeof(j->lora), "%s", tmp);
    }
    {
        double lv = 0.8;
        if (json_get_double(body, "lora_scale", &lv) || json_get_double(body, "lora_strength", &lv)) {
            if (lv < -2.0) lv = -2.0;
            if (lv > 2.5) lv = 2.5;
            j->lora_scale = (float)lv;
        }
    }
    if (json_get_str(body, "sampler_name", tmp, sizeof(tmp)) || json_get_str(body, "sampler_index", tmp, sizeof(tmp))) {
        if (strstr(tmp, "DPM") || strstr(tmp, "dpm")) j->sampler = 3;
        else if (strcmp(tmp, "LCM") == 0 || strcmp(tmp, "lcm") == 0) j->sampler = 2;
        else if (strcmp(tmp, "Euler") == 0 || strcmp(tmp, "euler") == 0) j->sampler = 1;
        else if (strcmp(tmp, "Euler a") == 0 || strcmp(tmp, "euler_a") == 0 || strcmp(tmp, "k_euler_a") == 0) j->sampler = 0;
        // Unsupported A1111 sampler names keep the active model's optimal preset sampler!
    }
    double v;
    if (json_get_str(body, "size", tmp, sizeof(tmp))) {
        unsigned w = 0, h = 0;
        if (sscanf(tmp, "%ux%u", &w, &h) == 2 && w >= 256 && h >= 256) { j->width = w; j->height = h; }
    }
    if (json_get_double(body, "width", &v) && v >= 64) j->width = (uint32_t)v;
    if (json_get_double(body, "height", &v) && v >= 64) j->height = (uint32_t)v;
    if (json_get_double(body, "steps", &v) && v >= 1) j->steps = (uint32_t)v;
    if (json_get_double(body, "cfg_scale", &v) || json_get_double(body, "cfg", &v)) j->cfg = (float)v;
    if (json_get_double(body, "seed", &v)) j->seed = (v > 0) ? (uint64_t)v : 0;
    if (json_get_double(body, "eco", &v)) j->eco_mode = (v >= 0 && v <= 3) ? (int)v : 2;
    if (json_get_double(body, "upscale", &v)) j->upscale = (v >= 0 && v <= 3) ? (int)v : 0;
    else if (strstr(body, "\"enable_hr\":true") || strstr(body, "\"enable_hr\": true")) j->upscale = 1;
    else if (strstr(body, "\"enable_hr\":false") || strstr(body, "\"enable_hr\": false")) j->upscale = 0;
    if (json_get_double(body, "batch_count", &v) || json_get_double(body, "batch", &v) || json_get_double(body, "n_iter", &v)) {
        int bc = (int)v;
        if (bc < 1) bc = 1;
        if (bc > DIFF_MAX_BATCH_GEN) bc = DIFF_MAX_BATCH_GEN;
        j->batch_count = (uint32_t)bc;
    } else {
        j->batch_count = 1;
    }
    if (json_get_double(body, "batch_rand_mask", &v) || json_get_double(body, "rand_mask", &v)) {
        j->batch_rand_mask = (v > 0) ? (uint32_t)v : 0u;
    }
    if (json_get_str(body, "batch_folder", tmp, sizeof(tmp)) || json_get_str(body, "folder", tmp, sizeof(tmp))) {
        gallery_clean_folder_name(tmp, j->batch_folder, sizeof(j->batch_folder));
    }
    if (json_get_double(body, "nsfw_allow", &v)) j->nsfw_allow = (v != 0) ? 1 : 0;
    else if (strstr(body, "\"nsfw_allow\":true") || strstr(body, "\"nsfw_allow\": true")) j->nsfw_allow = 1;
    else if (strstr(body, "\"nsfw_allow\":false") || strstr(body, "\"nsfw_allow\": false")) j->nsfw_allow = 0;
    if (json_get_double(body, "auto_quality", &v)) j->auto_quality = (v != 0) ? 1 : 0;
    else if (strstr(body, "\"auto_quality\":true") || strstr(body, "\"auto_quality\": true")) j->auto_quality = 1;
    else if (strstr(body, "\"auto_quality\":false") || strstr(body, "\"auto_quality\": false")) j->auto_quality = 0;
    if (json_get_double(body, "auto_face_fix", &v)) j->auto_face_fix = (v != 0) ? 1 : 0;
    else if (strstr(body, "\"auto_face_fix\":true") || strstr(body, "\"auto_face_fix\": true")) j->auto_face_fix = 1;
    else if (strstr(body, "\"auto_face_fix\":false") || strstr(body, "\"auto_face_fix\": false")) j->auto_face_fix = 0;
    if (json_get_double(body, "shutdown_after", &v)) j->shutdown_after = (v != 0) ? 1 : 0;
    else if (strstr(body, "\"shutdown_after\":true") || strstr(body, "\"shutdown_after\": true")) j->shutdown_after = 1;
    else if (strstr(body, "\"shutdown_after\":false") || strstr(body, "\"shutdown_after\": false")) j->shutdown_after = 0;
}

static int ci_contains(const char* hay, const char* needle) {
    if (!hay || !needle || !needle[0]) return 0;
    size_t nl = strlen(needle);
    for (; *hay; hay++) {
        size_t i = 0;
        while (i < nl && hay[i] && tolower((unsigned char)hay[i]) == tolower((unsigned char)needle[i])) i++;
        if (i == nl) return 1;
    }
    return 0;
}

static int ci_eq(const char* a, const char* b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == 0 && *b == 0;
}

static void clean_model_name(const char* in, char* out, size_t max) {
    if (!in || !max) { if (out && max) out[0] = 0; return; }
    while (*in == ' ' || *in == '\t') in++;
    snprintf(out, max, "%s", in);
    // Strip A1111 hash suffix " [abcdef1234]"
    char* br = strstr(out, " [");
    if (br) *br = 0;
    // Strip common extensions if client appended them
    size_t n = strlen(out);
    while (n > 0 && (out[n - 1] == ' ' || out[n - 1] == '\t')) out[--n] = 0;
    static const char* EXTS[] = {".safetensors", ".ckpt", ".psd1"};
    for (int i = 0; i < 3; i++) {
        size_t el = strlen(EXTS[i]);
        if (n > el && ci_eq(out + n - el, EXTS[i])) {
            out[n - el] = 0;
            break;
        }
    }
}

static int match_installed_model(const char* req_raw, char* matched, size_t max) {
    char clean[64] = {0};
    clean_model_name(req_raw, clean, sizeof(clean));
    if (!clean[0] || strchr(clean, '/') || strchr(clean, '\\') || strstr(clean, "..")) return 0;
    for (int i = 0; i < g_st->num_models; i++) {
        if (strcmp(g_st->models[i], clean) == 0 || ci_eq(g_st->models[i], clean)) {
            snprintf(matched, max, "%s", g_st->models[i]);
            return 1;
        }
    }
    return 0;
}

static void merge_style_prompt(DiffJob* j, const char* sp, const char* sn) {
    if (sp && sp[0]) {
        // Avoid duplicating if the client already expanded the style text into prompt
        char probe[32];
        snprintf(probe, sizeof(probe), "%.24s", sp);
        if (!strstr(j->prompt, probe)) {
            const char* ph = strstr(sp, "{prompt}");
            char merged[1024];
            if (ph) {
                size_t pre = (size_t)(ph - sp);
                if (pre > 400) pre = 400;
                snprintf(merged, sizeof(merged), "%.*s%s%s", (int)pre, sp, j->prompt, ph + 8);
            } else if (j->prompt[0]) {
                snprintf(merged, sizeof(merged), "%s, %s", j->prompt, sp);
            } else {
                snprintf(merged, sizeof(merged), "%s", sp);
            }
            snprintf(j->prompt, sizeof(j->prompt), "%s", merged);
        }
    }
    if (sn && sn[0]) {
        char probe[32];
        snprintf(probe, sizeof(probe), "%.24s", sn);
        if (!strstr(j->negative, probe)) {
            if (j->negative[0]) {
                char merged[512];
                snprintf(merged, sizeof(merged), "%s, %s", j->negative, sn);
                snprintf(j->negative, sizeof(j->negative), "%s", merged);
            } else {
                snprintf(j->negative, sizeof(j->negative), "%s", sn);
            }
        }
    }
}

static void apply_a1111_styles(const char* body, DiffJob* j) {
    const char* p = strstr(body, "\"styles\"");
    if (!p) return;
    p += 8;
    while (*p && *p != '[' && *p != '}') p++;
    if (*p != '[') return;
    p++;
    char names[4][64];
    int n_names = 0;
    while (*p && *p != ']' && n_names < 4) {
        while (*p && (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
        if (*p != '"') break;
        p++;
        size_t k = 0;
        while (*p && *p != '"' && k + 1 < sizeof(names[0])) {
            if (*p == '\\' && p[1]) { p++; names[n_names][k++] = *p++; }
            else names[n_names][k++] = *p++;
        }
        names[n_names][k] = 0;
        if (*p == '"') p++;
        if (k > 0 && !ci_eq(names[n_names], "None")) n_names++;
    }
    if (n_names <= 0) return;

    StyleItem* styles = (StyleItem*)malloc(sizeof(StyleItem) * STYLES_MAX_ITEMS);
    UserPreset* presets = (UserPreset*)malloc(sizeof(UserPreset) * PRESETS_MAX_ITEMS);
    int ns = styles ? styles_list(styles, STYLES_MAX_ITEMS) : 0;
    int np = presets ? presets_list(presets, PRESETS_MAX_ITEMS) : 0;
    for (int s = 0; s < n_names; s++) {
        int found = 0;
        for (int i = 0; i < ns; i++) {
            if (ci_eq(styles[i].name, names[s]) || ci_eq(styles[i].id, names[s])) {
                merge_style_prompt(j, styles[i].prompt, styles[i].negative);
                found = 1;
                break;
            }
        }
        if (!found) {
            for (int i = 0; i < np; i++) {
                if (ci_eq(presets[i].name, names[s]) || ci_eq(presets[i].id, names[s])) {
                    merge_style_prompt(j, presets[i].prompt, presets[i].negative);
                    break;
                }
            }
        }
    }
    free(styles);
    free(presets);
}

static int wait_engine_idle(int max_ticks_200ms) {
    for (int w = 0; w < max_ticks_200ms; w++) {
        http_srv_lock();
        int idle = g_st->gpu_ready && !g_st->busy && !g_job->pending && !g_job->switch_req && !g_job->gal_op;
        http_srv_unlock();
        if (idle) return 1;
        sceKernelUsleep(200000);
    }
    return 0;
}

static int switch_model_sync(const char* req_raw) {
    if (!req_raw || !req_raw[0]) return 0;
    if (!wait_engine_idle(300)) return -1;
    int need_sw = 0;
    char target[64] = {0};
    http_srv_lock();
    if (match_installed_model(req_raw, target, sizeof(target)) && strcmp(target, g_st->model) != 0) {
        snprintf(g_job->switch_model, sizeof(g_job->switch_model), "%s", target);
        g_job->switch_req = 1;
        need_sw = 1;
    }
    http_srv_unlock();
    if (need_sw) {
        glog("a1111: switching model to '%s' (waiting for reload)...", target);
        // Wait up to 60s for switch_req to be picked up & completed
        for (int w = 0; w < 300; w++) {
            sceKernelUsleep(200000);
            http_srv_lock();
            int done = !g_job->switch_req && !g_st->busy;
            http_srv_unlock();
            if (done) break;
        }
    }
    return 0;
}

static int run_sync_generation(const char* body, uint8_t** out_png, size_t* out_sz,
                               uint32_t* out_id, DiffStatus* snap, char* out_prompt, size_t pr_max,
                               char* out_neg, size_t ng_max, int* out_sampler, int a1111_adapt) {
    // Serialize concurrent A1111 / OpenAI requests in FIFO order instead of failing with 503
    scePthreadMutexLock(&sync_gen_mu);

    // Wait for any active TV/Web/prior job or model switch to finish (up to 10 min)
    if (!wait_engine_idle(3000)) {
        scePthreadMutexUnlock(&sync_gen_mu);
        return -2;
    }

    // Support A1111 override_settings.sd_model_checkpoint auto-switch before generating
    char req_mdl[128] = {0};
    if (json_get_str(body, "sd_model_checkpoint", req_mdl, sizeof(req_mdl)) && req_mdl[0]) {
        switch_model_sync(req_mdl);
    }

    // Snapshot current job defaults + model name, then parse JSON & styles outside srv_mu (disk I/O safe)
    DiffJob tmp;
    char active_mdl[64] = {0};
    http_srv_lock();
    tmp = *g_job;
    snprintf(active_mdl, sizeof(active_mdl), "%s", g_st->model);
    http_srv_unlock();

    apply_job_json(body, &tmp);
    apply_a1111_styles(body, &tmp);
    // Protect distilled models from non-distilled A1111 bot defaults (e.g. 24 steps / CFG 5.0 / Euler a on DMD2)
    if (a1111_adapt && active_mdl[0]) {
        if (ci_contains(active_mdl, "dmd2") && (tmp.steps > 10 || tmp.cfg > 2.2f)) {
            uint32_t os = tmp.steps; float oc = tmp.cfg;
            if (ci_contains(active_mdl, "animagine")) {
                if (tmp.steps > 10) tmp.steps = 5;
                if (tmp.cfg > 2.2f) tmp.cfg = 1.1f;
            } else if (ci_contains(active_mdl, "noob")) {
                if (tmp.steps > 10) tmp.steps = 6;
                if (tmp.cfg > 2.2f) tmp.cfg = 1.2f;
            } else {
                if (tmp.steps > 10) tmp.steps = 6;
                if (tmp.cfg > 2.2f) tmp.cfg = 1.4f;
            }
            tmp.sampler = 2; // LCM
            glog("a1111: adapted DMD2 '%s' (%u stp / CFG %.1f -> %u stp / CFG %.1f / LCM)",
                 active_mdl, os, (double)oc, tmp.steps, (double)tmp.cfg);
        } else if (ci_contains(active_mdl, "8step") && (tmp.steps > 12 || tmp.cfg > 5.0f)) {
            uint32_t os = tmp.steps; float oc = tmp.cfg;
            if (tmp.steps > 12) tmp.steps = 8;
            if (tmp.cfg > 5.0f) tmp.cfg = 3.5f;
            tmp.sampler = 1; // Euler
            glog("a1111: adapted 8-step '%s' (%u stp / CFG %.1f -> %u stp / CFG %.1f / Euler)",
                 active_mdl, os, (double)oc, tmp.steps, (double)tmp.cfg);
        }
    }

    uint32_t prev_id = 0;
    http_srv_lock();
    int ok = g_st->ready && !g_st->busy && !g_job->pending && !g_job->switch_req && !g_job->gal_op;
    if (ok) {
        prev_id = g_st->image_gen_id;
        *g_job = tmp;
        if (out_prompt && pr_max) snprintf(out_prompt, pr_max, "%s", g_job->prompt);
        if (out_neg && ng_max) snprintf(out_neg, ng_max, "%s", g_job->negative);
        if (out_sampler) *out_sampler = g_job->sampler;
        g_job->cancel_req = 0;
        g_job->pending = 1;
    }
    http_srv_unlock();
    if (!ok) {
        scePthreadMutexUnlock(&sync_gen_mu);
        return -2; // busy / not ready
    }

    for (;;) {
        sceKernelUsleep(200000);
        http_srv_lock();
        int done = !g_job->pending && !g_st->busy;
        http_srv_unlock();
        if (done) break;
    }

    uint8_t* copy = NULL;
    size_t sz = 0;
    uint32_t new_id = 0;
    http_srv_lock();
    new_id = g_st->image_gen_id;
    if (snap) *snap = *g_st;
    // Return final C-expanded & quality-shielded prompt/negative so Discord bots (Red cog) see all rolled tags!
    if (out_prompt && pr_max && g_st->cur_prompt[0]) snprintf(out_prompt, pr_max, "%s", g_st->cur_prompt);
    if (out_neg && ng_max && g_st->cur_negative[0]) snprintf(out_neg, ng_max, "%s", g_st->cur_negative);
    if (new_id != prev_id && g_st->png_data && g_st->png_size) {
        sz = g_st->png_size;
        copy = (uint8_t*)malloc(sz);
        if (copy) memcpy(copy, g_st->png_data, sz);
    }
    http_srv_unlock();
    scePthreadMutexUnlock(&sync_gen_mu);
    if (!copy) return -1;
    *out_png = copy;
    *out_sz = sz;
    if (out_id) *out_id = new_id;
    return 0;
}

static void handle_client(int fd) {
    size_t req_cap = 32768;
    char* req = (char*)malloc(req_cap);
    if (!req) return;
    int got = 0, header_end = -1, content_len = 0;
    while (got < (int)req_cap - 1) {
        int r = sceNetRecv(fd, req + got, req_cap - 1 - (size_t)got, 0);
        if (r <= 0) break;
        got += r;
        req[got] = 0;
        if (header_end < 0) {
            char* he = strstr(req, "\r\n\r\n");
            if (he) {
                header_end = (int)(he - req) + 4;
                char* cl = strstr(req, "Content-Length:");
                if (!cl) cl = strstr(req, "content-length:");
                if (cl && cl < he) content_len = atoi(cl + 15);
            }
        }
        if (header_end >= 0 && got >= header_end + content_len) break;
    }
    if (got <= 0) { free(req); return; }

    char method[16] = {0}, raw_uri[512] = {0}, path[160] = {0};
    sscanf(req, "%15s %511s", method, raw_uri);
    const char* query_str = "";
    char* q = strchr(raw_uri, '?');
    if (q) {
        *q = 0;
        query_str = q + 1;
    }
    snprintf(path, sizeof(path), "%s", raw_uri);
    const char* body = (header_end >= 0) ? (req + header_end) : "";

    if (strcmp(method, "OPTIONS") == 0) {
        send_resp(fd, 204, "No Content", "text/plain", "", 0);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
        if (g_web_ui && g_web_ui_len) send_resp(fd, 200, "OK", "text/html; charset=utf-8", g_web_ui, g_web_ui_len);
        else send_resp(fd, 200, "OK", "text/html; charset=utf-8", FALLBACK_UI, sizeof(FALLBACK_UI) - 1);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/gallery") == 0) {
        char lim_s[16] = {0}, off_s[16] = {0}, q_buf[160] = {0}, fld_buf[64] = {0}, star_s[8] = {0};
        int paged = url_get_param(query_str, "limit", lim_s, sizeof(lim_s));
        int lim = paged ? atoi(lim_s) : GALLERY_MAX_ITEMS;
        if (lim < 1) lim = 24;
        if (paged && lim > 200) lim = 200;
        if (!paged && lim > GALLERY_MAX_ITEMS) lim = GALLERY_MAX_ITEMS;
        int off = url_get_param(query_str, "offset", off_s, sizeof(off_s)) ? atoi(off_s) : 0;
        if (off < 0) off = 0;
        url_get_param(query_str, "q", q_buf, sizeof(q_buf));
        url_get_param(query_str, "folder", fld_buf, sizeof(fld_buf));
        int star_only = url_get_param(query_str, "starred", star_s, sizeof(star_s)) ? (atoi(star_s) != 0) : 0;
        if (strcmp(fld_buf, "__starred__") == 0) {
            star_only = 1;
            fld_buf[0] = 0;
        }

        GalleryItem* items = (GalleryItem*)malloc(sizeof(GalleryItem) * (size_t)lim);
        if (!items) { send_resp(fd, 500, "Internal Server Error", "text/plain", "oom", 3); free(req); return; }
        int total = 0;
        int cnt = (paged || fld_buf[0] || star_only)
                      ? gallery_list_page_ex(items, off, lim, q_buf, fld_buf, star_only, &total)
                      : gallery_list(items, lim);
        size_t cap = (size_t)(cnt + 1) * 2500u + 320u;
        char* out = (char*)malloc(cap);
        if (out) {
            size_t pos = 0;
            if (paged) {
                pos += (size_t)snprintf(out + pos, cap - pos,
                                        "{\"total\":%d,\"offset\":%d,\"limit\":%d,\"items\":[",
                                        total, off, lim);
            } else {
                out[pos++] = '[';
            }
            for (int i = 0; i < cnt && pos + 2400u < cap; i++) {
                char pr[1500], ng[800], md[128], sm[32], lr[192], fl[96];
                json_escape(items[i].prompt, pr, sizeof(pr));
                json_escape(items[i].negative, ng, sizeof(ng));
                json_escape(items[i].model, md, sizeof(md));
                json_escape(items[i].sampler, sm, sizeof(sm));
                json_escape(items[i].lora, lr, sizeof(lr));
                json_escape(items[i].folder, fl, sizeof(fl));
                pos += (size_t)snprintf(out + pos, cap - pos,
                                        "%s{\"id\":\"%s\",\"model\":\"%s\",\"width\":%u,\"height\":%u,"
                                        "\"steps\":%u,\"cfg\":%.2f,\"sampler\":\"%s\",\"seed\":%llu,"
                                        "\"eco\":%d,\"upscaled\":%d,\"sec\":%.2f,\"unet_sec\":%.2f,\"vae_sec\":%.2f,"
                                        "\"upscale_sec\":%.2f,\"face_fix_sec\":%.2f,\"cooldown_sec\":%.2f,\"eff_tflops\":%.3f,\"bytes\":%zu,"
                                        "\"lora\":\"%s\",\"folder\":\"%s\",\"starred\":%s,"
                                        "\"prompt\":\"%s\",\"negative\":\"%s\"}",
                                        i ? "," : "", items[i].id, md, items[i].width, items[i].height,
                                        items[i].steps, (double)items[i].cfg, sm,
                                        (unsigned long long)items[i].seed, items[i].eco, items[i].upscaled,
                                        items[i].sec, items[i].unet_sec, items[i].vae_sec,
                                        items[i].upscale_sec, items[i].face_fix_sec, items[i].cooldown_sec,
                                        items[i].eff_tflops, items[i].bytes, lr, fl,
                                        items[i].starred ? "true" : "false", pr, ng);
            }
            if (paged) { out[pos++] = ']'; out[pos++] = '}'; }
            else { out[pos++] = ']'; }
            out[pos] = 0;
            send_resp(fd, 200, "OK", "application/json", out, pos);
            free(out);
        }
        free(items);
    } else if (strcmp(method, "GET") == 0 && strncmp(path, "/gallery/thumb/", 15) == 0) {
        const char* fn = path + 15;
        size_t fl = strlen(fn);
        if (fl > 4 && strcmp(fn + fl - 4, ".png") == 0 && fl - 4 < 64) {
            char id[64];
            memcpy(id, fn, fl - 4);
            id[fl - 4] = 0;
            size_t sz = 0;
            uint8_t* tpng = gallery_load_thumb_png(id, &sz);
            if (tpng) {
                send_resp_ex(fd, 200, "OK", "image/png", "public, max-age=31536000, immutable", tpng, sz);
                free(tpng);
            } else {
                send_resp(fd, 404, "Not Found", "text/plain", "not found", 9);
            }
        } else {
            send_resp(fd, 404, "Not Found", "text/plain", "not found", 9);
        }
    } else if (strcmp(method, "GET") == 0 && strncmp(path, "/gallery/item/", 14) == 0) {
        const char* id = path + 14;
        GalleryItem it;
        if (gallery_get(id, &it) == 0) {
            char pr[1500], ng[800], md[128], sm[32], lr[192], fl[96], out[3200];
            json_escape(it.prompt, pr, sizeof(pr));
            json_escape(it.negative, ng, sizeof(ng));
            json_escape(it.model, md, sizeof(md));
            json_escape(it.sampler, sm, sizeof(sm));
            json_escape(it.lora, lr, sizeof(lr));
            json_escape(it.folder, fl, sizeof(fl));
            int n = snprintf(out, sizeof(out),
                             "{\"id\":\"%s\",\"model\":\"%s\",\"width\":%u,\"height\":%u,"
                             "\"steps\":%u,\"cfg\":%.2f,\"sampler\":\"%s\",\"seed\":%llu,"
                             "\"eco\":%d,\"upscaled\":%d,\"sec\":%.2f,\"unet_sec\":%.2f,\"vae_sec\":%.2f,"
                             "\"upscale_sec\":%.2f,\"face_fix_sec\":%.2f,\"cooldown_sec\":%.2f,\"eff_tflops\":%.3f,\"bytes\":%zu,"
                             "\"lora\":\"%s\",\"folder\":\"%s\",\"starred\":%s,"
                             "\"prompt\":\"%s\",\"negative\":\"%s\"}",
                             it.id, md, it.width, it.height, it.steps, (double)it.cfg, sm,
                             (unsigned long long)it.seed, it.eco, it.upscaled, it.sec,
                             it.unet_sec, it.vae_sec, it.upscale_sec, it.face_fix_sec,
                             it.cooldown_sec, it.eff_tflops, it.bytes,
                             lr, fl, it.starred ? "true" : "false", pr, ng);
            send_resp(fd, 200, "OK", "application/json", out, (size_t)n);
        } else {
            send_resp(fd, 404, "Not Found", "application/json", "{\"error\":\"not found\"}", 21);
        }
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/gallery/folders") == 0) {
        GalleryFolder flist[GALLERY_MAX_FOLDERS];
        int uns = 0, star = 0, all = 0;
        int nf = gallery_list_folders(flist, GALLERY_MAX_FOLDERS, &uns, &star, &all);
        char* out = (char*)malloc(8192);
        if (out) {
            size_t pos = (size_t)snprintf(out, 8192, "{\"all\":%d,\"unsorted\":%d,\"starred\":%d,\"folders\":[",
                                          all, uns, star);
            for (int i = 0; i < nf && pos + 128 < 8192; i++) {
                char ne[96];
                json_escape(flist[i].name, ne, sizeof(ne));
                pos += (size_t)snprintf(out + pos, 8192 - pos, "%s{\"name\":\"%s\",\"count\":%d}",
                                        i ? "," : "", ne, flist[i].count);
            }
            pos += (size_t)snprintf(out + pos, 8192 - pos, "]}");
            send_resp(fd, 200, "OK", "application/json", out, pos);
            free(out);
        } else {
            send_resp(fd, 500, "Internal Server Error", "text/plain", "oom", 3);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/gallery/folder/create") == 0) {
        char raw[64] = {0}, clean[48] = {0};
        json_get_str(body, "name", raw, sizeof(raw));
        if (gallery_create_folder(raw, clean, sizeof(clean)) == 0) {
            char ce[96], resp[160];
            json_escape(clean, ce, sizeof(ce));
            int l = snprintf(resp, sizeof(resp), "{\"ok\":true,\"name\":\"%s\"}", ce);
            send_resp(fd, 200, "OK", "application/json", resp, (size_t)l);
        } else {
            send_resp(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid folder name\"}", 31);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/gallery/folder/delete") == 0) {
        char raw[64] = {0};
        double dv = 0;
        json_get_str(body, "name", raw, sizeof(raw));
        int del_imgs = json_get_double(body, "delete_images", &dv) ? ((int)dv != 0) :
                       (strstr(body, "\"delete_images\":true") != NULL || strstr(body, "\"delete_images\": true") != NULL);
        if (gallery_delete_folder(raw, del_imgs) == 0) {
            http_srv_lock();
            if (ci_eq(g_job->batch_folder, raw)) g_job->batch_folder[0] = 0;
            http_srv_unlock();
            send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
        } else {
            send_resp(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid folder\"}", 26);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/gallery/folder/move") == 0) {
        char (*ids)[64] = (char (*)[64])malloc(256 * sizeof(*ids));
        char fld[64] = {0};
        json_get_str(body, "folder", fld, sizeof(fld));
        int n_ids = ids ? json_get_str_array(body, "ids", ids, 256) : 0;
        int moved = (ids && n_ids > 0) ? gallery_set_folder_batch((const char (*)[64])ids, n_ids, fld) : -1;
        free(ids);
        if (moved >= 0) {
            char resp[64];
            int l = snprintf(resp, sizeof(resp), "{\"ok\":true,\"moved\":%d}", moved);
            send_resp(fd, 200, "OK", "application/json", resp, (size_t)l);
        } else {
            send_resp(fd, 400, "Bad Request", "application/json", "{\"error\":\"move failed\"}", 23);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/gallery/star") == 0) {
        char id[64] = {0};
        double sv = 1.0;
        json_get_str(body, "id", id, sizeof(id));
        int st = json_get_double(body, "starred", &sv) ? ((int)sv != 0) :
                 (strstr(body, "\"starred\":false") == NULL && strstr(body, "\"starred\": false") == NULL);
        if (gallery_set_starred(id, st) == 0) send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
        else send_resp(fd, 404, "Not Found", "application/json", "{\"error\":\"not found\"}", 21);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/gallery/batch_delete") == 0) {
        char (*ids)[64] = (char (*)[64])malloc(512 * sizeof(*ids));
        int n_ids = ids ? json_get_str_array(body, "ids", ids, 512) : 0;
        int del = 0;
        for (int i = 0; i < n_ids; i++) {
            if (gallery_delete(ids[i]) == 0) del++;
        }
        free(ids);
        char resp[64];
        int l = snprintf(resp, sizeof(resp), "{\"ok\":true,\"deleted\":%d}", del);
        send_resp(fd, 200, "OK", "application/json", resp, (size_t)l);
    } else if ((strcmp(method, "GET") == 0 && (strcmp(path, "/gallery/zip") == 0 || strcmp(path, "/gallery/folder/zip") == 0)) ||
               (strcmp(method, "POST") == 0 && strcmp(path, "/gallery/batch_zip") == 0)) {
        char fld[64] = {0};
        char (*ids)[64] = NULL;
        int n_ids = 0;
        if (strcmp(method, "GET") == 0) {
            url_get_param(query_str, "folder", fld, sizeof(fld));
        } else {
            json_get_str(body, "folder", fld, sizeof(fld));
            ids = (char (*)[64])malloc(256 * sizeof(*ids));
            if (ids) n_ids = json_get_str_array(body, "ids", ids, 256);
        }
        size_t zsz = 0;
        uint8_t* zbuf = gallery_build_zip((const char (*)[64])ids, n_ids, fld, &zsz);
        free(ids);
        if (zbuf && zsz > 0) {
            char zname[96], clean_f[48];
            if (gallery_clean_folder_name(fld, clean_f, sizeof(clean_f)))
                snprintf(zname, sizeof(zname), "garlic_%s.zip", clean_f);
            else
                snprintf(zname, sizeof(zname), "garlic_gallery_batch.zip");
            send_zip_resp(fd, zname, zbuf, zsz);
            free(zbuf);
        } else {
            send_resp(fd, 404, "Not Found", "application/json", "{\"error\":\"no matching images\"}", 30);
        }
    } else if (strcmp(method, "GET") == 0 && strncmp(path, "/gallery/", 9) == 0) {
        const char* fn = path + 9;
        size_t fl = strlen(fn);
        if (fl > 4 && strcmp(fn + fl - 4, ".png") == 0 && fl - 4 < 64) {
            char id[64];
            memcpy(id, fn, fl - 4);
            id[fl - 4] = 0;
            size_t sz = 0;
            uint8_t* png = gallery_load_png(id, &sz);
            if (png) {
                send_resp_ex(fd, 200, "OK", "image/png", "public, max-age=3600", png, sz);
                free(png);
            } else {
                send_resp(fd, 404, "Not Found", "text/plain", "not found", 9);
            }
        } else {
            send_resp(fd, 404, "Not Found", "text/plain", "not found", 9);
        }
    } else if ((strcmp(method, "DELETE") == 0 && strncmp(path, "/gallery/", 9) == 0) ||
               (strcmp(method, "POST") == 0 && strcmp(path, "/gallery/delete") == 0)) {
        char id[64] = {0};
        if (strcmp(method, "DELETE") == 0) snprintf(id, sizeof(id), "%s", path + 9);
        else json_get_str(body, "id", id, sizeof(id));
        if (gallery_delete(id) == 0) send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
        else send_resp(fd, 404, "Not Found", "application/json", "{\"error\":\"not found\"}", 21);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/gallery/upscale") == 0) {
        char id[64] = {0};
        double md = 1.0;
        json_get_str(body, "id", id, sizeof(id));
        json_get_double(body, "mode", &md);
        http_srv_lock();
        int ok = gallery_valid_id(id) && g_st->ready && !g_st->busy && !g_job->pending && !g_job->switch_req && !g_job->gal_op;
        if (ok) {
            snprintf(g_job->gal_id, sizeof(g_job->gal_id), "%s", id);
            snprintf(g_job->gal_ids[0], sizeof(g_job->gal_ids[0]), "%s", id);
            g_job->gal_ids_n = 1;
            int m = (int)md;
            g_job->gal_op = (m == 3) ? 3 : ((m == 2) ? 2 : 1);
            g_job->cancel_req = 0;
        }
        http_srv_unlock();
        if (ok) send_resp(fd, 202, "Accepted", "application/json", "{\"ok\":true}", 11);
        else send_resp(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"busy\"}", 16);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/gallery/batch_upscale") == 0) {
        char raw_ids[DIFF_MAX_BATCH_GAL][64];
        int n_raw = json_get_str_array(body, "ids", raw_ids, DIFF_MAX_BATCH_GAL);
        double md = 1.0;
        json_get_double(body, "mode", &md);
        int m = ((int)md == 3) ? 3 : (((int)md == 2) ? 2 : 1);
        http_srv_lock();
        int can_q = g_st->ready && !g_st->busy && !g_job->pending && !g_job->switch_req && !g_job->gal_op;
        int valid_n = 0;
        if (can_q && n_raw > 0) {
            for (int i = 0; i < n_raw && valid_n < DIFF_MAX_BATCH_GAL; i++) {
                if (gallery_valid_id(raw_ids[i])) {
                    snprintf(g_job->gal_ids[valid_n], sizeof(g_job->gal_ids[0]), "%s", raw_ids[i]);
                    valid_n++;
                }
            }
            if (valid_n > 0) {
                snprintf(g_job->gal_id, sizeof(g_job->gal_id), "%s", g_job->gal_ids[0]);
                g_job->gal_ids_n = valid_n;
                g_job->gal_op = m;
                g_job->cancel_req = 0;
            }
        }
        http_srv_unlock();
        if (!can_q) send_resp(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"busy\"}", 16);
        else if (valid_n <= 0) send_resp(fd, 400, "Bad Request", "application/json", "{\"error\":\"no valid gallery ids\"}", 32);
        else {
            char resp[64];
            int l = snprintf(resp, sizeof(resp), "{\"ok\":true,\"count\":%d}", valid_n);
            send_resp(fd, 202, "Accepted", "application/json", resp, (size_t)l);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/gallery/fix_face") == 0) {
        char id[64] = {0};
        double dn = 0.42;
        json_get_str(body, "id", id, sizeof(id));
        json_get_double(body, "denoise", &dn);
        if (dn < 0.15) dn = 0.15;
        if (dn > 0.75) dn = 0.75;
        http_srv_lock();
        int ok = gallery_valid_id(id) && g_st->ready && !g_st->busy && !g_job->pending && !g_job->switch_req && !g_job->gal_op;
        if (ok) {
            snprintf(g_job->gal_id, sizeof(g_job->gal_id), "%s", id);
            snprintf(g_job->gal_ids[0], sizeof(g_job->gal_ids[0]), "%s", id);
            g_job->gal_ids_n = 1;
            g_job->gal_denoise = (float)dn;
            g_job->gal_op = 5;
            g_job->cancel_req = 0;
        }
        http_srv_unlock();
        if (ok) send_resp(fd, 202, "Accepted", "application/json", "{\"ok\":true}", 11);
        else send_resp(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"busy\"}", 16);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/gallery/resize") == 0) {
        char id[64] = {0};
        double rw = 0, rh = 0;
        json_get_str(body, "id", id, sizeof(id));
        json_get_double(body, "width", &rw);
        json_get_double(body, "height", &rh);
        http_srv_lock();
        int ok = gallery_valid_id(id) && rw >= 64 && rh >= 64 && rw <= 3840 && rh <= 3840 &&
                 g_st->ready && !g_st->busy && !g_job->pending && !g_job->switch_req && !g_job->gal_op;
        if (ok) {
            snprintf(g_job->gal_id, sizeof(g_job->gal_id), "%s", id);
            g_job->gal_w = (uint32_t)rw;
            g_job->gal_h = (uint32_t)rh;
            g_job->gal_op = 4;
        }
        http_srv_unlock();
        if (ok) send_resp(fd, 202, "Accepted", "application/json", "{\"ok\":true}", 11);
        else send_resp(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"busy\"}", 16);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/presets") == 0) {
        UserPreset* items = (UserPreset*)malloc(sizeof(UserPreset) * PRESETS_MAX_ITEMS);
        if (!items) { send_resp(fd, 500, "Internal Server Error", "text/plain", "oom", 3); free(req); return; }
        int cnt = presets_list(items, PRESETS_MAX_ITEMS);
        size_t cap = (size_t)(cnt + 1) * 2400u + 256u;
        char* out = (char*)malloc(cap);
        if (out) {
            size_t pos = 0;
            out[pos++] = '[';
            for (int i = 0; i < cnt && pos + 2200u < cap; i++) {
                char nm[128], md[128], pr[1500], ng[800], sm[32];
                json_escape(items[i].name, nm, sizeof(nm));
                json_escape(items[i].model, md, sizeof(md));
                json_escape(items[i].prompt, pr, sizeof(pr));
                json_escape(items[i].negative, ng, sizeof(ng));
                json_escape(items[i].sampler, sm, sizeof(sm));
                pos += (size_t)snprintf(out + pos, cap - pos,
                                        "%s{\"id\":\"%s\",\"name\":\"%s\",\"model\":\"%s\",\"width\":%u,\"height\":%u,"
                                        "\"steps\":%u,\"cfg\":%.2f,\"sampler\":\"%s\",\"upscale\":%d,\"eco\":%d,"
                                        "\"nsfw\":%s,\"nsfw_only\":%s,\"is_speed\":%s,"
                                        "\"prompt\":\"%s\",\"negative\":\"%s\"}",
                                        i ? "," : "", items[i].id, nm, md, items[i].width, items[i].height,
                                        items[i].steps, (double)items[i].cfg, sm, items[i].upscale, items[i].eco,
                                        items[i].nsfw ? "true" : "false", items[i].nsfw ? "true" : "false",
                                        items[i].is_speed ? "true" : "false",
                                        pr, ng);
            }
            out[pos++] = ']';
            out[pos] = 0;
            send_resp(fd, 200, "OK", "application/json", out, pos);
            free(out);
        }
        free(items);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/presets/save") == 0) {
        UserPreset p;
        memset(&p, 0, sizeof(p));
        json_get_str(body, "id", p.id, sizeof(p.id));
        json_get_str(body, "name", p.name, sizeof(p.name));
        json_get_str(body, "model", p.model, sizeof(p.model));
        json_get_str(body, "prompt", p.prompt, sizeof(p.prompt));
        if (!json_get_str(body, "negative", p.negative, sizeof(p.negative)))
            json_get_str(body, "negative_prompt", p.negative, sizeof(p.negative));
        if (!json_get_str(body, "sampler_name", p.sampler, sizeof(p.sampler)))
            snprintf(p.sampler, sizeof(p.sampler), "Euler a");
        double v = 0;
        p.width = (json_get_double(body, "width", &v) && v >= 256) ? (uint32_t)v : 768u;
        p.height = (json_get_double(body, "height", &v) && v >= 256) ? (uint32_t)v : 1024u;
        p.steps = (json_get_double(body, "steps", &v) && v >= 1) ? (uint32_t)v : 20u;
        p.cfg = (json_get_double(body, "cfg", &v) && v >= 1.0) ? (float)v : 7.0f;
        p.upscale = json_get_double(body, "upscale", &v) ? (int)v : 0;
        p.eco = json_get_double(body, "eco", &v) ? (int)v : 2;
        if (json_get_double(body, "nsfw", &v)) p.nsfw = (v != 0) ? 1 : 0;
        else if (strstr(body, "\"nsfw\":true") || strstr(body, "\"nsfw\": true")) p.nsfw = 1;
        if (json_get_double(body, "is_speed", &v)) p.is_speed = (v != 0) ? 1 : 0;
        else if (strstr(body, "\"is_speed\":true") || strstr(body, "\"is_speed\": true")) p.is_speed = 1;
        p.nsfw = json_get_double(body, "nsfw_only", &v) ? ((int)v != 0) :
                 (json_get_double(body, "nsfw", &v) ? ((int)v != 0) :
                  (strstr(body, "\"nsfw_only\":true") != NULL || strstr(body, "\"nsfw_only\": true") != NULL ||
                   strstr(body, "\"nsfw\":true") != NULL || strstr(body, "\"nsfw\": true") != NULL));
        p.is_speed = json_get_double(body, "is_speed", &v) ? ((int)v != 0) :
                     (strstr(body, "\"is_speed\":true") != NULL || strstr(body, "\"is_speed\": true") != NULL);
        char sid[64] = {0};
        if (presets_save(&p, sid, sizeof(sid)) == 0) {
            http_srv_lock(); g_job->presets_dirty = 1; http_srv_unlock();
            char resp[128];
            int n = snprintf(resp, sizeof(resp), "{\"ok\":true,\"id\":\"%s\"}", sid);
            send_resp(fd, 200, "OK", "application/json", resp, (size_t)n);
        } else {
            send_resp(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid preset\"}", 26);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/presets/delete") == 0) {
        char id[64] = {0};
        json_get_str(body, "id", id, sizeof(id));
        if (presets_delete(id) == 0) {
            http_srv_lock(); g_job->presets_dirty = 1; http_srv_unlock();
            send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
        } else {
            send_resp(fd, 404, "Not Found", "application/json", "{\"error\":\"not found\"}", 21);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/model/preset") == 0) {
        char mdl[64] = {0}, smp[16] = "Euler a", desc[96] = {0}, pr[512] = {0}, ng[384] = {0};
        json_get_str(body, "model", mdl, sizeof(mdl));
        json_get_str(body, "sampler_name", smp, sizeof(smp));
        json_get_str(body, "description", desc, sizeof(desc));
        json_get_str(body, "prompt", pr, sizeof(pr));
        json_get_str(body, "negative", ng, sizeof(ng));
        double v = 0;
        uint32_t w = (json_get_double(body, "width", &v) && v >= 256) ? (uint32_t)v : 768u;
        uint32_t h = (json_get_double(body, "height", &v) && v >= 256) ? (uint32_t)v : 1024u;
        uint32_t st = (json_get_double(body, "steps", &v) && v >= 1) ? (uint32_t)v : 20u;
        float cf = (json_get_double(body, "cfg", &v) && v >= 1.0) ? (float)v : 7.0f;
        int ups = json_get_double(body, "upscale", &v) ? (int)v : 0;
        http_srv_lock();
        if (!mdl[0]) snprintf(mdl, sizeof(mdl), "%s", g_st->model);
        int is15 = g_st->is_sd15;
        http_srv_unlock();
        if (model_preset_save("/data/garlic-diffusion/models", mdl, is15, w, h, st, cf, smp, ups, desc, pr, ng) == 0) {
            http_srv_lock();
            if (strcmp(mdl, g_st->model) == 0) {
                g_st->preset_w = w; g_st->preset_h = h; g_st->preset_steps = st;
                g_st->preset_cfg = cf; g_st->preset_upscale = ups;
                g_st->preset_sampler = (strstr(smp, "DPM") || strstr(smp, "dpm")) ? 3 : ((strcmp(smp, "LCM") == 0) ? 2 : (strcmp(smp, "Euler") == 0 ? 1 : 0));
                snprintf(g_st->preset_desc, sizeof(g_st->preset_desc), "%s", desc);
                snprintf(g_st->preset_prompt, sizeof(g_st->preset_prompt), "%s", pr);
                snprintf(g_st->preset_negative, sizeof(g_st->preset_negative), "%s", ng);
            }
            http_srv_unlock();
            send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
        } else {
            send_resp(fd, 400, "Bad Request", "application/json", "{\"error\":\"save failed\"}", 23);
        }
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/styles") == 0) {
        StyleItem* items = (StyleItem*)malloc(sizeof(StyleItem) * STYLES_MAX_ITEMS);
        if (!items) { send_resp(fd, 500, "Internal Server Error", "text/plain", "oom", 3); free(req); return; }
        int cnt = styles_list(items, STYLES_MAX_ITEMS);
        size_t cap = (size_t)(cnt + 1) * 1400u + 128u;
        char* out = (char*)malloc(cap);
        if (out) {
            size_t pos = 0;
            out[pos++] = '[';
            for (int i = 0; i < cnt && pos + 1300u < cap; i++) {
                char nm[128], pr[800], ng[600];
                json_escape(items[i].name, nm, sizeof(nm));
                json_escape(items[i].prompt, pr, sizeof(pr));
                json_escape(items[i].negative, ng, sizeof(ng));
                pos += (size_t)snprintf(out + pos, cap - pos,
                                        "%s{\"id\":\"%s\",\"name\":\"%s\",\"prompt\":\"%s\",\"negative\":\"%s\"}",
                                        i ? "," : "", items[i].id, nm, pr, ng);
            }
            out[pos++] = ']';
            out[pos] = 0;
            send_resp(fd, 200, "OK", "application/json", out, pos);
            free(out);
        }
        free(items);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/styles/save") == 0) {
        StyleItem s;
        memset(&s, 0, sizeof(s));
        json_get_str(body, "id", s.id, sizeof(s.id));
        json_get_str(body, "name", s.name, sizeof(s.name));
        json_get_str(body, "prompt", s.prompt, sizeof(s.prompt));
        if (!json_get_str(body, "negative", s.negative, sizeof(s.negative)))
            json_get_str(body, "negative_prompt", s.negative, sizeof(s.negative));
        char sid[64] = {0};
        if (styles_save(&s, sid, sizeof(sid)) == 0) {
            http_srv_lock(); g_job->styles_dirty = 1; http_srv_unlock();
            char resp[128];
            int n = snprintf(resp, sizeof(resp), "{\"ok\":true,\"id\":\"%s\"}", sid);
            send_resp(fd, 200, "OK", "application/json", resp, (size_t)n);
        } else {
            send_resp(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid style\"}", 25);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/styles/delete") == 0) {
        char id[64] = {0};
        json_get_str(body, "id", id, sizeof(id));
        if (styles_delete(id) == 0) {
            http_srv_lock(); g_job->styles_dirty = 1; http_srv_unlock();
            send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
        } else send_resp(fd, 404, "Not Found", "application/json", "{\"error\":\"not found\"}", 21);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/characters") == 0) {
        CharacterItem* items = (CharacterItem*)malloc(sizeof(CharacterItem) * CHARS_MAX_ITEMS);
        if (!items) { send_resp(fd, 500, "Internal Server Error", "text/plain", "oom", 3); free(req); return; }
        int cnt = characters_list(items, CHARS_MAX_ITEMS);
        size_t cap = (size_t)(cnt + 1) * 1500u + 256u;
        char* out = (char*)malloc(cap);
        if (out) {
            size_t pos = 0;
            out[pos++] = '[';
            for (int i = 0; i < cnt && pos + 1400u < cap; i++) {
                char nm[128], sr[96], pr[800], ng[600];
                json_escape(items[i].name, nm, sizeof(nm));
                json_escape(items[i].series, sr, sizeof(sr));
                json_escape(items[i].prompt, pr, sizeof(pr));
                json_escape(items[i].negative, ng, sizeof(ng));
                pos += (size_t)snprintf(out + pos, cap - pos,
                                        "%s{\"id\":\"%s\",\"name\":\"%s\",\"series\":\"%s\",\"prompt\":\"%s\",\"negative\":\"%s\"}",
                                        i ? "," : "", items[i].id, nm, sr, pr, ng);
            }
            out[pos++] = ']';
            out[pos] = 0;
            send_resp(fd, 200, "OK", "application/json", out, pos);
            free(out);
        }
        free(items);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/characters/save") == 0) {
        CharacterItem c;
        memset(&c, 0, sizeof(c));
        json_get_str(body, "id", c.id, sizeof(c.id));
        json_get_str(body, "name", c.name, sizeof(c.name));
        json_get_str(body, "series", c.series, sizeof(c.series));
        json_get_str(body, "prompt", c.prompt, sizeof(c.prompt));
        if (!json_get_str(body, "negative", c.negative, sizeof(c.negative)))
            json_get_str(body, "negative_prompt", c.negative, sizeof(c.negative));
        char cid[64] = {0};
        if (characters_save(&c, cid, sizeof(cid)) == 0) {
            http_srv_lock(); g_job->chars_dirty = 1; http_srv_unlock();
            char resp[128];
            int n = snprintf(resp, sizeof(resp), "{\"ok\":true,\"id\":\"%s\"}", cid);
            send_resp(fd, 200, "OK", "application/json", resp, (size_t)n);
        } else {
            send_resp(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid character\"}", 29);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/characters/delete") == 0) {
        char id[64] = {0};
        json_get_str(body, "id", id, sizeof(id));
        if (characters_delete(id) == 0) {
            http_srv_lock(); g_job->chars_dirty = 1; http_srv_unlock();
            send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
        } else send_resp(fd, 404, "Not Found", "application/json", "{\"error\":\"not found\"}", 21);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/models/manager") == 0) {
        InstalledModel* inst = (InstalledModel*)malloc(sizeof(InstalledModel) * MODEL_DL_MAX_INSTALLED);
        CatalogModel* cat = (CatalogModel*)malloc(sizeof(CatalogModel) * MODEL_DL_MAX_CATALOG);
        char* out = (char*)malloc(32768);
        if (inst && cat && out) {
            int ni = model_dl_list_installed(inst, MODEL_DL_MAX_INSTALLED);
            int nc = model_dl_get_catalog(cat, MODEL_DL_MAX_CATALOG);
            ModelDlStatus dl;
            model_dl_get_status(&dl);
            char repo[128], act_m[64] = {0};
            model_dl_get_repo(repo, sizeof(repo));
            http_srv_lock();
            snprintf(act_m, sizeof(act_m), "%s", g_st->model);
            http_srv_unlock();

            char repo_e[256], act_e[128], dlm_e[128], dlr_e[256], dlf_e[128], dls_e[256], dle_e[384];
            json_escape(repo, repo_e, sizeof(repo_e));
            json_escape(act_m, act_e, sizeof(act_e));
            json_escape(dl.model_id, dlm_e, sizeof(dlm_e));
            json_escape(dl.repo, dlr_e, sizeof(dlr_e));
            json_escape(dl.cur_file, dlf_e, sizeof(dlf_e));
            json_escape(dl.stage, dls_e, sizeof(dls_e));
            json_escape(dl.error, dle_e, sizeof(dle_e));

            size_t pos = 0;
            pos += (size_t)snprintf(out + pos, 32768 - pos,
                                    "{\"repo\":\"%s\",\"active_model\":\"%s\",\"has_vae_enc_sdxl\":%s,\"has_vae_enc_sd15\":%s,\"installed\":[",
                                    repo_e, act_e, vae_enc_is_installed(0) ? "true" : "false", vae_enc_is_installed(1) ? "true" : "false");
            for (int i = 0; i < ni && pos + 320 < 32768; i++) {
                char ne[128], de[192];
                json_escape(inst[i].name, ne, sizeof(ne));
                json_escape(inst[i].detail, de, sizeof(de));
                pos += (size_t)snprintf(out + pos, 32768 - pos,
                                        "%s{\"name\":\"%s\",\"arch\":\"%s\",\"size_mb\":%u,\"complete\":%s,\"detail\":\"%s\"}",
                                        i ? "," : "", ne, inst[i].arch, inst[i].size_mb,
                                        inst[i].complete ? "true" : "false", de);
            }
            pos += (size_t)snprintf(out + pos, 32768 - pos, "],\"catalog\":[");
            for (int i = 0; i < nc && pos + 640 < 32768; i++) {
                char ide[128], te[160], de[256], fe[384], re[256];
                json_escape(cat[i].id, ide, sizeof(ide));
                json_escape(cat[i].title, te, sizeof(te));
                json_escape(cat[i].desc, de, sizeof(de));
                json_escape(cat[i].files, fe, sizeof(fe));
                json_escape(cat[i].repo, re, sizeof(re));
                pos += (size_t)snprintf(out + pos, 32768 - pos,
                                        "%s{\"id\":\"%s\",\"title\":\"%s\",\"arch\":\"%s\",\"size_mb\":%u,"
                                        "\"desc\":\"%s\",\"files\":\"%s\",\"repo\":\"%s\"}",
                                        i ? "," : "", ide, te, cat[i].arch, cat[i].size_mb, de, fe, re);
            }
            pos += (size_t)snprintf(out + pos, 32768 - pos,
                                    "],\"download\":{\"active\":%s,\"finished\":%d,\"model_id\":\"%s\",\"repo\":\"%s\","
                                    "\"cur_file\":\"%s\",\"file_idx\":%d,\"file_total\":%d,"
                                    "\"bytes_done\":%llu,\"bytes_total\":%llu,\"speed_mbs\":%.2f,"
                                    "\"stage\":\"%s\",\"error\":\"%s\"}}",
                                    dl.active ? "true" : "false", dl.finished, dlm_e, dlr_e, dlf_e,
                                    dl.file_idx, dl.file_total, (unsigned long long)dl.bytes_done,
                                    (unsigned long long)dl.bytes_total, dl.speed_mbs, dls_e, dle_e);
            send_resp(fd, 200, "OK", "application/json", out, pos);
        } else {
            send_resp(fd, 500, "Internal Server Error", "text/plain", "oom", 3);
        }
        free(inst); free(cat); free(out);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/models/repo") == 0) {
        char repo[128] = {0};
        json_get_str(body, "repo", repo, sizeof(repo));
        model_dl_set_repo(repo);
        send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/models/catalog/refresh") == 0) {
        int n = model_dl_refresh_catalog();
        char resp[96];
        int l = snprintf(resp, sizeof(resp), "{\"ok\":%s,\"count\":%d}", n > 0 ? "true" : "false", n);
        send_resp(fd, 200, "OK", "application/json", resp, (size_t)l);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/models/download") == 0) {
        char id[64] = {0}, repo[128] = {0}, files[256] = {0};
        json_get_str(body, "id", id, sizeof(id));
        json_get_str(body, "repo", repo, sizeof(repo));
        json_get_str(body, "files", files, sizeof(files));
        int rc = model_dl_start(id, repo, files);
        if (rc == 0) {
            send_resp(fd, 202, "Accepted", "application/json", "{\"ok\":true}", 11);
        } else {
            ModelDlStatus dl;
            model_dl_get_status(&dl);
            char ee[320], resp[420];
            json_escape(dl.error[0] ? dl.error : (rc == -2 ? "Another download is already running" : "Invalid model ID"), ee, sizeof(ee));
            int l = snprintf(resp, sizeof(resp), "{\"error\":\"%s\"}", ee);
            send_resp(fd, rc == -2 ? 503 : 400, "Error", "application/json", resp, (size_t)l);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/models/download/cancel") == 0) {
        model_dl_cancel();
        send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/models/delete") == 0) {
        char id[64] = {0}, act[64] = {0};
        json_get_str(body, "id", id, sizeof(id));
        http_srv_lock();
        snprintf(act, sizeof(act), "%s", g_st->model);
        http_srv_unlock();
        int rc = model_dl_delete(id, act);
        if (rc == 0) {
            http_srv_lock(); g_job->rescan_req = 1; http_srv_unlock();
            send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
        } else if (rc == -2) {
            send_resp(fd, 409, "Conflict", "application/json", "{\"error\":\"Switch away from active model before deleting\"}", 57);
        } else {
            send_resp(fd, 404, "Not Found", "application/json", "{\"error\":\"model not found\"}", 27);
        }
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/loras/manager") == 0) {
        InstalledLora* inst = (InstalledLora*)malloc(sizeof(InstalledLora) * LORA_MAX_INSTALLED);
        CatalogLora* cat = (CatalogLora*)malloc(sizeof(CatalogLora) * LORA_MAX_CATALOG);
        char* out = (char*)malloc(40960);
        if (inst && cat && out) {
            int ni = lora_list_installed(inst, LORA_MAX_INSTALLED);
            int nc = lora_get_catalog(cat, LORA_MAX_CATALOG);
            ModelDlStatus dl;
            model_dl_get_status(&dl);
            int is15 = 0;
            char cur_l[80] = {0};
            float cur_ls = 0.8f;
            http_srv_lock();
            is15 = g_st->is_sd15;
            snprintf(cur_l, sizeof(cur_l), "%s", g_job->lora);
            cur_ls = g_job->lora_scale;
            http_srv_unlock();

            char cle[128], dlm_e[128], dlr_e[256], dlf_e[128], dls_e[256], dle_e[384];
            json_escape(cur_l, cle, sizeof(cle));
            json_escape(dl.model_id, dlm_e, sizeof(dlm_e));
            json_escape(dl.repo, dlr_e, sizeof(dlr_e));
            json_escape(dl.cur_file, dlf_e, sizeof(dlf_e));
            json_escape(dl.stage, dls_e, sizeof(dls_e));
            json_escape(dl.error, dle_e, sizeof(dle_e));

            size_t pos = 0;
            pos += (size_t)snprintf(out + pos, 40960 - pos,
                                    "{\"active_arch\":\"%s\",\"selected_lora\":\"%s\",\"lora_scale\":%.2f,\"installed\":[",
                                    is15 ? "sd15" : "sdxl", cle, (double)cur_ls);
            for (int i = 0; i < ni && pos + 320 < 40960; i++) {
                char ne[128], fe[160];
                json_escape(inst[i].name, ne, sizeof(ne));
                json_escape(inst[i].filename, fe, sizeof(fe));
                pos += (size_t)snprintf(out + pos, 40960 - pos,
                                        "%s{\"name\":\"%s\",\"filename\":\"%s\",\"arch\":\"%s\",\"size_mb\":%u,\"rank\":%u,\"has_te\":%s}",
                                        i ? "," : "", ne, fe, inst[i].arch, inst[i].size_mb, inst[i].rank,
                                        inst[i].has_te ? "true" : "false");
            }
            pos += (size_t)snprintf(out + pos, 40960 - pos, "],\"catalog\":[");
            for (int i = 0; i < nc && pos + 800 < 40960; i++) {
                char ide[128], te[160], tre[192], de[320], fe[160], ue[512];
                json_escape(cat[i].id, ide, sizeof(ide));
                json_escape(cat[i].title, te, sizeof(te));
                json_escape(cat[i].trigger, tre, sizeof(tre));
                json_escape(cat[i].desc, de, sizeof(de));
                json_escape(cat[i].filename, fe, sizeof(fe));
                json_escape(cat[i].url, ue, sizeof(ue));
                int qd = lora_dl_is_queued(cat[i].filename) || lora_dl_is_queued(cat[i].id);
                pos += (size_t)snprintf(out + pos, 40960 - pos,
                                        "%s{\"id\":\"%s\",\"title\":\"%s\",\"arch\":\"%s\",\"size_mb\":%u,"
                                        "\"default_weight\":%.2f,\"trigger\":\"%s\",\"desc\":\"%s\","
                                        "\"filename\":\"%s\",\"url\":\"%s\",\"queued\":%s}",
                                        i ? "," : "", ide, te, cat[i].arch, cat[i].size_mb,
                                        (double)cat[i].default_weight, tre, de, fe, ue, qd ? "true" : "false");
            }
            pos += (size_t)snprintf(out + pos, 40960 - pos,
                                    "],\"download\":{\"active\":%s,\"finished\":%d,\"queue_len\":%d,\"model_id\":\"%s\",\"repo\":\"%s\","
                                    "\"cur_file\":\"%s\",\"file_idx\":%d,\"file_total\":%d,"
                                    "\"bytes_done\":%llu,\"bytes_total\":%llu,\"speed_mbs\":%.2f,"
                                    "\"stage\":\"%s\",\"error\":\"%s\"}}",
                                    dl.active ? "true" : "false", dl.finished, dl.queue_len, dlm_e, dlr_e, dlf_e,
                                    dl.file_idx, dl.file_total, (unsigned long long)dl.bytes_done,
                                    (unsigned long long)dl.bytes_total, dl.speed_mbs, dls_e, dle_e);
            send_resp(fd, 200, "OK", "application/json", out, pos);
        } else {
            send_resp(fd, 500, "Internal Server Error", "text/plain", "oom", 3);
        }
        free(inst); free(cat); free(out);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/loras/catalog/refresh") == 0) {
        int n = lora_refresh_catalog();
        char resp[96];
        int l = snprintf(resp, sizeof(resp), "{\"ok\":%s,\"count\":%d}", n > 0 ? "true" : "false", n);
        send_resp(fd, 200, "OK", "application/json", resp, (size_t)l);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/loras/download") == 0) {
        char id[64] = {0}, url[384] = {0}, fn[80] = {0};
        json_get_str(body, "id", id, sizeof(id));
        json_get_str(body, "url", url, sizeof(url));
        json_get_str(body, "filename", fn, sizeof(fn));
        int rc = lora_dl_start(id, url, fn);
        if (rc == 0 || rc == 1) {
            char resp[64];
            int l = snprintf(resp, sizeof(resp), "{\"ok\":true,\"queued\":%s}", rc == 1 ? "true" : "false");
            send_resp(fd, 202, "Accepted", "application/json", resp, (size_t)l);
        } else {
            ModelDlStatus dl;
            model_dl_get_status(&dl);
            char ee[320], resp[420];
            json_escape(dl.error[0] ? dl.error : (rc == -2 ? "Another model download is already running" : "Invalid LoRA URL"), ee, sizeof(ee));
            int l = snprintf(resp, sizeof(resp), "{\"error\":\"%s\"}", ee);
            send_resp(fd, rc == -2 ? 503 : 400, "Error", "application/json", resp, (size_t)l);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/loras/download_all") == 0) {
        char arch[16] = {0};
        json_get_str(body, "arch", arch, sizeof(arch));
        if (!arch[0]) {
            http_srv_lock();
            snprintf(arch, sizeof(arch), "%s", g_st->is_sd15 ? "sd15" : "sdxl");
            http_srv_unlock();
        }
        int rc = lora_dl_queue_all(arch);
        if (rc >= 0) {
            char resp[96];
            int l = snprintf(resp, sizeof(resp), "{\"ok\":true,\"queued\":%d,\"arch\":\"%s\"}", rc, arch);
            send_resp(fd, 202, "Accepted", "application/json", resp, (size_t)l);
        } else {
            send_resp(fd, rc == -2 ? 503 : 400, "Error", "application/json",
                      "{\"error\":\"Another model download is already running\"}", 55);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/loras/delete") == 0) {
        char id[80] = {0};
        json_get_str(body, "id", id, sizeof(id));
        if (lora_delete(id) == 0) {
            http_srv_lock();
            if (strcmp(g_job->lora, id) == 0) g_job->lora[0] = 0;
            g_job->loras_dirty = 1;
            http_srv_unlock();
            send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
        } else {
            send_resp(fd, 404, "Not Found", "application/json", "{\"error\":\"lora not found\"}", 26);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/loras/select") == 0) {
        char lr[80] = {0};
        double ls = 0.8;
        json_get_str(body, "lora", lr, sizeof(lr));
        int has_s = json_get_double(body, "lora_scale", &ls);
        if (!strchr(lr, '/') && !strchr(lr, '\\') && !strstr(lr, "..")) {
            // Reject incompatible architecture LoRAs so they cannot be selected for the active model
            int is15 = 0;
            http_srv_lock(); is15 = g_st->is_sd15; http_srv_unlock();
            if (lr[0]) {
                InstalledLora* inst = (InstalledLora*)malloc(sizeof(InstalledLora) * LORA_MAX_INSTALLED);
                int incompatible = 0;
                char bad_arch[16] = {0};
                if (inst) {
                    int ni = lora_list_installed(inst, LORA_MAX_INSTALLED);
                    const char* act_arch = is15 ? "sd15" : "sdxl";
                    for (int i = 0; i < ni; i++) {
                        if (strcmp(inst[i].name, lr) == 0) {
                            if (inst[i].arch[0] && strcmp(inst[i].arch, "any") != 0 && strcmp(inst[i].arch, act_arch) != 0) {
                                incompatible = 1;
                                snprintf(bad_arch, sizeof(bad_arch), "%s", inst[i].arch);
                            }
                            break;
                        }
                    }
                    free(inst);
                }
                if (incompatible) {
                    char resp[192];
                    int l = snprintf(resp, sizeof(resp),
                                     "{\"error\":\"Incompatible LoRA (%s) for active %s model\"}",
                                     bad_arch, is15 ? "SD 1.5" : "SDXL");
                    send_resp(fd, 400, "Bad Request", "application/json", resp, (size_t)l);
                    return;
                }
            }
            http_srv_lock();
            snprintf(g_job->lora, sizeof(g_job->lora), "%s", lr);
            if (has_s) {
                if (ls < -2.0) ls = -2.0;
                if (ls > 2.5) ls = 2.5;
                g_job->lora_scale = (float)ls;
            }
            g_st->settings_seq++;
            http_srv_unlock();
            send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
        } else {
            send_resp(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid lora\"}", 24);
        }
    } else if (strcmp(method, "GET") == 0 && (strcmp(path, "/status") == 0 || strcmp(path, "/stats") == 0)) {
        char* out = (char*)malloc(16384);
        char* cpr_esc = (char*)malloc(1600);
        char* cng_esc = (char*)malloc(900);
        if (!out || !cpr_esc || !cng_esc) {
            free(out); free(cpr_esc); free(cng_esc);
            send_resp(fd, 500, "Internal Server Error", "text/plain", "oom", 3);
            free(req);
            return;
        }
        char st_esc[256], err_esc[512], mdl_esc[128], pd_esc[192], sd_esc[192], pp_esc[768], pn_esc[512], lr_esc[128], clr_esc[128], alr_esc[192], cfld_esc[96], mlist[2560];
        static const char* SMP_NAMES[4] = {"Euler a", "Euler", "LCM", "DPM++ 2M Karras"};
        uint32_t grev = gallery_revision();
        AppUpdateStatus upd;
        app_update_get_status(&upd);
        ModelDlStatus udl;
        model_dl_get_status(&udl);
        int ota_dl_active = (udl.active && strcmp(udl.model_id, "Garlic-Diffusion.pkg") == 0);
        int ota_dl_pct = (ota_dl_active && udl.bytes_total > 0) ? (int)((100ull * udl.bytes_done) / udl.bytes_total) : 0;
        char urem_esc[32], uchg_esc[384], ustg_esc[220];
        json_escape(upd.remote_ver, urem_esc, sizeof(urem_esc));
        json_escape(upd.changelog, uchg_esc, sizeof(uchg_esc));
        json_escape(udl.error[0] ? udl.error : udl.stage, ustg_esc, sizeof(ustg_esc));

        GalleryStageStatus stg;
        gallery_stage_get_status(&stg);
        static const char* const STG_NAMES[5] = {"idle", "staging", "flushing", "complete", "error"};
        const char* stg_name = STG_NAMES[(stg.state >= 0 && stg.state <= 4) ? stg.state : 0];
        char stg_err_esc[256];
        json_escape(stg.error, stg_err_esc, sizeof(stg_err_esc));

        http_srv_lock();
        json_escape(g_st->stage, st_esc, sizeof(st_esc));
        json_escape(g_st->error, err_esc, sizeof(err_esc));
        json_escape(g_st->model, mdl_esc, sizeof(mdl_esc));
        json_escape(g_st->preset_desc, pd_esc, sizeof(pd_esc));
        json_escape(g_st->speed_desc, sd_esc, sizeof(sd_esc));
        json_escape(g_st->preset_prompt, pp_esc, sizeof(pp_esc));
        json_escape(g_st->preset_negative, pn_esc, sizeof(pn_esc));
        json_escape(g_job->lora, lr_esc, sizeof(lr_esc));
        json_escape(g_st->active_lora, alr_esc, sizeof(alr_esc));
        float lscale = g_job->lora_scale;
        int is_switching = g_job->switch_req;
        int nsfw_ok = g_job->nsfw_allow;
        int auto_q = g_job->auto_quality;
        int auto_ff = g_job->auto_face_fix;
        int ui_m = g_job->ui_mode;

        // Expose active in-progress job settings when busy, or live Studio configuration when idle!
        int use_cur = (g_st->busy && g_st->cur_w > 0);
        json_escape(use_cur ? g_st->cur_prompt : g_job->prompt, cpr_esc, 1600);
        json_escape(use_cur ? g_st->cur_negative : g_job->negative, cng_esc, 900);
        json_escape(use_cur ? g_st->cur_lora : g_job->lora, clr_esc, sizeof(clr_esc));
        json_escape(use_cur ? g_st->cur_folder : g_job->batch_folder, cfld_esc, sizeof(cfld_esc));
        float cls = use_cur ? g_st->cur_lora_scale : g_job->lora_scale;
        uint32_t cw = use_cur ? g_st->cur_w : g_job->width;
        uint32_t ch = use_cur ? g_st->cur_h : g_job->height;
        uint32_t cst = use_cur ? g_st->cur_steps : g_job->steps;
        float ccfg = use_cur ? g_st->cur_cfg : g_job->cfg;
        int csmp = use_cur ? g_st->cur_sampler : g_job->sampler;
        if (csmp < 0 || csmp > 3) csmp = 0;
        int cups = use_cur ? g_st->cur_upscale : g_job->upscale;
        int ceco = use_cur ? g_st->cur_eco : g_job->eco_mode;
        uint64_t cseed = use_cur ? g_st->cur_seed : g_job->seed;
        uint32_t cbatch = use_cur ? g_st->cur_batch : (g_job->batch_count ? g_job->batch_count : 1u);
        uint32_t crand = use_cur ? g_st->cur_rand_mask : g_job->batch_rand_mask;

        size_t mp = 0;
        mlist[mp++] = '[';
        for (int i = 0; i < g_st->num_models && mp + 80 < sizeof(mlist); i++) {
            char me[128];
            json_escape(g_st->models[i], me, sizeof(me));
            mp += (size_t)snprintf(mlist + mp, sizeof(mlist) - mp, "%s\"%s\"", i ? "," : "", me);
        }
        mlist[mp++] = ']';
        mlist[mp] = 0;
        int ps = (g_st->preset_sampler >= 0 && g_st->preset_sampler <= 3) ? g_st->preset_sampler : 0;
        int sps = (g_st->speed_sampler >= 0 && g_st->speed_sampler <= 3) ? g_st->speed_sampler : 0;
        int n = snprintf(out, 16384,
                         "{\"version\":\"%s\",\"ui_mode\":%d,"
                         "\"gpu_ready\":%s,\"ready\":%s,\"busy\":%s,\"switching\":%s,\"neo\":%d,\"arch\":\"%s\",\"has_upscaler\":%s,"
                         "\"nsfw_allow\":%s,\"auto_quality\":%s,\"auto_face_fix\":%s,"
                         "\"temp_c\":%d,\"temp_peak_c\":%d,\"eco_mode\":%d,"
                         "\"gpu_used_mb\":%u,\"gpu_total_mb\":%u,\"progress\":%.3f,\"eta_sec\":%.1f,\"elapsed_sec\":%.1f,"
                         "\"stage\":\"%s\",\"model\":\"%s\","
                         "\"lora\":\"%s\",\"lora_scale\":%.2f,\"active_lora\":\"%s\",\"gallery_rev\":%u,\"settings_seq\":%u,"
                         "\"models\":%s,\"preset\":{\"width\":%u,\"height\":%u,\"steps\":%u,\"cfg\":%.2f,"
                         "\"sampler\":\"%s\",\"upscale\":%d,\"description\":\"%s\",\"prompt\":\"%s\",\"negative\":\"%s\"},"
                         "\"speed_preset\":{\"width\":%u,\"height\":%u,\"steps\":%u,\"cfg\":%.2f,"
                         "\"sampler\":\"%s\",\"upscale\":%d,\"description\":\"%s\"},"
                         "\"current\":{\"prompt\":\"%s\",\"negative\":\"%s\",\"lora\":\"%s\",\"lora_scale\":%.2f,"
                         "\"width\":%u,\"height\":%u,\"steps\":%u,\"cfg\":%.2f,\"sampler\":\"%s\",\"upscale\":%d,"
                         "\"eco\":%d,\"seed\":%llu,\"batch_count\":%u,\"rand_mask\":%u,\"folder\":\"%s\",\"shutdown_after\":%s},"
                         "\"update\":{\"checked\":%s,\"available\":%s,\"pkg_ready\":%s,\"current\":\"%s\",\"remote\":\"%s\","
                         "\"size_mb\":%u,\"changelog\":\"%s\",\"dl_active\":%s,\"dl_pct\":%d,\"dl_speed\":%.2f,\"dl_stage\":\"%s\"},"
                         "\"error\":\"%s\",\"width\":%u,\"height\":%u,\"steps\":%u,\"cfg\":%.2f,\"seed\":%llu,"
                         "\"last_sec\":%.2f,\"unet_gpu_sec\":%.2f,\"vae_sec\":%.2f,\"upscale_sec\":%.2f,\"face_fix_sec\":%.2f,\"cooldown_sec\":%.2f,"
                         "\"eff_tflops\":%.3f,\"bench_gemm_tflops\":%.3f,\"bench_attn_ms\":%.2f,"
                         "\"bench_mode_tf\":[%.3f,%.3f,%.3f,%.3f],"
                         "\"has_image\":%s,\"image_gen_id\":%u,\"batch_cur\":%u,\"batch_total\":%u,\"shutdown_after\":%s,\"last_gallery_id\":\"%s\","
                         "\"storage\":{\"state\":\"%s\",\"queued\":%u,\"flushed\":%u,\"used_mb\":%.2f,\"max_mb\":%.1f,\"fallback\":%s,\"last_id\":\"%s\",\"error\":\"%s\"}}",
                         GD_APP_VERSION, ui_m,
                         g_st->gpu_ready ? "true" : "false", g_st->ready ? "true" : "false", g_st->busy ? "true" : "false",
                         is_switching ? "true" : "false",
                         g_st->neo, g_st->is_sd15 ? "sd15" : "sdxl", g_st->has_upscaler ? "true" : "false",
                         nsfw_ok ? "true" : "false", auto_q ? "true" : "false", auto_ff ? "true" : "false",
                         g_st->temp_c, g_st->temp_peak_c, g_st->eco_mode,
                         g_st->gpu_used_mb, g_st->gpu_total_mb,
                         (double)g_st->progress, (double)g_st->eta_sec, (double)g_st->elapsed_sec,
                         st_esc, mdl_esc,
                         lr_esc, (double)lscale, alr_esc, grev, g_st->settings_seq, mlist,
                         g_st->preset_w, g_st->preset_h, g_st->preset_steps, (double)g_st->preset_cfg,
                         SMP_NAMES[ps], g_st->preset_upscale, pd_esc, pp_esc, pn_esc,
                         g_st->speed_w, g_st->speed_h, g_st->speed_steps, (double)g_st->speed_cfg,
                         SMP_NAMES[sps], g_st->speed_upscale, sd_esc,
                         cpr_esc, cng_esc, clr_esc, (double)cls, cw, ch, cst, (double)ccfg, SMP_NAMES[csmp],
                         cups, ceco, (unsigned long long)cseed, cbatch, crand, cfld_esc, g_job->shutdown_after ? "true" : "false",
                         upd.checked ? "true" : "false", upd.update_available ? "true" : "false", upd.pkg_ready ? "true" : "false",
                         upd.current_ver, urem_esc, upd.size_mb, uchg_esc,
                         ota_dl_active ? "true" : "false", ota_dl_pct, udl.speed_mbs, ustg_esc,
                         err_esc, g_st->width, g_st->height, g_st->steps, (double)g_st->cfg,
                         (unsigned long long)g_st->seed, g_st->last_sec,
                         g_st->unet_gpu_sec, g_st->vae_sec, g_st->upscale_sec, g_st->face_fix_sec, g_st->cooldown_sec,
                         g_st->eff_tflops, g_st->bench_gemm_tflops, g_st->bench_attn_ms,
                         g_st->bench_mode_tf[0], g_st->bench_mode_tf[1],
                         g_st->bench_mode_tf[2], g_st->bench_mode_tf[3],
                         (g_st->png_data && g_st->png_size) ? "true" : "false", g_st->image_gen_id,
                         g_st->batch_cur, g_st->batch_total, g_st->shutdown_after ? "true" : "false", g_st->last_gallery_id,
                         stg_name, stg.queued_count, stg.flushed_total,
                         (double)stg.used_bytes / 1048576.0, (double)stg.max_bytes / 1048576.0,
                         stg.fallback_used ? "true" : "false", stg.last_id, stg_err_esc);
        http_srv_unlock();
        send_resp(fd, 200, "OK", "application/json", out, (size_t)n);
        free(out); free(cpr_esc); free(cng_esc);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/image.png") == 0) {
        uint8_t* copy = NULL;
        size_t sz = 0;
        http_srv_lock();
        if (g_st->png_data && g_st->png_size) {
            sz = g_st->png_size;
            copy = (uint8_t*)malloc(sz);
            if (copy) memcpy(copy, g_st->png_data, sz);
        }
        http_srv_unlock();
        if (copy) {
            send_resp(fd, 200, "OK", "image/png", copy, sz);
            free(copy);
        } else {
            send_resp(fd, 404, "Not Found", "text/plain", "no image yet", 12);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/settings") == 0) {
        double v = 0;
        char tmp[64] = {0};
        int save_uim = -99, nsfw_changed = 0;
        http_srv_lock();
        int prev_ns = g_job->nsfw_allow;
        if (json_get_double(body, "nsfw_allow", &v)) g_job->nsfw_allow = (v != 0) ? 1 : 0;
        else if (strstr(body, "\"nsfw_allow\":true") || strstr(body, "\"nsfw_allow\": true")) g_job->nsfw_allow = 1;
        else if (strstr(body, "\"nsfw_allow\":false") || strstr(body, "\"nsfw_allow\": false")) g_job->nsfw_allow = 0;
        nsfw_changed = (g_job->nsfw_allow != prev_ns);
        if (nsfw_changed) g_job->presets_dirty = 1;
        if (json_get_double(body, "auto_quality", &v)) g_job->auto_quality = (v != 0) ? 1 : 0;
        else if (strstr(body, "\"auto_quality\":true") || strstr(body, "\"auto_quality\": true")) g_job->auto_quality = 1;
        else if (strstr(body, "\"auto_quality\":false") || strstr(body, "\"auto_quality\": false")) g_job->auto_quality = 0;
        if (json_get_double(body, "auto_face_fix", &v)) g_job->auto_face_fix = (v != 0) ? 1 : 0;
        else if (strstr(body, "\"auto_face_fix\":true") || strstr(body, "\"auto_face_fix\": true")) g_job->auto_face_fix = 1;
        else if (strstr(body, "\"auto_face_fix\":false") || strstr(body, "\"auto_face_fix\": false")) g_job->auto_face_fix = 0;
        if (json_get_double(body, "ui_mode", &v)) {
            int nm = (int)v;
            if (nm == 0 || nm == 1) {
                g_job->ui_mode = nm;
                g_st->ui_mode = nm;
                save_uim = nm;
            }
        }
        if (json_get_double(body, "rand_mask", &v) || json_get_double(body, "batch_rand_mask", &v))
            g_job->batch_rand_mask = (v > 0) ? (uint32_t)v : 0u;
        if (json_get_str(body, "folder", tmp, sizeof(tmp)) || json_get_str(body, "batch_folder", tmp, sizeof(tmp)))
            gallery_clean_folder_name(tmp, g_job->batch_folder, sizeof(g_job->batch_folder));
        g_st->nsfw_allow = g_job->nsfw_allow;
        g_st->auto_quality = g_job->auto_quality;
        g_st->auto_face_fix = g_job->auto_face_fix;
        g_st->cur_rand_mask = g_job->batch_rand_mask;
        g_st->settings_seq++;
        int ns = g_job->nsfw_allow, aq = g_job->auto_quality, aff = g_job->auto_face_fix, uim = g_job->ui_mode;
        uint32_t sseq = g_st->settings_seq, rmk = g_job->batch_rand_mask;
        http_srv_unlock();
        if (save_uim == 0 || save_uim == 1) save_ui_mode_file(save_uim);
        char resp[256];
        int l = snprintf(resp, sizeof(resp),
                         "{\"ok\":true,\"nsfw_allow\":%s,\"auto_quality\":%s,\"auto_face_fix\":%s,\"ui_mode\":%d,\"rand_mask\":%u,\"settings_seq\":%u}",
                         ns ? "true" : "false", aq ? "true" : "false", aff ? "true" : "false", uim, rmk, sseq);
        send_resp(fd, 200, "OK", "application/json", resp, (size_t)l);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/update/status") == 0) {
        AppUpdateStatus upd;
        app_update_get_status(&upd);
        char rem_e[32], url_e[384], chg_e[384], out[1024];
        json_escape(upd.remote_ver, rem_e, sizeof(rem_e));
        json_escape(upd.pkg_url, url_e, sizeof(url_e));
        json_escape(upd.changelog, chg_e, sizeof(chg_e));
        int l = snprintf(out, sizeof(out),
                         "{\"checked\":%s,\"available\":%s,\"pkg_ready\":%s,\"current\":\"%s\","
                         "\"remote\":\"%s\",\"pkg_url\":\"%s\",\"size_mb\":%u,\"changelog\":\"%s\"}",
                         upd.checked ? "true" : "false", upd.update_available ? "true" : "false",
                         upd.pkg_ready ? "true" : "false", upd.current_ver, rem_e, url_e, upd.size_mb, chg_e);
        send_resp(fd, 200, "OK", "application/json", out, (size_t)l);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/update/check") == 0) {
        int rc = app_update_check_now();
        AppUpdateStatus upd;
        app_update_get_status(&upd);
        char rem_e[32], chg_e[384], out[768];
        json_escape(upd.remote_ver, rem_e, sizeof(rem_e));
        json_escape(upd.changelog, chg_e, sizeof(chg_e));
        int l = snprintf(out, sizeof(out),
                         "{\"ok\":%s,\"checked\":%s,\"available\":%s,\"pkg_ready\":%s,\"current\":\"%s\",\"remote\":\"%s\",\"changelog\":\"%s\"}",
                         rc == 0 ? "true" : "false", upd.checked ? "true" : "false",
                         upd.update_available ? "true" : "false", upd.pkg_ready ? "true" : "false",
                         upd.current_ver, rem_e, chg_e);
        send_resp(fd, 200, "OK", "application/json", out, (size_t)l);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/update/download") == 0) {
        int rc = app_update_download_pkg();
        if (rc == 0) {
            send_resp(fd, 202, "Accepted", "application/json", "{\"ok\":true,\"dest\":\"" OTA_PKG_DEST "\"}", 44);
        } else {
            ModelDlStatus dl;
            model_dl_get_status(&dl);
            char ee[256], resp[320];
            json_escape(dl.error[0] ? dl.error : (rc == -2 ? "Another download is already running" : "Failed to start OTA download"), ee, sizeof(ee));
            int l = snprintf(resp, sizeof(resp), "{\"error\":\"%s\"}", ee);
            send_resp(fd, rc == -2 ? 503 : 400, "Error", "application/json", resp, (size_t)l);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/update/install_exit") == 0) {
        char msg[256] = {0}, me[320], resp[384];
        int rc = app_update_prepare_and_exit(msg, sizeof(msg));
        json_escape(msg, me, sizeof(me));
        int l = snprintf(resp, sizeof(resp), "{\"ok\":%s,\"message\":\"%s\"}", rc == 0 ? "true" : "false", me);
        send_resp(fd, rc == 0 ? 200 : 400, rc == 0 ? "OK" : "Bad Request", "application/json", resp, (size_t)l);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/shutdown_after") == 0) {
        double v = 0;
        int en = 0;
        if (json_get_double(body, "enabled", &v)) en = (v != 0) ? 1 : 0;
        else if (strstr(body, "\"enabled\":true") || strstr(body, "\"enabled\": true")) en = 1;
        http_srv_lock();
        g_job->shutdown_after = en;
        g_st->shutdown_after = en;
        g_st->settings_seq++;
        http_srv_unlock();
        send_resp(fd, 200, "OK", "application/json", en ? "{\"shutdown_after\":true}" : "{\"shutdown_after\":false}", en ? 23 : 24);
    } else if (strcmp(method, "POST") == 0 &&
               (strcmp(path, "/cancel") == 0 || strcmp(path, "/sdapi/v1/interrupt") == 0)) {
        http_srv_lock();
        g_job->cancel_req = 1;
        http_srv_unlock();
        send_resp(fd, 200, "OK", "application/json", "{}", 2);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/models/rescan") == 0) {
        http_srv_lock();
        g_job->rescan_req = 1;
        http_srv_unlock();
        for (int w = 0; w < 30; w++) {
            sceKernelUsleep(25000);
            http_srv_lock();
            int done = !g_job->rescan_req;
            http_srv_unlock();
            if (done) break;
        }
        send_resp(fd, 200, "OK", "application/json", "{\"ok\":true}", 11);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/model") == 0) {
        char req_m[64] = {0};
        if (json_get_str(body, "model", req_m, sizeof(req_m)) && req_m[0] &&
            !strchr(req_m, '/') && !strchr(req_m, '\\') && !strstr(req_m, "..")) {
            http_srv_lock();
            int ok = g_st->gpu_ready && !g_st->busy && !g_job->pending && !g_job->switch_req;
            if (ok) {
                snprintf(g_job->switch_model, sizeof(g_job->switch_model), "%s", req_m);
                g_job->switch_req = 1;
            }
            http_srv_unlock();
            if (ok) send_resp(fd, 202, "Accepted", "application/json", "{\"ok\":true}", 11);
            else send_resp(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"busy\"}", 16);
        } else {
            send_resp(fd, 400, "Bad Request", "application/json", "{\"error\":\"missing model\"}", 25);
        }
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/generate") == 0) {
        http_srv_lock();
        int ok = g_st->ready && !g_st->busy && !g_job->pending && !g_job->switch_req;
        if (ok) {
            apply_job_json(body, g_job);
            g_job->cancel_req = 0;
            g_job->pending = 1;
            g_st->settings_seq++;
        }
        http_srv_unlock();
        if (ok) send_resp(fd, 202, "Accepted", "application/json", "{\"ok\":true}", 11);
        else send_resp(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"busy\"}", 16);
    // ---- Automatic1111 (A1111) compatible endpoints ----
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/sdapi/v1/sd-models") == 0) {
        char* out = (char*)malloc(12288);
        if (!out) { send_resp(fd, 500, "Internal Server Error", "text/plain", "oom", 3); free(req); return; }
        size_t pos = 0;
        out[pos++] = '[';
        http_srv_lock();
        int nm = g_st->num_models;
        for (int i = 0; i < nm && pos + 280 < 12288; i++) {
            char me[128];
            json_escape(g_st->models[i], me, sizeof(me));
            pos += (size_t)snprintf(out + pos, 12288 - pos,
                                    "%s{\"title\":\"%s\",\"model_name\":\"%s\",\"hash\":\"0000000000\","
                                    "\"sha256\":null,\"filename\":\"/data/garlic-diffusion/models/%s\",\"config\":null}",
                                    i ? "," : "", me, me, me);
        }
        if (nm == 0) {
            char me[128];
            json_escape(g_st->model[0] ? g_st->model : "waiNSFWIllustrious_v140", me, sizeof(me));
            pos += (size_t)snprintf(out + pos, 12288 - pos,
                                    "{\"title\":\"%s\",\"model_name\":\"%s\",\"hash\":\"0000000000\","
                                    "\"sha256\":null,\"filename\":\"/data/garlic-diffusion/models/%s\",\"config\":null}",
                                    me, me, me);
        }
        http_srv_unlock();
        out[pos++] = ']';
        out[pos] = 0;
        send_resp(fd, 200, "OK", "application/json", out, pos);
        free(out);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/sdapi/v1/samplers") == 0) {
        const char* s = "[{\"name\":\"Euler a\",\"aliases\":[\"euler_a\",\"k_euler_a\"],\"options\":{}},"
                        "{\"name\":\"Euler\",\"aliases\":[\"euler\",\"k_euler\"],\"options\":{}},"
                        "{\"name\":\"LCM\",\"aliases\":[\"lcm\"],\"options\":{}},"
                        "{\"name\":\"DPM++ 2M Karras\",\"aliases\":[\"dpmpp_2m\",\"k_dpmpp_2m\"],\"options\":{\"scheduler\":\"karras\"}}]";
        send_resp(fd, 200, "OK", "application/json", s, strlen(s));
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/sdapi/v1/schedulers") == 0) {
        const char* s = "[{\"name\":\"Automatic\",\"label\":\"Automatic\",\"aliases\":null,\"default_rho\":-1,\"need_inner_model\":false}]";
        send_resp(fd, 200, "OK", "application/json", s, strlen(s));
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/sdapi/v1/upscalers") == 0) {
        const char* s = "[{\"name\":\"None\",\"model_name\":null,\"model_path\":null,\"model_url\":null,\"scale\":1},"
                        "{\"name\":\"RealESRGAN Anime v3 (1080p)\",\"model_name\":\"realesr-animevideov3\",\"model_path\":null,\"model_url\":null,\"scale\":2},"
                        "{\"name\":\"RealESRGAN Anime v3 (2x)\",\"model_name\":\"realesr-animevideov3\",\"model_path\":null,\"model_url\":null,\"scale\":2},"
                        "{\"name\":\"RealESRGAN Anime v3 (4K UHD)\",\"model_name\":\"realesr-animevideov3\",\"model_path\":null,\"model_url\":null,\"scale\":4}]";
        send_resp(fd, 200, "OK", "application/json", s, strlen(s));
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/sdapi/v1/scripts") == 0) {
        const char* s = "{\"txt2img\":[],\"img2img\":[]}";
        send_resp(fd, 200, "OK", "application/json", s, strlen(s));
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/sdapi/v1/prompt-styles") == 0) {
        UserPreset* items = (UserPreset*)malloc(sizeof(UserPreset) * PRESETS_MAX_ITEMS);
        if (!items) { send_resp(fd, 200, "OK", "application/json", "[]", 2); free(req); return; }
        int cnt = presets_list(items, PRESETS_MAX_ITEMS);
        int allow_nsfw = 0;
        http_srv_lock();
        allow_nsfw = g_job->nsfw_allow;
        http_srv_unlock();
        size_t cap = (size_t)(cnt + 1) * 2000u + 64u;
        char* out = (char*)malloc(cap);
        if (out) {
            size_t pos = 0;
            int emitted = 0;
            out[pos++] = '[';
            for (int i = 0; i < cnt && pos + 1800u < cap; i++) {
                if (items[i].nsfw && !allow_nsfw) continue;
                char nm[128], pr[1200], ng[600];
                json_escape(items[i].name, nm, sizeof(nm));
                json_escape(items[i].prompt, pr, sizeof(pr));
                json_escape(items[i].negative, ng, sizeof(ng));
                pos += (size_t)snprintf(out + pos, cap - pos,
                                        "%s{\"name\":\"%s\",\"prompt\":\"%s\",\"negative_prompt\":\"%s\"}",
                                        emitted ? "," : "", nm, pr, ng);
                emitted++;
            }
            out[pos++] = ']';
            out[pos] = 0;
            send_resp(fd, 200, "OK", "application/json", out, pos);
            free(out);
        }
        free(items);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/sdapi/v1/loras") == 0) {
        InstalledLora* inst = (InstalledLora*)malloc(sizeof(InstalledLora) * LORA_MAX_INSTALLED);
        char* out = (char*)malloc(16384);
        if (inst && out) {
            int ni = lora_list_installed(inst, LORA_MAX_INSTALLED);
            size_t pos = 0;
            out[pos++] = '[';
            for (int i = 0; i < ni && pos + 320 < 16384; i++) {
                char ne[128], fe[160];
                json_escape(inst[i].name, ne, sizeof(ne));
                json_escape(inst[i].filename, fe, sizeof(fe));
                pos += (size_t)snprintf(out + pos, 16384 - pos,
                                        "%s{\"name\":\"%s\",\"alias\":\"%s\",\"path\":\"/data/garlic-diffusion/loras/%s\","
                                        "\"metadata\":{\"arch\":\"%s\"}}",
                                        i ? "," : "", ne, ne, fe, inst[i].arch);
            }
            out[pos++] = ']';
            out[pos] = 0;
            send_resp(fd, 200, "OK", "application/json", out, pos);
        } else {
            send_resp(fd, 200, "OK", "application/json", "[]", 2);
        }
        free(inst); free(out);
    } else if (strcmp(method, "GET") == 0 &&
               (strcmp(path, "/sdapi/v1/sd-vae") == 0 || strcmp(path, "/sdapi/v1/embeddings") == 0)) {
        send_resp(fd, 200, "OK", "application/json", "[]", 2);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/sdapi/v1/options") == 0) {
        char mdl_esc[128], out[320];
        http_srv_lock();
        json_escape(g_st->model[0] ? g_st->model : "waiNSFWIllustrious_v140", mdl_esc, sizeof(mdl_esc));
        http_srv_unlock();
        int n = snprintf(out, sizeof(out), "{\"sd_model_checkpoint\":\"%s\",\"sd_vae\":\"Automatic\"}", mdl_esc);
        send_resp(fd, 200, "OK", "application/json", out, (size_t)n);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/sdapi/v1/options") == 0) {
        char req_m[128] = {0};
        if (json_get_str(body, "sd_model_checkpoint", req_m, sizeof(req_m)) && req_m[0]) {
            scePthreadMutexLock(&sync_gen_mu);
            switch_model_sync(req_m);
            scePthreadMutexUnlock(&sync_gen_mu);
        }
        send_resp(fd, 200, "OK", "application/json", "{}", 2);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/sdapi/v1/progress") == 0) {
        char st_esc[256], out[768];
        http_srv_lock();
        json_escape(g_st->stage, st_esc, sizeof(st_esc));
        int busy = g_st->busy;
        float prog = busy ? g_st->progress : 0.0f;
        float eta = busy ? g_st->eta_sec : 0.0f;
        uint32_t steps = (busy && g_st->cur_steps) ? g_st->cur_steps : g_job->steps;
        uint32_t cur_step = (uint32_t)(prog * (float)steps);
        http_srv_unlock();
        int n = snprintf(out, sizeof(out),
                         "{\"progress\":%.3f,\"eta_relative\":%.1f,\"state\":{\"skipped\":false,\"interrupted\":false,"
                         "\"job\":\"%s\",\"job_count\":%d,\"job_timestamp\":\"0\",\"job_no\":0,"
                         "\"sampling_step\":%u,\"sampling_steps\":%u},\"current_image\":null,\"textinfo\":\"%s\"}",
                         (double)prog, (double)eta, busy ? "txt2img" : "", busy ? 1 : 0, cur_step, steps, st_esc);
        send_resp(fd, 200, "OK", "application/json", out, (size_t)n);
    } else if (strcmp(method, "GET") == 0 &&
               (strcmp(path, "/internal/ping") == 0 || strcmp(path, "/sdapi/v1/cmd-flags") == 0 ||
                strcmp(path, "/sdapi/v1/memory") == 0)) {
        send_resp(fd, 200, "OK", "application/json", "{}", 2);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/sdapi/v1/txt2img") == 0) {
        uint8_t* copy = NULL;
        size_t sz = 0;
        DiffStatus snap;
        char used_pr[1024] = {0}, used_ng[512] = {0};
        int used_smp = 0;
        int rc = run_sync_generation(body, &copy, &sz, NULL, &snap, used_pr, sizeof(used_pr), used_ng, sizeof(used_ng), &used_smp, 1);
        if (rc == -2) {
            send_resp(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"busy\"}", 16);
            free(req);
            return;
        }
        if (rc != 0 || !copy) {
            send_resp(fd, 500, "Internal Server Error", "application/json", "{\"error\":\"generation failed\"}", 29);
            free(req);
            return;
        }
        size_t b64_len = 0;
        char* b64 = b64_encode(copy, sz, &b64_len);
        free(copy);
        if (!b64) { send_resp(fd, 500, "Internal Server Error", "text/plain", "oom", 3); free(req); return; }

        // Build full A1111-compliant `info` JSON (including `infotexts` required by bz-cogs/aimage, SillyTavern, etc.)
        static const char* SMP_N[4] = {"Euler a", "Euler", "LCM", "DPM++ 2M Karras"};
        const char* smpn = SMP_N[(used_smp >= 0 && used_smp <= 3) ? used_smp : 0];
        char* pr_e = (char*)malloc(1600);
        char* ng_e = (char*)malloc(900);
        char* md_e = (char*)malloc(160);
        char* infotxt = (char*)malloc(2600);
        char* info_raw = (char*)malloc(6400);
        char* info_esc = (char*)malloc(12800);
        size_t jcap = b64_len + 16384;
        char* resp = (char*)malloc(jcap);
        if (pr_e && ng_e && md_e && infotxt && info_raw && info_esc && resp) {
            json_escape(used_pr, pr_e, 1600);
            json_escape(used_ng, ng_e, 900);
            json_escape(snap.model[0] ? snap.model : "GarlicDiffusion", md_e, 160);
            snprintf(infotxt, 2600,
                     "%s\\nNegative prompt: %s\\nSteps: %u, Sampler: %s, CFG scale: %.1f, Seed: %llu, Size: %ux%u, Model: %s",
                     pr_e, ng_e, snap.steps, smpn, (double)snap.cfg,
                     (unsigned long long)snap.seed, snap.width, snap.height, md_e);
            snprintf(info_raw, 6400,
                     "{\"prompt\":\"%s\",\"all_prompts\":[\"%s\"],\"negative_prompt\":\"%s\","
                     "\"all_negative_prompts\":[\"%s\"],\"seed\":%llu,\"all_seeds\":[%llu],"
                     "\"subseed\":-1,\"all_subseeds\":[-1],\"width\":%u,\"height\":%u,"
                     "\"sampler_name\":\"%s\",\"cfg_scale\":%.2f,\"steps\":%u,\"batch_size\":1,"
                     "\"sd_model_name\":\"%s\",\"infotexts\":[\"%s\"],\"extra_generation_params\":{}}",
                     pr_e, pr_e, ng_e, ng_e, (unsigned long long)snap.seed, (unsigned long long)snap.seed,
                     snap.width, snap.height, smpn, (double)snap.cfg, snap.steps, md_e, infotxt);
            json_escape(info_raw, info_esc, 12800);
            int n = snprintf(resp, jcap,
                             "{\"images\":[\"%s\"],\"parameters\":{\"prompt\":\"%s\",\"negative_prompt\":\"%s\","
                             "\"width\":%u,\"height\":%u,\"steps\":%u,\"cfg_scale\":%.2f,\"seed\":%llu,"
                             "\"sampler_name\":\"%s\"},\"info\":\"%s\"}",
                             b64, pr_e, ng_e, snap.width, snap.height, snap.steps, (double)snap.cfg,
                             (unsigned long long)snap.seed, smpn, info_esc);
            send_resp(fd, 200, "OK", "application/json", resp, (size_t)n);
        }
        free(pr_e); free(ng_e); free(md_e); free(infotxt); free(info_raw); free(info_esc); free(resp);
        free(b64);
    // ---- OpenAI compatible /v1/images/generations ----
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/images/generations") == 0) {
        uint8_t* copy = NULL;
        size_t sz = 0;
        uint32_t new_id = 0;
        int rc = run_sync_generation(body, &copy, &sz, &new_id, NULL, NULL, 0, NULL, 0, NULL, 1);
        if (rc == -2) {
            send_resp(fd, 503, "Service Unavailable", "application/json",
                      "{\"error\":{\"message\":\"busy\",\"type\":\"server_error\"}}", 54);
            free(req);
            return;
        }
        if (rc != 0 || !copy) {
            send_resp(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":{\"message\":\"generation failed\",\"type\":\"server_error\"}}", 67);
            free(req);
            return;
        }
        size_t b64_len = 0;
        char* b64 = b64_encode(copy, sz, &b64_len);
        free(copy);
        if (!b64) { send_resp(fd, 500, "Internal Server Error", "text/plain", "oom", 3); free(req); return; }
        size_t jcap = b64_len + 128;
        char* resp = (char*)malloc(jcap);
        if (resp) {
            int n = snprintf(resp, jcap, "{\"created\":%u,\"data\":[{\"b64_json\":\"%s\"}]}", new_id, b64);
            send_resp(fd, 200, "OK", "application/json", resp, (size_t)n);
            free(resp);
        }
        free(b64);
    } else {
        send_resp(fd, 404, "Not Found", "text/plain", "not found", 9);
    }
    free(req);
}

static void find_local_ip(void) {
    snprintf(g_addr, sizeof(g_addr), ":%d", DIFF_HTTP_PORT);
    OrbisNetId s = sceNetSocket("gdiff_ip", ORBIS_NET_AF_INET, ORBIS_NET_SOCK_DGRAM, IPPROTO_UDP_PS4);
    if (s < 0) return;
    static const uint8_t far[4] = {192, 0, 2, 1};
    SockAddrIn a;
    memset(&a, 0, sizeof(a));
    a.len = sizeof(a);
    a.family = ORBIS_NET_AF_INET;
    a.port = sceNetHtons(9);
    memcpy(&a.addr, far, 4);
    OrbisNetSocklen_t len = sizeof(a);
    if (sceNetConnect(s, (const OrbisNetSockaddr*)&a, sizeof(a)) == 0 &&
        sceNetGetsockname(s, (OrbisNetSockaddr*)&a, &len) == 0 && a.addr) {
        const uint8_t* b = (const uint8_t*)&a.addr;
        snprintf(g_addr, sizeof(g_addr), "%u.%u.%u.%u:%d", b[0], b[1], b[2], b[3], DIFF_HTTP_PORT);
    }
    sceNetSocketClose(s);
}

static volatile int g_active_clients = 0;

static void* client_worker(void* arg) {
    int fd = (int)(intptr_t)arg;
    handle_client(fd);
    sceNetSocketClose(fd);
    http_srv_lock();
    if (g_active_clients > 0) g_active_clients--;
    http_srv_unlock();
    return NULL;
}

static void* srv_loop(void* arg) {
    (void)arg;
    OrbisNetId l = sceNetSocket("gdiff_http", ORBIS_NET_AF_INET, ORBIS_NET_SOCK_STREAM, IPPROTO_TCP_PS4);
    if (l < 0) { glog("http: socket err %d", l); return NULL; }
    int one = 1;
    sceNetSetsockopt(l, SOL_SOCKET_PS4, SO_REUSEADDR_PS4, &one, sizeof(one));
    sceNetSetsockopt(l, SOL_SOCKET_PS4, SO_NOSIGPIPE_PS4, &one, sizeof(one));
    SockAddrIn a;
    memset(&a, 0, sizeof(a));
    a.len = sizeof(a);
    a.family = ORBIS_NET_AF_INET;
    a.port = sceNetHtons(DIFF_HTTP_PORT);
    if (sceNetBind(l, (const OrbisNetSockaddr*)&a, sizeof(a)) || sceNetListen(l, 16)) {
        glog("http: bind/listen failed on %d", DIFF_HTTP_PORT);
        sceNetSocketClose(l);
        return NULL;
    }
    glog("http: Web UI + A1111 + OpenAI API on http://%s", g_addr);
    for (;;) {
        SockAddrIn ca;
        OrbisNetSocklen_t clen = sizeof(ca);
        OrbisNetId c = sceNetAccept(l, (OrbisNetSockaddr*)&ca, &clen);
        if (c < 0) { sceKernelUsleep(50000); continue; }
        sceNetSetsockopt(c, SOL_SOCKET_PS4, SO_NOSIGPIPE_PS4, &one, sizeof(one));
        http_srv_lock();
        int can_spawn = (g_active_clients < 16);
        if (can_spawn) g_active_clients++;
        http_srv_unlock();
        if (can_spawn) {
            OrbisPthread th;
            OrbisPthreadAttr attr;
            scePthreadAttrInit(&attr);
            scePthreadAttrSetstacksize(&attr, 128 << 10);
            if (scePthreadCreate(&th, &attr, client_worker, (void*)(intptr_t)c, "gdiff_req") == 0) {
                scePthreadDetach(th);
                continue;
            }
            http_srv_lock();
            if (g_active_clients > 0) g_active_clients--;
            http_srv_unlock();
        }
        handle_client(c);
        sceNetSocketClose(c);
    }
    return NULL;
}

int http_srv_start(DiffJob* shared_job, DiffStatus* shared_status) {
    g_job = shared_job;
    g_st = shared_status;
    load_web_ui();
    scePthreadMutexInit(&srv_mu, NULL, "gdiff_srv");
    scePthreadMutexInit(&sync_gen_mu, NULL, "gdiff_sgen");
    if (sceSysmoduleLoadModuleInternal(ORBIS_SYSMODULE_INTERNAL_NET) < 0) return -1;
    if (sceNetInit() < 0) return -1;
    uint64_t act[20] = {1};
    _sigaction(SIGPIPE_PS4, act, NULL);
    find_local_ip();
    OrbisPthreadAttr attr;
    scePthreadAttrInit(&attr);
    scePthreadAttrSetstacksize(&attr, 256 << 10);
    if (scePthreadCreate(&srv_th, &attr, srv_loop, NULL, "gdiff_http")) return -1;
    return 0;
}
