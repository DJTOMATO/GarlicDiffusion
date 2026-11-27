// Garlic Diffusion: Persistent Image Gallery, Folders, ZIP Builder & User/Model Preset Manager
// Uses raw POSIX open/read/write/unlink + opendir for full OpenOrbis PS4 compatibility.

#include "gallery.h"
#include "png_write.h"
#include "preset_seeds.h"

#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#ifndef SDXL_CPU
#include <orbis/libkernel.h>
#endif
#else
#include <io.h>
#include <direct.h>
#define mkdir(p, m) _mkdir(p)
#endif

static char g_out_dir[256] = "/data/garlic-diffusion/out";
static char g_thumbs_dir[280] = "/data/garlic-diffusion/out/.thumbs";
static char g_folders_file[280] = "/data/garlic-diffusion/out/.folders";
static char g_presets_dir[256] = "/data/garlic-diffusion/presets";
static char g_styles_dir[256] = "/data/garlic-diffusion/styles";
static char g_chars_dir[256] = "/data/garlic-diffusion/characters";
static uint32_t g_next_seq = 1;
static volatile uint32_t g_gal_rev = 1;

uint32_t gallery_revision(void) { return g_gal_rev; }

static int path_exists_local(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    close(fd);
    return 1;
}

static void ensure_dir(const char* path) {
    char tmp[256];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char c = *p;
            *p = 0;
            mkdir(tmp, 0777);
            *p = c;
        }
    }
    mkdir(tmp, 0777);
}

static int write_all_file(const char* path, const void* data, size_t len) {
#ifdef _WIN32
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0666);
#else
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
#endif
    if (fd < 0) return -1;
    const uint8_t* p = (const uint8_t*)data;
    size_t done = 0;
    while (done < len) {
        size_t chunk = (len - done) > (1u << 20) ? (1u << 20) : (len - done);
        ssize_t w = write(fd, p + done, (unsigned)chunk);
        if (w <= 0) { close(fd); return -1; }
        done += (size_t)w;
    }
    close(fd);
    return 0;
}

static uint8_t* read_all_file(const char* path, size_t* out_size) {
#ifdef _WIN32
    int fd = open(path, O_RDONLY | O_BINARY);
#else
    int fd = open(path, O_RDONLY);
#endif
    if (fd < 0) return NULL;
    off_t sz = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    if (sz <= 0 || sz > (64 * 1024 * 1024)) { close(fd); return NULL; }
    uint8_t* buf = (uint8_t*)malloc((size_t)sz + 1);
    if (!buf) { close(fd); return NULL; }
    size_t done = 0;
    while (done < (size_t)sz) {
        size_t chunk = ((size_t)sz - done) > (1u << 20) ? (1u << 20) : ((size_t)sz - done);
        ssize_t r = read(fd, buf + done, (unsigned)chunk);
        if (r <= 0) break;
        done += (size_t)r;
    }
    close(fd);
    if (done != (size_t)sz) { free(buf); return NULL; }
    buf[done] = 0;
    if (out_size) *out_size = done;
    return buf;
}

static void strip_newlines(char* s) {
    for (; *s; s++) {
        if (*s == '\r' || *s == '\n') *s = ' ';
    }
}

static int ci_eq_gal(const char* a, const char* b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == 0 && *b == 0;
}

static void make_safe_id(const char* name, char* out, size_t max) {
    size_t j = 0;
    for (size_t i = 0; name && name[i] && j + 1 < max; i++) {
        unsigned char c = (unsigned char)name[i];
        if (isalnum(c) || c == '-' || c == '_') out[j++] = (char)tolower(c);
        else if (c == ' ' || c == '.') out[j++] = '_';
    }
    while (j > 0 && out[j - 1] == '_') j--;
    if (j == 0) { snprintf(out, max, "preset"); return; }
    out[j] = 0;
}

int gallery_clean_folder_name(const char* raw, char* out, size_t max) {
    if (!out || max == 0) return -1;
    out[0] = 0;
    if (!raw) return 0;
    while (*raw == ' ' || *raw == '\t' || *raw == '\r' || *raw == '\n') raw++;
    size_t j = 0;
    int prev_sp = 0;
    for (size_t i = 0; raw[i] && j + 1 < max && j < 48; i++) {
        unsigned char c = (unsigned char)raw[i];
        if (isalnum(c) || c == '-' || c == '_') {
            out[j++] = (char)c;
            prev_sp = 0;
        } else if (c == ' ') {
            if (!prev_sp && j > 0) { out[j++] = ' '; prev_sp = 1; }
        }
    }
    while (j > 0 && out[j - 1] == ' ') j--;
    out[j] = 0;
    if (strncmp(out, "__", 2) == 0) { out[0] = 0; return -1; }
    return (int)j;
}

int gallery_valid_id(const char* id) {
    if (!id || !id[0] || strlen(id) > 60) return 0;
    for (const char* p = id; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (!(isalnum(c) || c == '_' || c == '-')) return 0;
    }
    return 1;
}

static uint32_t fnv1a_str(uint32_t h, const char* s) {
    if (!s) return h;
    for (const unsigned char* p = (const unsigned char*)s; *p; p++) {
        h ^= (uint32_t)(*p);
        h *= 16777619u;
    }
    h ^= 0xFFu;
    return h * 16777619u;
}

static int seed_sig_matches(const char* marker_path, const char* expected_sig) {
    size_t sz = 0;
    uint8_t* buf = read_all_file(marker_path, &sz);
    if (!buf) return 0;
    int eq = (strcmp((const char*)buf, expected_sig) == 0);
    free(buf);
    return eq;
}

static void seed_default_presets(void) {
    gd_init_preset_seeds();
    size_t cnt = sizeof(SEED_PRESETS) / sizeof(SEED_PRESETS[0]);
    uint32_t h = 2166136261u ^ (uint32_t)cnt;
    for (size_t i = 0; i < cnt; i++) {
        h = fnv1a_str(h, SEED_PRESETS[i].id);
        h = fnv1a_str(h, SEED_PRESETS[i].name);
        h = fnv1a_str(h, SEED_PRESETS[i].model);
        h = fnv1a_str(h, SEED_PRESETS[i].prompt);
        h = fnv1a_str(h, SEED_PRESETS[i].negative);
        h ^= (uint32_t)(SEED_PRESETS[i].width ^ (SEED_PRESETS[i].height << 16) ^
                        (SEED_PRESETS[i].steps << 8) ^ (SEED_PRESETS[i].nsfw << 1) ^ SEED_PRESETS[i].is_speed);
        h *= 16777619u;
    }
    char sig[48];
    int sn = snprintf(sig, sizeof(sig), "%u:%08x\n", (unsigned)cnt, h);
    char marker[320];
    snprintf(marker, sizeof(marker), "%s/.seeded_sig", g_presets_dir);
    if (seed_sig_matches(marker, sig)) return;

    for (size_t i = 0; i < cnt; i++) {
        presets_save(&SEED_PRESETS[i], NULL, 0);
    }
    write_all_file(marker, sig, (size_t)(sn > 0 ? sn : 0));
}

void gallery_init(const char* out_dir, const char* presets_dir) {
    if (out_dir && out_dir[0]) snprintf(g_out_dir, sizeof(g_out_dir), "%s", out_dir);
    snprintf(g_thumbs_dir, sizeof(g_thumbs_dir), "%s/.thumbs", g_out_dir);
    snprintf(g_folders_file, sizeof(g_folders_file), "%s/.folders", g_out_dir);
    if (presets_dir && presets_dir[0]) snprintf(g_presets_dir, sizeof(g_presets_dir), "%s", presets_dir);
    ensure_dir(g_out_dir);
    ensure_dir(g_thumbs_dir);
    ensure_dir(g_presets_dir);

    uint32_t max_seq = 0;
    DIR* d = opendir(g_out_dir);
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL) {
            unsigned seq = 0;
            if (sscanf(ent->d_name, "g_%u_", &seq) == 1 && seq > max_seq) max_seq = seq;
        }
        closedir(d);
    }
    g_next_seq = max_seq + 1;
    g_gal_rev++;
    seed_default_presets();
}

static void parse_meta_buf(char* buf, GalleryItem* it) {
    char* line = buf;
    while (*line) {
        char* nl = strchr(line, '\n');
        if (nl) *nl = 0;
        size_t len = strlen(line);
        while (len > 0 && line[len - 1] == '\r') line[--len] = 0;
        char* eq = strchr(line, '=');
        if (eq) {
            *eq = 0;
            const char* k = line;
            const char* v = eq + 1;
            if (strcmp(k, "model") == 0) snprintf(it->model, sizeof(it->model), "%s", v);
            else if (strcmp(k, "prompt") == 0) snprintf(it->prompt, sizeof(it->prompt), "%s", v);
            else if (strcmp(k, "negative") == 0) snprintf(it->negative, sizeof(it->negative), "%s", v);
            else if (strcmp(k, "sampler") == 0) snprintf(it->sampler, sizeof(it->sampler), "%s", v);
            else if (strcmp(k, "lora") == 0) snprintf(it->lora, sizeof(it->lora), "%s", v);
            else if (strcmp(k, "folder") == 0) gallery_clean_folder_name(v, it->folder, sizeof(it->folder));
            else if (strcmp(k, "starred") == 0) it->starred = atoi(v) ? 1 : 0;
            else if (strcmp(k, "width") == 0) it->width = (uint32_t)atoi(v);
            else if (strcmp(k, "height") == 0) it->height = (uint32_t)atoi(v);
            else if (strcmp(k, "steps") == 0) it->steps = (uint32_t)atoi(v);
            else if (strcmp(k, "cfg") == 0) it->cfg = (float)atof(v);
            else if (strcmp(k, "seed") == 0) it->seed = strtoull(v, NULL, 10);
            else if (strcmp(k, "eco") == 0) it->eco = atoi(v);
            else if (strcmp(k, "upscaled") == 0) it->upscaled = atoi(v);
            else if (strcmp(k, "sec") == 0) it->sec = atof(v);
            else if (strcmp(k, "unet_sec") == 0) it->unet_sec = atof(v);
            else if (strcmp(k, "vae_sec") == 0) it->vae_sec = atof(v);
            else if (strcmp(k, "upscale_sec") == 0) it->upscale_sec = atof(v);
            else if (strcmp(k, "face_fix_sec") == 0) it->face_fix_sec = atof(v);
            else if (strcmp(k, "cooldown_sec") == 0) it->cooldown_sec = atof(v);
            else if (strcmp(k, "eff_tflops") == 0) it->eff_tflops = atof(v);
        }
        if (!nl) break;
        line = nl + 1;
    }
}

static int write_meta_item(const char* id, const GalleryItem* meta) {
    char pr[1024], ng[512], lr[128], fld[64] = {0};
    snprintf(pr, sizeof(pr), "%s", meta->prompt); strip_newlines(pr);
    snprintf(ng, sizeof(ng), "%s", meta->negative); strip_newlines(ng);
    snprintf(lr, sizeof(lr), "%s", meta->lora); strip_newlines(lr);
    if (meta->folder[0]) gallery_clean_folder_name(meta->folder, fld, sizeof(fld));

    char mbuf[2800];
    int n = snprintf(mbuf, sizeof(mbuf),
                     "id=%s\nmodel=%s\nwidth=%u\nheight=%u\nsteps=%u\ncfg=%.2f\nsampler=%s\n"
                     "seed=%llu\neco=%d\nupscaled=%d\nsec=%.2f\nunet_sec=%.2f\nvae_sec=%.2f\n"
                     "upscale_sec=%.2f\nface_fix_sec=%.2f\ncooldown_sec=%.2f\neff_tflops=%.3f\n"
                     "lora=%s\nfolder=%s\nstarred=%d\nprompt=%s\nnegative=%s\n",
                     id, meta->model, meta->width, meta->height, meta->steps, (double)meta->cfg,
                     meta->sampler[0] ? meta->sampler : "Euler a",
                     (unsigned long long)meta->seed, meta->eco, meta->upscaled, meta->sec,
                     meta->unet_sec, meta->vae_sec, meta->upscale_sec, meta->face_fix_sec,
                     meta->cooldown_sec, meta->eff_tflops, lr,
                     fld, meta->starred ? 1 : 0, pr, ng);
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.meta", g_out_dir, id);
    return write_all_file(fp, mbuf, (size_t)(n > 0 ? n : 0));
}

static uint8_t* build_thumb_png_mem(const uint8_t* rgb, uint32_t w, uint32_t h, size_t* out_sz) {
    if (!rgb || !w || !h) return NULL;
    const uint32_t MAX_EDGE = 240u;
    uint32_t tw = w, th = h;
    if (w >= h && w > MAX_EDGE) {
        tw = MAX_EDGE;
        th = (uint32_t)(((uint64_t)h * MAX_EDGE + w / 2u) / w);
    } else if (h > w && h > MAX_EDGE) {
        th = MAX_EDGE;
        tw = (uint32_t)(((uint64_t)w * MAX_EDGE + h / 2u) / h);
    }
    if (tw < 1u) tw = 1u;
    if (th < 1u) th = 1u;
    uint8_t* sub = (tw == w && th == h) ? NULL : rgb_resize(rgb, w, h, tw, th);
    const uint8_t* src = sub ? sub : rgb;
    size_t tsz = 0;
    uint8_t* tpng = png_encode_rgb(src, tw, th, &tsz);
    free(sub);
    if (out_sz) *out_sz = tsz;
    return tpng;
}

static uint8_t* build_thumb_png(const char* id, const uint8_t* rgb, uint32_t w, uint32_t h, size_t* out_sz) {
    size_t tsz = 0;
    uint8_t* tpng = build_thumb_png_mem(rgb, w, h, &tsz);
    if (tpng && tsz > 0 && id && id[0]) {
        ensure_dir(g_thumbs_dir);
        char tfp[360];
        snprintf(tfp, sizeof(tfp), "%s/%s.png", g_thumbs_dir, id);
        write_all_file(tfp, tpng, tsz);
    }
    if (out_sz) *out_sz = tsz;
    return tpng;
}

// ---- Bounded RAM-First Image Staging Buffer (tmpfs-style) ---------------------------------------
#define GALLERY_STAGE_MAX_ITEMS 8

typedef struct {
    int active;
    char id[64];
    uint8_t* png;
    size_t png_size;
    uint8_t* thumb_png;
    size_t thumb_size;
    GalleryItem meta;
} StagedEntry;

static StagedEntry g_stage_q[GALLERY_STAGE_MAX_ITEMS];
static StagedEntry g_stage_active_flush;
static int g_stage_head = 0, g_stage_tail = 0, g_stage_count = 0;
static int g_stage_flushing_now = 0;
static size_t g_stage_used_bytes = 0;
static size_t g_stage_max_bytes = (64u << 20);
static int g_stage_state = GALLERY_STAGE_IDLE;
static uint32_t g_stage_flushed_total = 0;
static int g_stage_fallback_used = 0;
static char g_stage_last_id[64] = "";
static char g_stage_error[128] = "";
static int g_stage_thread_ok = 0;

#if !defined(_WIN32) && !defined(SDXL_CPU)
static OrbisPthreadMutex g_stage_mu;
static int g_stage_mu_ok = 0;
static void stage_lock(void) { if (g_stage_mu_ok) scePthreadMutexLock(&g_stage_mu); }
static void stage_unlock(void) { if (g_stage_mu_ok) scePthreadMutexUnlock(&g_stage_mu); }
static void stage_sleep_us(uint32_t us) { sceKernelUsleep(us); }
#else
static void stage_lock(void) {}
static void stage_unlock(void) {}
static void stage_sleep_us(uint32_t us) { (void)us; }
#endif

static int persist_staged_files(const char* id, const uint8_t* png, size_t png_size,
                                const uint8_t* thumb_png, size_t thumb_size, const GalleryItem* meta) {
    ensure_dir(g_out_dir);
    char fp[360];
    snprintf(fp, sizeof(fp), "%s/%s.png", g_out_dir, id);
    if (write_all_file(fp, png, png_size) != 0) return -1;

    if (thumb_png && thumb_size > 0) {
        ensure_dir(g_thumbs_dir);
        snprintf(fp, sizeof(fp), "%s/%s.png", g_thumbs_dir, id);
        write_all_file(fp, thumb_png, thumb_size);
    }
    if (meta->folder[0]) {
        char clean_f[64];
        if (gallery_clean_folder_name(meta->folder, clean_f, sizeof(clean_f)) > 0)
            gallery_create_folder(clean_f, NULL, 0);
    }
    if (write_meta_item(id, meta) != 0) return -1;
    return 0;
}

#if !defined(_WIN32) && !defined(SDXL_CPU)
static void* stage_flush_loop(void* arg) {
    (void)arg;
    for (;;) {
        StagedEntry item;
        int has_item = 0;
        stage_lock();
        if (g_stage_count > 0) {
            item = g_stage_q[g_stage_head];
            memset(&g_stage_q[g_stage_head], 0, sizeof(g_stage_q[g_stage_head]));
            g_stage_head = (g_stage_head + 1) % GALLERY_STAGE_MAX_ITEMS;
            g_stage_count--;
            g_stage_active_flush = item;
            g_stage_flushing_now = 1;
            g_stage_state = GALLERY_STAGE_FLUSHING;
            g_stage_error[0] = 0;
            has_item = 1;
        }
        stage_unlock();

        if (!has_item) {
            stage_sleep_us(35000);
            continue;
        }

        int rc = persist_staged_files(item.id, item.png, item.png_size,
                                      item.thumb_png, item.thumb_size, &item.meta);

        stage_lock();
        size_t freed = item.png_size + item.thumb_size;
        if (g_stage_used_bytes >= freed) g_stage_used_bytes -= freed;
        else g_stage_used_bytes = 0;
        memset(&g_stage_active_flush, 0, sizeof(g_stage_active_flush));
        g_stage_flushing_now = 0;
        if (rc == 0) {
            g_stage_flushed_total++;
            snprintf(g_stage_last_id, sizeof(g_stage_last_id), "%s", item.id);
            g_stage_state = (g_stage_count > 0) ? GALLERY_STAGE_FLUSHING : GALLERY_STAGE_COMPLETE;
            g_gal_rev++;
        } else {
            snprintf(g_stage_error, sizeof(g_stage_error), "Failed writing %s.png to disk", item.id);
            g_stage_state = GALLERY_STAGE_ERROR;
        }
        stage_unlock();

        free(item.png);
        free(item.thumb_png);
    }
    return NULL;
}
#endif

void gallery_stage_init(size_t max_bytes) {
    if (max_bytes >= (8u << 20) && max_bytes <= (128u << 20)) {
        g_stage_max_bytes = max_bytes;
    }
#if !defined(_WIN32) && !defined(SDXL_CPU)
    if (!g_stage_mu_ok) {
        if (scePthreadMutexInit(&g_stage_mu, NULL, "gdiff_stg") == 0) g_stage_mu_ok = 1;
    }
    if (!g_stage_thread_ok && g_stage_mu_ok) {
        OrbisPthread th;
        OrbisPthreadAttr attr;
        scePthreadAttrInit(&attr);
        scePthreadAttrSetstacksize(&attr, 128 << 10);
        if (scePthreadCreate(&th, &attr, stage_flush_loop, NULL, "gdiff_flush") == 0) {
            scePthreadDetach(th);
            g_stage_thread_ok = 1;
        }
        scePthreadAttrDestroy(&attr);
    }
#endif
}

void gallery_stage_flush_all(void) {
    for (int wait_it = 0; wait_it < 3000; wait_it++) {
        stage_lock();
        int pending = g_stage_count + g_stage_flushing_now;
        stage_unlock();
        if (pending == 0) break;
        stage_sleep_us(20000);
    }
}

void gallery_stage_get_status(GalleryStageStatus* out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    stage_lock();
    out->state = g_stage_state;
    out->queued_count = (uint32_t)(g_stage_count + g_stage_flushing_now);
    out->flushed_total = g_stage_flushed_total;
    out->used_bytes = (uint32_t)g_stage_used_bytes;
    out->max_bytes = (uint32_t)g_stage_max_bytes;
    out->fallback_used = g_stage_fallback_used;
    snprintf(out->last_id, sizeof(out->last_id), "%s", g_stage_last_id);
    snprintf(out->error, sizeof(out->error), "%s", g_stage_error);
    stage_unlock();
}

static int stage_find_copy_png(const char* id, int want_thumb, uint8_t** out_buf, size_t* out_sz) {
    int found = 0;
    stage_lock();
    for (int k = -1; k < GALLERY_STAGE_MAX_ITEMS; k++) {
        const StagedEntry* e = (k < 0) ? (g_stage_flushing_now ? &g_stage_active_flush : NULL)
                                       : (g_stage_q[k].active ? &g_stage_q[k] : NULL);
        if (!e || strcmp(e->id, id) != 0) continue;
        const uint8_t* src = want_thumb ? e->thumb_png : e->png;
        size_t sz = want_thumb ? e->thumb_size : e->png_size;
        if (src && sz > 0) {
            uint8_t* cp = (uint8_t*)malloc(sz);
            if (cp) {
                memcpy(cp, src, sz);
                *out_buf = cp;
                if (out_sz) *out_sz = sz;
                found = 1;
            }
        }
        break;
    }
    stage_unlock();
    return found;
}

static int stage_find_meta(const char* id, GalleryItem* out) {
    int found = 0;
    stage_lock();
    for (int k = -1; k < GALLERY_STAGE_MAX_ITEMS; k++) {
        const StagedEntry* e = (k < 0) ? (g_stage_flushing_now ? &g_stage_active_flush : NULL)
                                       : (g_stage_q[k].active ? &g_stage_q[k] : NULL);
        if (!e || strcmp(e->id, id) != 0) continue;
        *out = e->meta;
        snprintf(out->id, sizeof(out->id), "%s", e->id);
        out->bytes = e->png_size;
        found = 1;
        break;
    }
    stage_unlock();
    return found;
}

int gallery_stage_enqueue(const uint8_t* png, size_t png_size, const uint8_t* rgb_opt, uint32_t rgb_w, uint32_t rgb_h,
                          const GalleryItem* meta, char* out_id, size_t out_id_max) {
    if (!png || !png_size || !meta) return -1;

    char id[64];
    stage_lock();
    snprintf(id, sizeof(id), "g_%05u_%llu", g_next_seq++, (unsigned long long)(meta->seed & 0xFFFFFFFFull));
    g_stage_state = GALLERY_STAGE_STAGING;
    g_stage_fallback_used = 0;
    g_stage_error[0] = 0;
    snprintf(g_stage_last_id, sizeof(g_stage_last_id), "%s", id);
    stage_unlock();

    size_t thumb_sz = 0;
    uint8_t* thumb_png = (rgb_opt && rgb_w > 0 && rgb_h > 0)
        ? build_thumb_png_mem(rgb_opt, rgb_w, rgb_h, &thumb_sz) : NULL;
    size_t need = png_size + thumb_sz;

    // Safe fallback if background flusher isn't active or single image exceeds hard RAM budget
    if (!g_stage_thread_ok || need > g_stage_max_bytes) {
        if (g_stage_thread_ok) gallery_stage_flush_all();
        stage_lock();
        g_stage_state = GALLERY_STAGE_FLUSHING;
        g_stage_fallback_used = 1;
        stage_unlock();
        int rc = persist_staged_files(id, png, png_size, thumb_png, thumb_sz, meta);
        free(thumb_png);
        stage_lock();
        if (rc == 0) {
            g_stage_flushed_total++;
            g_stage_state = GALLERY_STAGE_COMPLETE;
            g_gal_rev++;
        } else {
            snprintf(g_stage_error, sizeof(g_stage_error), "Direct-write fallback failed for %s", id);
            g_stage_state = GALLERY_STAGE_ERROR;
        }
        stage_unlock();
        if (rc == 0 && out_id && out_id_max) snprintf(out_id, out_id_max, "%s", id);
        return rc;
    }

    // Backpressure: wait for background flusher to free RAM if buffer is full or would exceed cap
    for (int wait_it = 0; wait_it < 1500; wait_it++) {
        stage_lock();
        int room = (g_stage_count < GALLERY_STAGE_MAX_ITEMS) &&
                   (g_stage_used_bytes + need <= g_stage_max_bytes);
        stage_unlock();
        if (room) break;
        stage_sleep_us(20000);
    }

    uint8_t* png_copy = (uint8_t*)malloc(png_size);
    if (!png_copy) {
        // OOM safe fallback: flush queue and write directly
        gallery_stage_flush_all();
        stage_lock();
        g_stage_state = GALLERY_STAGE_FLUSHING;
        g_stage_fallback_used = 1;
        stage_unlock();
        int rc = persist_staged_files(id, png, png_size, thumb_png, thumb_sz, meta);
        free(thumb_png);
        stage_lock();
        if (rc == 0) {
            g_stage_flushed_total++;
            g_stage_state = GALLERY_STAGE_COMPLETE;
            g_gal_rev++;
        } else {
            snprintf(g_stage_error, sizeof(g_stage_error), "OOM fallback disk write failed for %s", id);
            g_stage_state = GALLERY_STAGE_ERROR;
        }
        stage_unlock();
        if (rc == 0 && out_id && out_id_max) snprintf(out_id, out_id_max, "%s", id);
        return rc;
    }
    memcpy(png_copy, png, png_size);

    stage_lock();
    StagedEntry* slot = &g_stage_q[g_stage_tail];
    slot->active = 1;
    snprintf(slot->id, sizeof(slot->id), "%s", id);
    slot->png = png_copy;
    slot->png_size = png_size;
    slot->thumb_png = thumb_png;
    slot->thumb_size = thumb_sz;
    slot->meta = *meta;
    snprintf(slot->meta.id, sizeof(slot->meta.id), "%s", id);
    slot->meta.bytes = png_size;
    g_stage_tail = (g_stage_tail + 1) % GALLERY_STAGE_MAX_ITEMS;
    g_stage_count++;
    g_stage_used_bytes += need;
    g_gal_rev++;
    stage_unlock();

    if (out_id && out_id_max) snprintf(out_id, out_id_max, "%s", id);
    return 0;
}

int gallery_save_ex(const uint8_t* png, size_t png_size, const uint8_t* rgb_opt, uint32_t rgb_w, uint32_t rgb_h,
                    const GalleryItem* meta, char* out_id, size_t out_id_max) {
    if (!png || !png_size || !meta) return -1;
    ensure_dir(g_out_dir);

    char id[64];
    stage_lock();
    snprintf(id, sizeof(id), "g_%05u_%llu", g_next_seq++, (unsigned long long)(meta->seed & 0xFFFFFFFFull));
    stage_unlock();

    size_t tsz = 0;
    uint8_t* tpng = (rgb_opt && rgb_w > 0 && rgb_h > 0) ? build_thumb_png_mem(rgb_opt, rgb_w, rgb_h, &tsz) : NULL;
    int rc = persist_staged_files(id, png, png_size, tpng, tsz, meta);
    free(tpng);
    if (rc != 0) return -1;

    g_gal_rev++;
    if (out_id && out_id_max) snprintf(out_id, out_id_max, "%s", id);
    return 0;
}

int gallery_save(const uint8_t* png, size_t png_size, const GalleryItem* meta, char* out_id, size_t out_id_max) {
    return gallery_save_ex(png, png_size, NULL, 0, 0, meta, out_id, out_id_max);
}

int gallery_get(const char* id, GalleryItem* out) {
    if (!gallery_valid_id(id) || !out) return -1;
    if (stage_find_meta(id, out)) return 0;
    memset(out, 0, sizeof(*out));
    snprintf(out->id, sizeof(out->id), "%s", id);
    snprintf(out->sampler, sizeof(out->sampler), "Euler a");

    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.png", g_out_dir, id);
#ifdef _WIN32
    int fd = open(fp, O_RDONLY | O_BINARY);
#else
    int fd = open(fp, O_RDONLY);
#endif
    if (fd < 0) return -1;
    off_t sz = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    uint8_t hdr[36];
    ssize_t hr = read(fd, hdr, sizeof(hdr));
    close(fd);
    out->bytes = (sz > 0) ? (size_t)sz : 0;
    if (hr >= 33) png_probe_dims(hdr, (size_t)hr, &out->width, &out->height);

    snprintf(fp, sizeof(fp), "%s/%s.meta", g_out_dir, id);
    size_t msz = 0;
    uint8_t* mb = read_all_file(fp, &msz);
    if (mb) {
        parse_meta_buf((char*)mb, out);
        free(mb);
    }
    return 0;
}

static int cmp_id_desc(const void* a, const void* b) {
    const char* ia = (const char*)a;
    const char* ib = (const char*)b;
    unsigned sa = 0, sb = 0;
    int ha = (sscanf(ia, "g_%u_", &sa) == 1);
    int hb = (sscanf(ib, "g_%u_", &sb) == 1);
    if (ha && hb) {
        if (sb != sa) return (sb > sa) ? 1 : -1;
        return strcmp(ib, ia);
    }
    if (ha != hb) return hb - ha;
    return strcmp(ib, ia);
}

static int ci_substr_gal(const char* hay, const char* needle) {
    if (!hay || !needle || !needle[0]) return 0;
    size_t nl = strlen(needle);
    for (; *hay; hay++) {
        size_t i = 0;
        while (i < nl && hay[i] && tolower((unsigned char)hay[i]) == tolower((unsigned char)needle[i])) i++;
        if (i == nl) return 1;
    }
    return 0;
}

static size_t scan_sorted_gallery_ids(char (**out_ids)[64]) {
    *out_ids = NULL;
    DIR* d = opendir(g_out_dir);
    if (!d) return 0;

    size_t cap = 512, cnt = 0;
    char (*ids)[64] = (char (*)[64])malloc(cap * sizeof(*ids));
    if (!ids) { closedir(d); return 0; }

    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        const char* nm = ent->d_name;
        if (nm[0] == '.') continue;
        size_t l = strlen(nm);
        if (l <= 4 || strcmp(nm + l - 4, ".png") != 0 || l - 4 >= 64) continue;
        char id[64];
        memcpy(id, nm, l - 4);
        id[l - 4] = 0;
        if (!gallery_valid_id(id)) continue;
        if (cnt >= cap) {
            size_t ncap = cap * 2;
            char (*nids)[64] = (char (*)[64])realloc(ids, ncap * sizeof(*ids));
            if (!nids) break;
            ids = nids;
            cap = ncap;
        }
        memcpy(ids[cnt++], id, 64);
    }
    closedir(d);

    // Include any items currently staged in RAM (not yet written to HDD) so Gallery sees them immediately
    stage_lock();
    for (int k = -1; k < GALLERY_STAGE_MAX_ITEMS; k++) {
        const StagedEntry* e = (k < 0) ? (g_stage_flushing_now ? &g_stage_active_flush : NULL)
                                       : (g_stage_q[k].active ? &g_stage_q[k] : NULL);
        if (!e || !e->id[0]) continue;
        int dup = 0;
        for (size_t j = 0; j < cnt; j++) {
            if (strcmp(ids[j], e->id) == 0) { dup = 1; break; }
        }
        if (dup) continue;
        if (cnt >= cap) {
            size_t ncap = cap * 2;
            char (*nids)[64] = (char (*)[64])realloc(ids, ncap * sizeof(*ids));
            if (!nids) break;
            ids = nids;
            cap = ncap;
        }
        memcpy(ids[cnt++], e->id, 64);
    }
    stage_unlock();

    if (cnt > 1) qsort(ids, cnt, sizeof(*ids), cmp_id_desc);
    *out_ids = ids;
    return cnt;
}

int gallery_list_page_ex(GalleryItem* items, int offset, int limit, const char* query,
                         const char* folder_filter, int starred_only, int* out_total) {
    if (out_total) *out_total = 0;
    if (!items || limit <= 0) return 0;
    if (offset < 0) offset = 0;

    char (*ids)[64] = NULL;
    size_t cnt = scan_sorted_gallery_ids(&ids);
    if (!ids || cnt == 0) { free(ids); return 0; }

    int has_q = (query && query[0] != 0);
    int has_f = (folder_filter && folder_filter[0] != 0);
    int want_unsorted = has_f && strcmp(folder_filter, "__unsorted__") == 0;
    int want_starred = starred_only || (has_f && strcmp(folder_filter, "__starred__") == 0);
    if (has_f && strcmp(folder_filter, "__starred__") == 0) has_f = 0;

    int n = 0;
    if (!has_q && !has_f && !want_starred) {
        if (out_total) *out_total = (int)cnt;
        for (size_t i = (size_t)offset; i < cnt && n < limit; i++) {
            if (gallery_get(ids[i], &items[n]) == 0) n++;
        }
    } else {
        int matched = 0;
        for (size_t i = 0; i < cnt; i++) {
            GalleryItem tmp;
            if (gallery_get(ids[i], &tmp) != 0) continue;
            if (want_starred && !tmp.starred) continue;
            if (has_f) {
                if (want_unsorted) {
                    if (tmp.folder[0] != 0) continue;
                } else if (!ci_eq_gal(tmp.folder, folder_filter)) {
                    continue;
                }
            }
            if (has_q) {
                char sbuf[32];
                snprintf(sbuf, sizeof(sbuf), "%llu", (unsigned long long)tmp.seed);
                if (!ci_substr_gal(tmp.prompt, query) && !ci_substr_gal(tmp.model, query) &&
                    !ci_substr_gal(tmp.lora, query) && !ci_substr_gal(tmp.folder, query) &&
                    !ci_substr_gal(tmp.id, query) && strstr(sbuf, query) == NULL) {
                    continue;
                }
            }
            if (matched >= offset && n < limit) {
                items[n++] = tmp;
            }
            matched++;
        }
        if (out_total) *out_total = matched;
    }
    free(ids);
    return n;
}

int gallery_list_page(GalleryItem* items, int offset, int limit, const char* query, int* out_total) {
    return gallery_list_page_ex(items, offset, limit, query, NULL, 0, out_total);
}

int gallery_list(GalleryItem* items, int max_items) {
    return gallery_list_page_ex(items, 0, max_items, NULL, NULL, 0, NULL);
}

uint8_t* gallery_load_png(const char* id, size_t* out_size) {
    if (!gallery_valid_id(id)) return NULL;
    uint8_t* staged = NULL;
    if (stage_find_copy_png(id, 0, &staged, out_size)) return staged;
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.png", g_out_dir, id);
    return read_all_file(fp, out_size);
}

uint8_t* gallery_load_thumb_png(const char* id, size_t* out_size) {
    if (!gallery_valid_id(id)) return NULL;
    uint8_t* staged = NULL;
    if (stage_find_copy_png(id, 1, &staged, out_size)) return staged;
    char tfp[360];
    snprintf(tfp, sizeof(tfp), "%s/%s.png", g_thumbs_dir, id);
    uint8_t* cached = read_all_file(tfp, out_size);
    if (cached) return cached;

    size_t fsz = 0;
    uint8_t* full = gallery_load_png(id, &fsz);
    if (!full) return NULL;
    uint32_t w = 0, h = 0;
    uint8_t* rgb = png_decode_rgb(full, fsz, &w, &h);
    if (!rgb) {
        if (out_size) *out_size = fsz;
        return full;
    }
    free(full);
    uint8_t* tpng = build_thumb_png(id, rgb, w, h, out_size);
    free(rgb);
    return tpng;
}

int gallery_delete(const char* id) {
    if (!gallery_valid_id(id)) return -1;
    char fp[360];
    snprintf(fp, sizeof(fp), "%s/%s.png", g_out_dir, id);
    int r = unlink(fp);
    snprintf(fp, sizeof(fp), "%s/%s.meta", g_out_dir, id);
    unlink(fp);
    snprintf(fp, sizeof(fp), "%s/%s.png", g_thumbs_dir, id);
    unlink(fp);
    g_gal_rev++;
    return r;
}

// ---- Gallery Folders, Starred Favorites & Stored-ZIP Builder ------------------------------------

static int cmp_folder_asc(const void* a, const void* b) {
    return strcmp(((const GalleryFolder*)a)->name, ((const GalleryFolder*)b)->name);
}

static int load_registered_folders(char out[][64], int max_f) {
    size_t sz = 0;
    uint8_t* buf = read_all_file(g_folders_file, &sz);
    if (!buf) return 0;
    int n = 0;
    char* line = (char*)buf;
    while (*line && n < max_f) {
        char* nl = strchr(line, '\n');
        if (nl) *nl = 0;
        char cl[64];
        if (gallery_clean_folder_name(line, cl, sizeof(cl)) > 0) {
            int dup = 0;
            for (int i = 0; i < n; i++) if (ci_eq_gal(out[i], cl)) { dup = 1; break; }
            if (!dup) snprintf(out[n++], 64, "%s", cl);
        }
        if (!nl) break;
        line = nl + 1;
    }
    free(buf);
    return n;
}

static void save_registered_folders(char list[][64], int n) {
    char buf[GALLERY_MAX_FOLDERS * 66 + 8];
    size_t pos = 0;
    for (int i = 0; i < n && pos + 68 < sizeof(buf); i++) {
        if (list[i][0]) pos += (size_t)snprintf(buf + pos, sizeof(buf) - pos, "%s\n", list[i]);
    }
    buf[pos] = 0;
    ensure_dir(g_out_dir);
    write_all_file(g_folders_file, buf, pos);
}

int gallery_create_folder(const char* name, char* clean_out, size_t max) {
    char cl[64];
    if (gallery_clean_folder_name(name, cl, sizeof(cl)) <= 0) return -1;
    if (clean_out && max) snprintf(clean_out, max, "%s", cl);

    char reg[GALLERY_MAX_FOLDERS][64];
    int n = load_registered_folders(reg, GALLERY_MAX_FOLDERS);
    for (int i = 0; i < n; i++) {
        if (ci_eq_gal(reg[i], cl)) {
            if (clean_out && max) snprintf(clean_out, max, "%s", reg[i]);
            return 0;
        }
    }
    if (n >= GALLERY_MAX_FOLDERS) return -1;
    snprintf(reg[n++], 64, "%s", cl);
    save_registered_folders(reg, n);
    g_gal_rev++;
    return 0;
}

int gallery_delete_folder(const char* name, int delete_images) {
    char cl[64];
    if (gallery_clean_folder_name(name, cl, sizeof(cl)) <= 0) return -1;

    char reg[GALLERY_MAX_FOLDERS][64];
    int n = load_registered_folders(reg, GALLERY_MAX_FOLDERS);
    int out_n = 0;
    for (int i = 0; i < n; i++) {
        if (!ci_eq_gal(reg[i], cl)) {
            if (out_n != i) memcpy(reg[out_n], reg[i], 64);
            out_n++;
        }
    }
    save_registered_folders(reg, out_n);

    char (*ids)[64] = NULL;
    size_t cnt = scan_sorted_gallery_ids(&ids);
    for (size_t i = 0; i < cnt; i++) {
        GalleryItem it;
        if (gallery_get(ids[i], &it) == 0 && ci_eq_gal(it.folder, cl)) {
            if (delete_images) {
                gallery_delete(ids[i]);
            } else {
                it.folder[0] = 0;
                write_meta_item(ids[i], &it);
            }
        }
    }
    free(ids);
    g_gal_rev++;
    return 0;
}

int gallery_list_folders(GalleryFolder* out, int max_folders, int* out_unsorted, int* out_starred, int* out_total) {
    if (out_unsorted) *out_unsorted = 0;
    if (out_starred) *out_starred = 0;
    if (out_total) *out_total = 0;
    if (!out || max_folders <= 0) return 0;

    char reg[GALLERY_MAX_FOLDERS][64];
    int n_reg = load_registered_folders(reg, GALLERY_MAX_FOLDERS);
    int nf = 0;
    for (int i = 0; i < n_reg && nf < max_folders; i++) {
        snprintf(out[nf].name, sizeof(out[nf].name), "%s", reg[i]);
        out[nf].count = 0;
        nf++;
    }

    char (*ids)[64] = NULL;
    size_t cnt = scan_sorted_gallery_ids(&ids);
    if (out_total) *out_total = (int)cnt;
    int unsorted = 0, starred = 0;
    for (size_t i = 0; i < cnt; i++) {
        GalleryItem it;
        if (gallery_get(ids[i], &it) != 0) continue;
        if (it.starred) starred++;
        if (!it.folder[0]) {
            unsorted++;
        } else {
            int idx = -1;
            for (int k = 0; k < nf; k++) {
                if (ci_eq_gal(out[k].name, it.folder)) { idx = k; break; }
            }
            if (idx >= 0) out[idx].count++;
            else if (nf < max_folders) {
                snprintf(out[nf].name, sizeof(out[nf].name), "%s", it.folder);
                out[nf].count = 1;
                nf++;
            }
        }
    }
    free(ids);
    if (out_unsorted) *out_unsorted = unsorted;
    if (out_starred) *out_starred = starred;
    if (nf > 1) qsort(out, (size_t)nf, sizeof(GalleryFolder), cmp_folder_asc);
    return nf;
}

int gallery_set_folder(const char* id, const char* folder) {
    GalleryItem it;
    if (gallery_get(id, &it) != 0) return -1;
    char cl[64] = {0};
    if (folder && folder[0]) {
        if (gallery_clean_folder_name(folder, cl, sizeof(cl)) <= 0) return -1;
        gallery_create_folder(cl, cl, sizeof(cl));
    }
    snprintf(it.folder, sizeof(it.folder), "%s", cl);
    if (write_meta_item(id, &it) != 0) return -1;
    g_gal_rev++;
    return 0;
}

int gallery_set_folder_batch(const char (*ids)[64], int n_ids, const char* folder) {
    if (!ids || n_ids <= 0) return -1;
    char cl[64] = {0};
    if (folder && folder[0]) {
        if (gallery_clean_folder_name(folder, cl, sizeof(cl)) <= 0) return -1;
        gallery_create_folder(cl, cl, sizeof(cl));
    }
    int moved = 0;
    for (int i = 0; i < n_ids; i++) {
        GalleryItem it;
        if (gallery_get(ids[i], &it) != 0) continue;
        snprintf(it.folder, sizeof(it.folder), "%s", cl);
        if (write_meta_item(ids[i], &it) == 0) moved++;
    }
    if (moved > 0) g_gal_rev++;
    return moved;
}

int gallery_set_starred(const char* id, int starred) {
    GalleryItem it;
    if (gallery_get(id, &it) != 0) return -1;
    it.starred = starred ? 1 : 0;
    if (write_meta_item(id, &it) != 0) return -1;
    g_gal_rev++;
    return 0;
}

static void zip_put16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void zip_put32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

// Fast zero-compression PKZIP Store archive builder (PNGs are already compressed; <50ms on PS4!)
uint8_t* gallery_build_zip(const char (*ids)[64], int n_ids, const char* folder_filter, size_t* out_zip_size) {
    if (out_zip_size) *out_zip_size = 0;

    char (*alloc_ids)[64] = NULL;
    const char (*use_ids)[64] = ids;
    int count = n_ids;

    if ((!ids || n_ids <= 0) && folder_filter && folder_filter[0]) {
        char (*all)[64] = NULL;
        size_t tot = scan_sorted_gallery_ids(&all);
        if (!all || tot == 0) { free(all); return NULL; }
        alloc_ids = (char (*)[64])malloc(tot * sizeof(*alloc_ids));
        if (!alloc_ids) { free(all); return NULL; }
        count = 0;
        int want_starred = (strcmp(folder_filter, "__starred__") == 0);
        int want_unsorted = (strcmp(folder_filter, "__unsorted__") == 0);
        for (size_t i = 0; i < tot; i++) {
            GalleryItem it;
            if (gallery_get(all[i], &it) != 0) continue;
            int ok = want_starred ? it.starred
                   : (want_unsorted ? (it.folder[0] == 0) : ci_eq_gal(it.folder, folder_filter));
            if (ok) memcpy(alloc_ids[count++], all[i], 64);
        }
        free(all);
        use_ids = (const char (*)[64])alloc_ids;
    }
    if (!use_ids || count <= 0) { free(alloc_ids); return NULL; }
    if (count > 256) count = 256;

    typedef struct {
        char fname[128];
        uint32_t crc, sz, local_off;
    } ZipRec;
    ZipRec* recs = (ZipRec*)calloc((size_t)count, sizeof(ZipRec));
    if (!recs) { free(alloc_ids); return NULL; }

    size_t cap = 4 * 1024 * 1024;
    const size_t MAX_ZIP_BYTES = 192u * 1024u * 1024u; // Safe 192 MiB RAM ceiling on PS4
    uint8_t* zbuf = (uint8_t*)malloc(cap);
    if (!zbuf) { free(recs); free(alloc_ids); return NULL; }

    size_t pos = 0;
    int packed = 0;
    for (int i = 0; i < count; i++) {
        if (!gallery_valid_id(use_ids[i])) continue;
        size_t psz = 0;
        uint8_t* png = gallery_load_png(use_ids[i], &psz);
        if (!png || psz == 0) { free(png); continue; }
        GalleryItem it;
        char prefix[72] = {0};
        if (gallery_get(use_ids[i], &it) == 0 && it.folder[0]) {
            snprintf(prefix, sizeof(prefix), "%s/", it.folder);
        }
        char fn[128];
        snprintf(fn, sizeof(fn), "%s%s.png", prefix, use_ids[i]);
        uint16_t fn_len = (uint16_t)strlen(fn);

        size_t need = 30u + fn_len + psz + (size_t)(packed + 2) * 180u + 64u;
        if (pos + need > MAX_ZIP_BYTES) { free(png); break; }
        while (pos + need > cap) {
            size_t ncap = cap * 2;
            if (ncap > MAX_ZIP_BYTES) ncap = MAX_ZIP_BYTES;
            uint8_t* nb = (uint8_t*)realloc(zbuf, ncap);
            if (!nb) { free(png); goto finish_cd; }
            zbuf = nb;
            cap = ncap;
        }

        uint32_t crc = png_crc32(png, psz);
        uint32_t loff = (uint32_t)pos;
        zip_put32(zbuf + pos, 0x04034b50u);
        zip_put16(zbuf + pos + 4, 20);
        zip_put16(zbuf + pos + 6, 0);
        zip_put16(zbuf + pos + 8, 0); // Store (0 = uncompressed)
        zip_put16(zbuf + pos + 10, 0);
        zip_put16(zbuf + pos + 12, 0x0021);
        zip_put32(zbuf + pos + 14, crc);
        zip_put32(zbuf + pos + 18, (uint32_t)psz);
        zip_put32(zbuf + pos + 22, (uint32_t)psz);
        zip_put16(zbuf + pos + 26, fn_len);
        zip_put16(zbuf + pos + 28, 0);
        pos += 30;
        memcpy(zbuf + pos, fn, fn_len);
        pos += fn_len;
        memcpy(zbuf + pos, png, psz);
        pos += psz;
        free(png);

        snprintf(recs[packed].fname, sizeof(recs[0].fname), "%s", fn);
        recs[packed].crc = crc;
        recs[packed].sz = (uint32_t)psz;
        recs[packed].local_off = loff;
        packed++;
    }

finish_cd:
    free(alloc_ids);
    if (packed == 0) { free(recs); free(zbuf); return NULL; }

    uint32_t cd_off = (uint32_t)pos;
    for (int i = 0; i < packed; i++) {
        uint16_t fn_len = (uint16_t)strlen(recs[i].fname);
        zip_put32(zbuf + pos, 0x02014b50u);
        zip_put16(zbuf + pos + 4, 20);
        zip_put16(zbuf + pos + 6, 20);
        zip_put16(zbuf + pos + 8, 0);
        zip_put16(zbuf + pos + 10, 0);
        zip_put16(zbuf + pos + 12, 0);
        zip_put16(zbuf + pos + 14, 0x0021);
        zip_put32(zbuf + pos + 16, recs[i].crc);
        zip_put32(zbuf + pos + 20, recs[i].sz);
        zip_put32(zbuf + pos + 24, recs[i].sz);
        zip_put16(zbuf + pos + 28, fn_len);
        zip_put16(zbuf + pos + 30, 0);
        zip_put16(zbuf + pos + 32, 0);
        zip_put16(zbuf + pos + 34, 0);
        zip_put16(zbuf + pos + 36, 0);
        zip_put32(zbuf + pos + 38, 0);
        zip_put32(zbuf + pos + 42, recs[i].local_off);
        pos += 46;
        memcpy(zbuf + pos, recs[i].fname, fn_len);
        pos += fn_len;
    }
    uint32_t cd_sz = (uint32_t)(pos - cd_off);
    zip_put32(zbuf + pos, 0x06054b50u);
    zip_put16(zbuf + pos + 4, 0);
    zip_put16(zbuf + pos + 6, 0);
    zip_put16(zbuf + pos + 8, (uint16_t)packed);
    zip_put16(zbuf + pos + 10, (uint16_t)packed);
    zip_put32(zbuf + pos + 12, cd_sz);
    zip_put32(zbuf + pos + 16, cd_off);
    zip_put16(zbuf + pos + 20, 0);
    pos += 22;

    free(recs);
    if (out_zip_size) *out_zip_size = pos;
    return zbuf;
}

// ---- User Presets -------------------------------------------------------------------------------

static int cmp_preset_asc(const void* a, const void* b) {
    return strcmp(((const UserPreset*)a)->name, ((const UserPreset*)b)->name);
}

static void parse_preset_buf(char* buf, UserPreset* p) {
    char* line = buf;
    while (*line) {
        char* nl = strchr(line, '\n');
        if (nl) *nl = 0;
        size_t len = strlen(line);
        while (len > 0 && line[len - 1] == '\r') line[--len] = 0;
        char* eq = strchr(line, '=');
        if (eq) {
            *eq = 0;
            const char* k = line;
            const char* v = eq + 1;
            if (strcmp(k, "name") == 0) snprintf(p->name, sizeof(p->name), "%s", v);
            else if (strcmp(k, "model") == 0) snprintf(p->model, sizeof(p->model), "%s", v);
            else if (strcmp(k, "prompt") == 0) snprintf(p->prompt, sizeof(p->prompt), "%s", v);
            else if (strcmp(k, "negative") == 0) snprintf(p->negative, sizeof(p->negative), "%s", v);
            else if (strcmp(k, "sampler") == 0) snprintf(p->sampler, sizeof(p->sampler), "%s", v);
            else if (strcmp(k, "width") == 0) p->width = (uint32_t)atoi(v);
            else if (strcmp(k, "height") == 0) p->height = (uint32_t)atoi(v);
            else if (strcmp(k, "steps") == 0) p->steps = (uint32_t)atoi(v);
            else if (strcmp(k, "cfg") == 0) p->cfg = (float)atof(v);
            else if (strcmp(k, "upscale") == 0) p->upscale = atoi(v);
            else if (strcmp(k, "eco") == 0) p->eco = atoi(v);
            else if (strcmp(k, "nsfw") == 0) p->nsfw = atoi(v) ? 1 : 0;
            else if (strcmp(k, "is_speed") == 0) p->is_speed = atoi(v) ? 1 : 0;
        }
        if (!nl) break;
        line = nl + 1;
    }
}

int presets_list(UserPreset* items, int max_items) {
    DIR* d = opendir(g_presets_dir);
    if (!d) return 0;
    int n = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL && n < max_items) {
        const char* nm = ent->d_name;
        size_t l = strlen(nm);
        if (l <= 4 || strcmp(nm + l - 4, ".cfg") != 0 || l - 4 >= 64) continue;
        char id[64];
        memcpy(id, nm, l - 4);
        id[l - 4] = 0;
        if (!gallery_valid_id(id)) continue;
        char fp[320];
        snprintf(fp, sizeof(fp), "%s/%s.cfg", g_presets_dir, id);
        size_t sz = 0;
        uint8_t* buf = read_all_file(fp, &sz);
        if (!buf) continue;
        UserPreset* p = &items[n];
        memset(p, 0, sizeof(*p));
        snprintf(p->id, sizeof(p->id), "%s", id);
        snprintf(p->name, sizeof(p->name), "%s", id);
        snprintf(p->sampler, sizeof(p->sampler), "Euler a");
        p->width = 768; p->height = 1024; p->steps = 20; p->cfg = 7.0f; p->eco = 2;
        parse_preset_buf((char*)buf, p);
        free(buf);
        n++;
    }
    closedir(d);
    if (n > 1) qsort(items, (size_t)n, sizeof(UserPreset), cmp_preset_asc);
    return n;
}

int presets_save(const UserPreset* p, char* out_id, size_t out_id_max) {
    if (!p || !p->name[0]) return -1;
    ensure_dir(g_presets_dir);
    char id[64];
    if (p->id[0] && gallery_valid_id(p->id)) snprintf(id, sizeof(id), "%s", p->id);
    else make_safe_id(p->name, id, sizeof(id));

    char nm[64], pr[1024], ng[512];
    snprintf(nm, sizeof(nm), "%s", p->name); strip_newlines(nm);
    snprintf(pr, sizeof(pr), "%s", p->prompt); strip_newlines(pr);
    snprintf(ng, sizeof(ng), "%s", p->negative); strip_newlines(ng);
    char buf[2300];
    int n = snprintf(buf, sizeof(buf),
                     "name=%s\nmodel=%s\nwidth=%u\nheight=%u\nsteps=%u\ncfg=%.2f\nsampler=%s\n"
                     "upscale=%d\neco=%d\nnsfw=%d\nis_speed=%d\nprompt=%s\nnegative=%s\n",
                     nm, p->model, p->width ? p->width : 768u, p->height ? p->height : 1024u,
                     p->steps ? p->steps : 20u, (double)(p->cfg > 0 ? p->cfg : 7.0f),
                     p->sampler[0] ? p->sampler : "Euler a", p->upscale, p->eco,
                     p->nsfw ? 1 : 0, p->is_speed ? 1 : 0, pr, ng);
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.cfg", g_presets_dir, id);
    if (write_all_file(fp, buf, (size_t)(n > 0 ? n : 0)) != 0) return -1;
    if (out_id && out_id_max) snprintf(out_id, out_id_max, "%s", id);
    return 0;
}

int presets_delete(const char* id) {
    if (!gallery_valid_id(id)) return -1;
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.cfg", g_presets_dir, id);
    return unlink(fp);
}

// ---- Art Styles Manager -------------------------------------------------------------------------

static int cmp_style_asc(const void* a, const void* b) {
    return strcmp(((const StyleItem*)a)->name, ((const StyleItem*)b)->name);
}

static void seed_default_styles(void) {
    char marker[320];
    snprintf(marker, sizeof(marker), "%s/.seeded_v2", g_styles_dir);
    if (path_exists_local(marker)) return;

    static const StyleItem DEFAULTS[] = {
        { "cel_shaded", "Anime Cel-Shaded",
          "anime style, crisp lineart, cel shading, clean vibrant colors, sharp eyes, studio anime key visual",
          "3d, photorealistic, blurry, muddy shading, sketch" },
        { "vibrant_illus", "Vibrant Illustration",
          "detailed anime illustration, dramatic rim lighting, glowing particles, rich color palette, high contrast, glossy hair",
          "washed out, flat colors, low contrast, monochrome" },
        { "soft_pastel", "Soft Pastel Dream",
          "pastel colors, soft diffused lighting, dreamy atmosphere, delicate lineart, gentle bloom, ethereal",
          "harsh shadows, dark, gritty, oversaturated" },
        { "cinematic_night", "Cinematic Neon Night",
          "cinematic lighting, neon reflections, night sky, bokeh background, volumetric light, depth of field",
          "flat lighting, plain white background, dull" },
        { "retro_90s", "90s Retro OVA Anime",
          "1990s anime style, retro artstyle, cel animation, subtle film grain, classic OVA aesthetic, warm tones",
          "modern 3d, cgi, plastic skin" },
        { "watercolor_ink", "Watercolor & Ink",
          "watercolor painting, delicate ink outlines, textured paper, soft color washes, painterly anime",
          "3d render, harsh digital lines, photo" },
        { "dark_fantasy", "Dark Fantasy Epic",
          "dark fantasy anime, dramatic chiaroscuro, glowing magic aura, intricate armor and drapery, epic atmosphere",
          "bright cheerful, cartoon, simple background" },
        { "high_detail_portrait", "High-Detail Portrait (Clean Eyes)",
          "highly detailed face, expressive sparkling symmetrical eyes, crisp pupils, intricate hair strands, upper body portrait, crisp focus",
          "blurry face, bad eyes, asymmetrical eyes, cross-eyed, deformed hands, extra fingers, bad anatomy" },
        { "shinkai_scenic", "Makoto Shinkai Scenic",
          "Makoto Shinkai anime background style, towering luminous clouds, dramatic lens flare, rich twilight gradient, sparkling light reflections, cinematic anime movie",
          "flat sky, muddy colors, low detail, sketch" },
        { "ghibli_painterly", "Studio Ghibli Painterly",
          "Studio Ghibli style, hand-painted gouache background, lush green foliage, warm nostalgic summer sunlight, fluffy cumulus clouds, storybook charm",
          "3d, cgi, cyberpunk, harsh neon, dark gritty" },
        { "anti_uncanny_2d", "2D Anime Fix (For DreamShaper)",
          "2d anime illustration, cel shaded anime art, crisp ink outlines, expressive anime eyes, vibrant clean colors",
          "photorealistic, realistic, 3d, render, doll, uncanny, photography, plastic skin, real human" },
        { "glossy_key_visual", "Glossy Movie Key Visual",
          "official anime movie key visual, ultra crisp lineart, luminous rim light, sparkling bokeh, dramatic composition, vibrant studio coloring",
          "rough sketch, unfinished, lowres, dull colors" },
        { "manga_monochrome", "Monochrome Manga Ink",
          "black and white manga panel, crisp ink lineart, classic screentone shading, dramatic speed lines, high contrast Japanese comic art",
          "color, blurry, 3d, watercolor, photo" },
        { "chibi_sticker", "Chibi Cute Sticker",
          "chibi anime style, adorable big sparkling eyes, cute rounded proportions, clean bold outlines, cheerful pastel sticker art",
          "realistic, tall proportions, dark gritty, scary" },
        { "cyberpunk_holo", "Cyberpunk Hologram",
          "cyberpunk anime aesthetic, glowing cyan and magenta neon reflections, holographic HUD particles, rainy night glass, futuristic sci-fi atmosphere",
          "medieval, rustic, daytime plain background" },
        { "ukiyo_e", "Ukiyo-e Woodblock Print",
          "traditional Japanese ukiyo-e woodblock print style, bold ink contours, textured washi paper, stylized ocean waves and clouds, indigo and vermilion palette",
          "3d, cgi, modern neon, photorealistic" },
        { "oil_painting", "Painterly Oil Fantasy",
          "fine art oil painting, expressive visible brushwork, baroque golden chiaroscuro lighting, rich textured canvas, epic fantasy concept art",
          "flat vector, cel shaded, cartoon, plastic" },
        { "flat_vector_poster", "Flat Graphic Poster",
          "modern Japanese graphic poster design, bold clean color blocking, crisp vector shapes, stylish minimalist shadows, vibrant pop art",
          "muddy texture, photorealistic, blurry, noisy" },
        { "golden_hour", "Golden Hour Backlight",
          "warm golden hour sunset light, glowing backlit hair strands, soft amber sunbeams, gentle lens bloom, nostalgic cinema mood",
          "cold fluorescent light, overcast gray, flat shading" },
        { "cozy_lofi", "Cozy Lo-Fi Evening",
          "lo-fi anime aesthetic, warm desk lamp glow, soft twilight window view, gentle analog film grain, peaceful cozy atmosphere",
          "harsh spotlight, chaotic action, oversaturated" },
        { "pixel_16bit", "16-Bit Retro Pixel Art",
          "crisp 16-bit pixel art, classic JRPG portrait aesthetic, clean pixel clusters, vibrant limited color palette, nostalgic retro game art",
          "blurry, smooth airbrush, 3d render, photorealistic" },
        { "fullbody_sheet", "Clean Full-Body Stance",
          "full body shot, head to toe, well-proportioned anatomy, symmetrical eyes, natural hands and fingers, clean dynamic stance, crisp focus",
          "cropped feet, cropped head, bad anatomy, extra fingers, deformed hands, asymmetrical eyes" },
    };
    for (size_t i = 0; i < sizeof(DEFAULTS) / sizeof(DEFAULTS[0]); i++) {
        char fp[320];
        snprintf(fp, sizeof(fp), "%s/%s.cfg", g_styles_dir, DEFAULTS[i].id);
        if (!path_exists_local(fp)) styles_save(&DEFAULTS[i], NULL, 0);
    }
    write_all_file(marker, "2\n", 2);
}

void styles_init(const char* styles_dir) {
    if (styles_dir && styles_dir[0]) snprintf(g_styles_dir, sizeof(g_styles_dir), "%s", styles_dir);
    ensure_dir(g_styles_dir);
    seed_default_styles();
}

int styles_list(StyleItem* items, int max_items) {
    DIR* d = opendir(g_styles_dir);
    if (!d) return 0;
    int n = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL && n < max_items) {
        const char* nm = ent->d_name;
        size_t l = strlen(nm);
        if (l <= 4 || strcmp(nm + l - 4, ".cfg") != 0 || l - 4 >= 64) continue;
        char id[64];
        memcpy(id, nm, l - 4);
        id[l - 4] = 0;
        if (!gallery_valid_id(id)) continue;
        char fp[320];
        snprintf(fp, sizeof(fp), "%s/%s.cfg", g_styles_dir, id);
        size_t sz = 0;
        uint8_t* buf = read_all_file(fp, &sz);
        if (!buf) continue;
        StyleItem* s = &items[n];
        memset(s, 0, sizeof(*s));
        snprintf(s->id, sizeof(s->id), "%s", id);
        snprintf(s->name, sizeof(s->name), "%s", id);
        char* line = (char*)buf;
        while (*line) {
            char* nl = strchr(line, '\n');
            if (nl) *nl = 0;
            size_t len = strlen(line);
            while (len > 0 && line[len - 1] == '\r') line[--len] = 0;
            char* eq = strchr(line, '=');
            if (eq) {
                *eq = 0;
                if (strcmp(line, "name") == 0) snprintf(s->name, sizeof(s->name), "%s", eq + 1);
                else if (strcmp(line, "prompt") == 0) snprintf(s->prompt, sizeof(s->prompt), "%s", eq + 1);
                else if (strcmp(line, "negative") == 0) snprintf(s->negative, sizeof(s->negative), "%s", eq + 1);
            }
            if (!nl) break;
            line = nl + 1;
        }
        free(buf);
        n++;
    }
    closedir(d);
    if (n > 1) qsort(items, (size_t)n, sizeof(StyleItem), cmp_style_asc);
    return n;
}

int styles_save(const StyleItem* s, char* out_id, size_t out_id_max) {
    if (!s || !s->name[0]) return -1;
    ensure_dir(g_styles_dir);
    char id[64];
    if (s->id[0] && gallery_valid_id(s->id)) snprintf(id, sizeof(id), "%s", s->id);
    else make_safe_id(s->name, id, sizeof(id));
    char nm[64], pr[512], ng[384];
    snprintf(nm, sizeof(nm), "%s", s->name); strip_newlines(nm);
    snprintf(pr, sizeof(pr), "%s", s->prompt); strip_newlines(pr);
    snprintf(ng, sizeof(ng), "%s", s->negative); strip_newlines(ng);
    char buf[1200];
    int n = snprintf(buf, sizeof(buf), "name=%s\nprompt=%s\nnegative=%s\n", nm, pr, ng);
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.cfg", g_styles_dir, id);
    if (write_all_file(fp, buf, (size_t)(n > 0 ? n : 0)) != 0) return -1;
    if (out_id && out_id_max) snprintf(out_id, out_id_max, "%s", id);
    return 0;
}

int styles_delete(const char* id) {
    if (!gallery_valid_id(id)) return -1;
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.cfg", g_styles_dir, id);
    return unlink(fp);
}

// ---- Character Presets Manager ------------------------------------------------------------------

static int cmp_char_asc(const void* a, const void* b) {
    const CharacterItem* ca = (const CharacterItem*)a;
    const CharacterItem* cb = (const CharacterItem*)b;
    int s = strcmp(ca->series, cb->series);
    return s != 0 ? s : strcmp(ca->name, cb->name);
}

static void seed_default_characters(void) {
    gd_init_preset_seeds();
    size_t cnt = sizeof(SEED_CHARS) / sizeof(SEED_CHARS[0]);
    uint32_t h = 2166136261u ^ (uint32_t)cnt;
    for (size_t i = 0; i < cnt; i++) {
        h = fnv1a_str(h, SEED_CHARS[i].id);
        h = fnv1a_str(h, SEED_CHARS[i].name);
        h = fnv1a_str(h, SEED_CHARS[i].series);
        h = fnv1a_str(h, SEED_CHARS[i].prompt);
        h = fnv1a_str(h, SEED_CHARS[i].negative);
    }
    char sig[48];
    int sn = snprintf(sig, sizeof(sig), "%u:%08x\n", (unsigned)cnt, h);
    char marker[320];
    snprintf(marker, sizeof(marker), "%s/.seeded_sig", g_chars_dir);
    if (seed_sig_matches(marker, sig)) return;

    for (size_t i = 0; i < cnt; i++) {
        characters_save(&SEED_CHARS[i], NULL, 0);
    }
    write_all_file(marker, sig, (size_t)(sn > 0 ? sn : 0));
}

void characters_init(const char* chars_dir) {
    if (chars_dir && chars_dir[0]) snprintf(g_chars_dir, sizeof(g_chars_dir), "%s", chars_dir);
    ensure_dir(g_chars_dir);
    seed_default_characters();
}

int characters_list(CharacterItem* items, int max_items) {
    DIR* d = opendir(g_chars_dir);
    if (!d) return 0;
    int n = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL && n < max_items) {
        const char* nm = ent->d_name;
        size_t l = strlen(nm);
        if (l <= 4 || strcmp(nm + l - 4, ".cfg") != 0 || l - 4 >= 64) continue;
        char id[64];
        memcpy(id, nm, l - 4);
        id[l - 4] = 0;
        if (!gallery_valid_id(id)) continue;
        char fp[320];
        snprintf(fp, sizeof(fp), "%s/%s.cfg", g_chars_dir, id);
        size_t sz = 0;
        uint8_t* buf = read_all_file(fp, &sz);
        if (!buf) continue;
        CharacterItem* c = &items[n];
        memset(c, 0, sizeof(*c));
        snprintf(c->id, sizeof(c->id), "%s", id);
        snprintf(c->name, sizeof(c->name), "%s", id);
        snprintf(c->series, sizeof(c->series), "Custom");
        char* line = (char*)buf;
        while (*line) {
            char* nl = strchr(line, '\n');
            if (nl) *nl = 0;
            size_t len = strlen(line);
            while (len > 0 && line[len - 1] == '\r') line[--len] = 0;
            char* eq = strchr(line, '=');
            if (eq) {
                *eq = 0;
                if (strcmp(line, "name") == 0) snprintf(c->name, sizeof(c->name), "%s", eq + 1);
                else if (strcmp(line, "series") == 0) snprintf(c->series, sizeof(c->series), "%s", eq + 1);
                else if (strcmp(line, "prompt") == 0) snprintf(c->prompt, sizeof(c->prompt), "%s", eq + 1);
                else if (strcmp(line, "negative") == 0) snprintf(c->negative, sizeof(c->negative), "%s", eq + 1);
            }
            if (!nl) break;
            line = nl + 1;
        }
        free(buf);
        n++;
    }
    closedir(d);
    if (n > 1) qsort(items, (size_t)n, sizeof(CharacterItem), cmp_char_asc);
    return n;
}

int characters_save(const CharacterItem* c, char* out_id, size_t out_id_max) {
    if (!c || !c->name[0]) return -1;
    ensure_dir(g_chars_dir);
    char id[64];
    if (c->id[0] && gallery_valid_id(c->id)) snprintf(id, sizeof(id), "%s", c->id);
    else make_safe_id(c->name, id, sizeof(id));
    char nm[64], sr[48], pr[512], ng[384];
    snprintf(nm, sizeof(nm), "%s", c->name); strip_newlines(nm);
    snprintf(sr, sizeof(sr), "%s", c->series[0] ? c->series : "Custom"); strip_newlines(sr);
    snprintf(pr, sizeof(pr), "%s", c->prompt); strip_newlines(pr);
    snprintf(ng, sizeof(ng), "%s", c->negative); strip_newlines(ng);
    char buf[1300];
    int n = snprintf(buf, sizeof(buf), "name=%s\nseries=%s\nprompt=%s\nnegative=%s\n", nm, sr, pr, ng);
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.cfg", g_chars_dir, id);
    if (write_all_file(fp, buf, (size_t)(n > 0 ? n : 0)) != 0) return -1;
    if (out_id && out_id_max) snprintf(out_id, out_id_max, "%s", id);
    return 0;
}

int characters_delete(const char* id) {
    if (!gallery_valid_id(id)) return -1;
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.cfg", g_chars_dir, id);
    return unlink(fp);
}

int model_preset_save(const char* models_root, const char* model_name,
                      int is_sd15, uint32_t width, uint32_t height, uint32_t steps,
                      float cfg, const char* sampler, int upscale,
                      const char* description, const char* prompt, const char* negative) {
    if (!models_root || !model_name || !gallery_valid_id(model_name)) return -1;
    char desc[96], pr[512], ng[384];
    snprintf(desc, sizeof(desc), "%s", description ? description : ""); strip_newlines(desc);
    snprintf(pr, sizeof(pr), "%s", prompt ? prompt : ""); strip_newlines(pr);
    snprintf(ng, sizeof(ng), "%s", negative ? negative : ""); strip_newlines(ng);
    char buf[1600];
    int n = snprintf(buf, sizeof(buf),
                     "custom=1\narch=%s\nwidth=%u\nheight=%u\nsteps=%u\ncfg=%.2f\nsampler=%s\nupscale=%d\n"
                     "description=%s\nprompt=%s\nnegative=%s\n",
                     is_sd15 ? "sd15" : "sdxl", width, height, steps, (double)cfg,
                     (sampler && sampler[0]) ? sampler : "Euler a", upscale, desc, pr, ng);
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s/preset.cfg", models_root, model_name);
    return write_all_file(fp, buf, (size_t)(n > 0 ? n : 0));
}
