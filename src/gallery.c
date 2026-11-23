// Garlic Diffusion: Persistent Image Gallery & User/Model Preset Manager
// Uses raw POSIX open/read/write/unlink + opendir for full OpenOrbis PS4 compatibility.

#include "gallery.h"
#include "png_write.h"

#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#include <direct.h>
#define mkdir(p, m) _mkdir(p)
#endif

static char g_out_dir[256] = "/data/garlic-diffusion/out";
static char g_presets_dir[256] = "/data/garlic-diffusion/presets";
static char g_styles_dir[256] = "/data/garlic-diffusion/styles";
static uint32_t g_next_seq = 1;

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

int gallery_valid_id(const char* id) {
    if (!id || !id[0] || strlen(id) > 60) return 0;
    for (const char* p = id; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (!(isalnum(c) || c == '_' || c == '-')) return 0;
    }
    return 1;
}

static void seed_default_presets(void) {
    DIR* d = opendir(g_presets_dir);
    int count = 0;
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL) {
            if (strstr(ent->d_name, ".cfg")) count++;
        }
        closedir(d);
    }
    if (count > 0) return;

    static const UserPreset DEFAULTS[] = {
        { "sakura_night", "Sakura Night", "",
          "masterpiece, best quality, amazing quality, 1girl, silver hair, blue eyes, glowing sakura petals, night sky, anime style",
          "lowres, bad anatomy, bad hands, text, error, blurry, worst quality, low quality",
          "Euler a", 768, 1024, 20, 7.0f, 0, 2 },
        { "cyberpunk_samurai", "Cyberpunk Samurai", "",
          "masterpiece, best quality, 1girl, cyberpunk samurai, neon Tokyo street, rain, katana, reflections, dramatic lighting, anime",
          "lowres, bad anatomy, bad hands, text, error, blurry, worst quality, low quality",
          "Euler a", 768, 1024, 20, 7.0f, 0, 2 },
        { "fantasy_wizard", "Fantasy Wizard", "",
          "masterpiece, best quality, 1girl, wizard hat, glowing crystal staff, enchanted forest, fireflies, detailed anime illustration",
          "lowres, bad anatomy, bad hands, text, error, blurry, worst quality, low quality",
          "Euler a", 768, 1024, 20, 7.0f, 0, 2 },
        { "seaside_sunset", "Seaside Sunset", "",
          "masterpiece, best quality, 1girl, seaside sunset, white sundress, wind blowing hair, golden ocean light, cinematic anime",
          "lowres, bad anatomy, bad hands, text, error, blurry, worst quality, low quality",
          "Euler a", 768, 1024, 20, 7.0f, 0, 2 },
        { "cozy_cafe", "Cozy Cafe", "",
          "masterpiece, best quality, 1girl, cozy anime cafe, holding coffee cup, warm sunlight through window, soft bokeh, detailed",
          "lowres, bad anatomy, bad hands, text, error, blurry, worst quality, low quality",
          "Euler a", 512, 768, 20, 7.0f, 1, 2 },
        { "wallpaper_1080p", "1080p Anime Wallpaper", "",
          "masterpiece, best quality, amazing quality, wide shot, scenic anime landscape, floating castle in clouds, sunset sky, vibrant colors",
          "lowres, bad anatomy, text, watermark, blurry, worst quality, low quality",
          "Euler", 896, 512, 8, 3.5f, 1, 2 },
    };
    for (size_t i = 0; i < sizeof(DEFAULTS) / sizeof(DEFAULTS[0]); i++) {
        presets_save(&DEFAULTS[i], NULL, 0);
    }
}

void gallery_init(const char* out_dir, const char* presets_dir) {
    if (out_dir && out_dir[0]) snprintf(g_out_dir, sizeof(g_out_dir), "%s", out_dir);
    if (presets_dir && presets_dir[0]) snprintf(g_presets_dir, sizeof(g_presets_dir), "%s", presets_dir);
    ensure_dir(g_out_dir);
    ensure_dir(g_presets_dir);

    // Find max existing g_NNNN sequence so new images sort cleanly on top
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
            else if (strcmp(k, "width") == 0) it->width = (uint32_t)atoi(v);
            else if (strcmp(k, "height") == 0) it->height = (uint32_t)atoi(v);
            else if (strcmp(k, "steps") == 0) it->steps = (uint32_t)atoi(v);
            else if (strcmp(k, "cfg") == 0) it->cfg = (float)atof(v);
            else if (strcmp(k, "seed") == 0) it->seed = strtoull(v, NULL, 10);
            else if (strcmp(k, "eco") == 0) it->eco = atoi(v);
            else if (strcmp(k, "upscaled") == 0) it->upscaled = atoi(v);
            else if (strcmp(k, "sec") == 0) it->sec = atof(v);
        }
        if (!nl) break;
        line = nl + 1;
    }
}

int gallery_save(const uint8_t* png, size_t png_size, const GalleryItem* meta, char* out_id, size_t out_id_max) {
    if (!png || !png_size || !meta) return -1;
    ensure_dir(g_out_dir);

    char id[64];
    snprintf(id, sizeof(id), "g_%05u_%llu", g_next_seq++, (unsigned long long)(meta->seed & 0xFFFFFFFFull));

    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.png", g_out_dir, id);
    if (write_all_file(fp, png, png_size) != 0) return -1;

    char pr[1024], ng[512];
    snprintf(pr, sizeof(pr), "%s", meta->prompt); strip_newlines(pr);
    snprintf(ng, sizeof(ng), "%s", meta->negative); strip_newlines(ng);
    char mbuf[2200];
    int n = snprintf(mbuf, sizeof(mbuf),
                     "id=%s\nmodel=%s\nwidth=%u\nheight=%u\nsteps=%u\ncfg=%.2f\nsampler=%s\n"
                     "seed=%llu\neco=%d\nupscaled=%d\nsec=%.2f\nprompt=%s\nnegative=%s\n",
                     id, meta->model, meta->width, meta->height, meta->steps, (double)meta->cfg,
                     meta->sampler[0] ? meta->sampler : "Euler a",
                     (unsigned long long)meta->seed, meta->eco, meta->upscaled, meta->sec, pr, ng);
    snprintf(fp, sizeof(fp), "%s/%s.meta", g_out_dir, id);
    write_all_file(fp, mbuf, (size_t)(n > 0 ? n : 0));

    if (out_id && out_id_max) snprintf(out_id, out_id_max, "%s", id);
    return 0;
}

int gallery_get(const char* id, GalleryItem* out) {
    if (!gallery_valid_id(id) || !out) return -1;
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

static int cmp_gallery_desc(const void* a, const void* b) {
    const GalleryItem* ga = (const GalleryItem*)a;
    const GalleryItem* gb = (const GalleryItem*)b;
    unsigned sa = 0, sb = 0;
    int ha = (sscanf(ga->id, "g_%u_", &sa) == 1);
    int hb = (sscanf(gb->id, "g_%u_", &sb) == 1);
    // New sequenced gallery images (g_NNNNN_*) ALWAYS sort newest-first ahead of legacy img_* files!
    if (ha && hb) {
        if (sb != sa) return (sb > sa) ? 1 : -1;
        return strcmp(gb->id, ga->id);
    }
    if (ha != hb) return hb - ha;
    return strcmp(gb->id, ga->id);
}

int gallery_list(GalleryItem* items, int max_items) {
    DIR* d = opendir(g_out_dir);
    if (!d) return 0;
    int n = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL && n < max_items) {
        const char* nm = ent->d_name;
        size_t l = strlen(nm);
        if (l <= 4 || strcmp(nm + l - 4, ".png") != 0 || l - 4 >= 64) continue;
        char id[64];
        memcpy(id, nm, l - 4);
        id[l - 4] = 0;
        if (!gallery_valid_id(id)) continue;
        if (gallery_get(id, &items[n]) == 0) n++;
    }
    closedir(d);
    if (n > 1) qsort(items, (size_t)n, sizeof(GalleryItem), cmp_gallery_desc);
    return n;
}

uint8_t* gallery_load_png(const char* id, size_t* out_size) {
    if (!gallery_valid_id(id)) return NULL;
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.png", g_out_dir, id);
    return read_all_file(fp, out_size);
}

int gallery_delete(const char* id) {
    if (!gallery_valid_id(id)) return -1;
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s.png", g_out_dir, id);
    int r = unlink(fp);
    snprintf(fp, sizeof(fp), "%s/%s.meta", g_out_dir, id);
    unlink(fp);
    return r;
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
    char buf[2200];
    int n = snprintf(buf, sizeof(buf),
                     "name=%s\nmodel=%s\nwidth=%u\nheight=%u\nsteps=%u\ncfg=%.2f\nsampler=%s\n"
                     "upscale=%d\neco=%d\nprompt=%s\nnegative=%s\n",
                     nm, p->model, p->width ? p->width : 768u, p->height ? p->height : 1024u,
                     p->steps ? p->steps : 20u, (double)(p->cfg > 0 ? p->cfg : 7.0f),
                     p->sampler[0] ? p->sampler : "Euler a", p->upscale, p->eco, pr, ng);
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
    DIR* d = opendir(g_styles_dir);
    int count = 0;
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL) if (strstr(ent->d_name, ".cfg")) count++;
        closedir(d);
    }
    if (count > 0) return;

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
        { "retro_90s", "90s Retro Anime",
          "1990s anime style, retro artstyle, cel animation, subtle film grain, classic OVA aesthetic, warm tones",
          "modern 3d, cgi, plastic skin" },
        { "watercolor_ink", "Watercolor & Ink",
          "watercolor painting, delicate ink outlines, textured paper, soft color washes, painterly anime",
          "3d render, harsh digital lines, photo" },
        { "dark_fantasy", "Dark Fantasy Epic",
          "dark fantasy anime, dramatic chiaroscuro, glowing magic aura, intricate armor and drapery, epic atmosphere",
          "bright cheerful, cartoon, simple background" },
        { "high_detail_portrait", "High-Detail Portrait",
          "highly detailed face, expressive sparkling eyes, intricate hair strands, upper body portrait, crisp focus",
          "blurry face, deformed eyes, bad anatomy, extra fingers" },
    };
    for (size_t i = 0; i < sizeof(DEFAULTS) / sizeof(DEFAULTS[0]); i++)
        styles_save(&DEFAULTS[i], NULL, 0);
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
                     "arch=%s\nwidth=%u\nheight=%u\nsteps=%u\ncfg=%.2f\nsampler=%s\nupscale=%d\n"
                     "description=%s\nprompt=%s\nnegative=%s\n",
                     is_sd15 ? "sd15" : "sdxl", width, height, steps, (double)cfg,
                     (sampler && sampler[0]) ? sampler : "Euler a", upscale, desc, pr, ng);
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s/preset.cfg", models_root, model_name);
    return write_all_file(fp, buf, (size_t)(n > 0 ? n : 0));
}
