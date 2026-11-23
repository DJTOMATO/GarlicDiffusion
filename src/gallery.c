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
static char g_chars_dir[256] = "/data/garlic-diffusion/characters";
static uint32_t g_next_seq = 1;

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
    char marker[320];
    snprintf(marker, sizeof(marker), "%s/.seeded_v2", g_presets_dir);
    if (path_exists_local(marker)) return;

    static const UserPreset DEFAULTS[] = {
        { "sakura_night", "Sakura Night Portrait", "",
          "masterpiece, best quality, absurdres, 1girl, solo, upper body, silver hair, blue eyes, closed mouth, smile, glowing sakura petals, night sky, sfw",
          "lowres, bad anatomy, bad eyes, worst quality, low quality, nsfw",
          "DPM++ 2M Karras", 768, 1024, 20, 6.5f, 0, 2 },
        { "miku_concert_1080p", "Hatsune Miku Neon Concert (1080p)", "",
          "masterpiece, best quality, absurdres, neon Tokyo skyline, starry night sky, glowing stage lights, 1girl, solo, hatsune miku, upper body, looking at viewer, aqua eyes, closed mouth, smile, sfw",
          "lowres, bad anatomy, bad eyes, worst quality, low quality, nsfw",
          "LCM", 768, 1024, 6, 1.4f, 1, 2 },
        { "miku_16x9_overpass", "Miku 16:9 Coastal Sunset (1080p)", "",
          "masterpiece, best quality, absurdres, sunset ocean, coastal bridge, golden clouds, 1girl, solo, hatsune miku, upper body, looking at viewer, aqua eyes, closed mouth, smile, white jacket, sfw",
          "lowres, bad anatomy, bad eyes, worst quality, low quality, nsfw",
          "LCM", 1024, 576, 6, 1.4f, 1, 2 },
        { "fast_draft_preview", "Fast Draft Preview (~17s)", "",
          "masterpiece, best quality, absurdres, blue concert lights, starry bokeh, 1girl, solo, hatsune miku, upper body, portrait, aqua eyes, closed mouth, smile, sfw",
          "lowres, bad eyes, worst quality, low quality, nsfw",
          "LCM", 512, 512, 4, 1.0f, 0, 2 },
        { "cyberpunk_samurai", "Cyberpunk Samurai Rain", "",
          "masterpiece, best quality, 1girl, cyberpunk samurai, neon Tokyo street, rain, katana, glowing reflections, dramatic lighting, vibrant anime illustration, fully clothed, sfw",
          "lowres, bad anatomy, bad hands, text, error, blurry, worst quality, low quality, nsfw",
          "Euler a", 768, 1024, 20, 7.0f, 0, 2 },
        { "fantasy_wizard", "Enchanted Crystal Mage", "",
          "masterpiece, best quality, 1girl, wizard hat, glowing crystal staff, enchanted twilight forest, fireflies, detailed anime illustration, fully clothed, sfw",
          "lowres, bad anatomy, bad hands, text, error, blurry, worst quality, low quality, nsfw",
          "Euler a", 768, 1024, 20, 7.0f, 0, 2 },
        { "seaside_sunset", "Seaside Golden Sunset", "",
          "masterpiece, best quality, 1girl, seaside sunset, white sundress, wind blowing hair, golden ocean light, cinematic anime key visual, fully clothed, sfw",
          "lowres, bad anatomy, bad hands, text, error, blurry, worst quality, low quality, nsfw",
          "Euler a", 768, 1024, 20, 7.0f, 0, 2 },
        { "cozy_cafe", "Cozy Rainy Cafe (1080p)", "",
          "masterpiece, best quality, 1girl, cozy anime cafe, wearing knit sweater, holding warm coffee cup, rain outside window, warm amber sunlight, soft bokeh, detailed, fully clothed, sfw",
          "lowres, bad anatomy, bad hands, text, error, blurry, worst quality, low quality, nsfw",
          "Euler", 512, 768, 8, 3.5f, 1, 2 },
        { "wallpaper_1080p", "16:9 Floating Sky Castle (1080p)", "",
          "masterpiece, best quality, amazing quality, wide panoramic anime landscape, floating fantasy castle above golden sunset clouds, cascading waterfalls, glowing sky lanterns, Makoto Shinkai anime wallpaper, no humans",
          "lowres, people, human, text, watermark, blurry, worst quality, low quality, nsfw",
          "LCM", 1024, 576, 6, 1.0f, 1, 2 },
        { "scenic_aurora_16x9", "16:9 Alpine Aurora Lake (1080p)", "",
          "masterpiece, best quality, amazing quality, wide panoramic anime wallpaper, serene alpine lake reflecting snow-capped mountains and glowing Milky Way aurora sky, wooden dock with warm lantern, Makoto Shinkai background, no humans",
          "lowres, people, human, text, watermark, blurry, worst quality, nsfw",
          "LCM", 1024, 576, 6, 1.0f, 1, 2 },
        { "scenic_canyon_16x9", "16:9 Emerald Canyon Bridges (1080p)", "",
          "masterpiece, best quality, breathtaking wide panoramic fantasy landscape painting, ancient arched stone bridges over glowing emerald canyon river, waterfalls, towering misty cliffs, golden sunrise rays, studio concept art, no humans",
          "lowres, people, human, text, watermark, blurry, worst quality, nsfw",
          "Euler a", 896, 512, 20, 7.0f, 1, 2 },
        { "scenic_ghibli_16x9", "16:9 Ghibli Coastal Train (1080p)", "",
          "masterpiece, best quality, scenic cel-shaded anime background, wide shot, coastal seaside train tracks overlooking sparkling turquoise ocean, giant fluffy summer cumulonimbus clouds, red torii gate on grassy cliff, Studio Ghibli style, no humans",
          "lowres, people, human, text, watermark, blurry, worst quality, nsfw",
          "Euler a", 896, 512, 20, 7.0f, 1, 2 },
        { "winter_lantern_fest", "Winter Lantern Night Festival", "",
          "masterpiece, best quality, amazing quality, 1girl, warm winter scarf and coat, smiling warmly under glowing floating paper lanterns at snowy Japanese night festival, bokeh street stalls, vibrant anime art, fully clothed, sfw",
          "nsfw, bad eyes, deformed hands, lowres, blurry, text, watermark, worst quality",
          "LCM", 768, 1024, 6, 1.0f, 1, 2 },
        { "sakura_school_walk", "Cherry Blossom Courtyard (1080p)", "",
          "masterpiece, best quality, 1girl, sparkling eyes, gentle smile, cozy cream cardigan over blue collared shirt, cherry blossom trees in golden hour sunlight, drifting sakura petals, crisp anime illustration, fully clothed, sfw",
          "nsfw, bad eyes, deformed hands, lowres, blurry, text, watermark, worst quality",
          "Euler", 512, 768, 8, 3.5f, 1, 2 },
        { "mecha_hangar_16x9", "16:9 Mecha Hangar Bay", "",
          "masterpiece, best quality, amazing quality, wide cinematic anime scene, massive sleek white and blue humanoid mecha standing in illuminated sci-fi hangar bay, volumetric skylight beams, sparks, detailed mechanical design, dramatic scale",
          "lowres, blurry, text, watermark, worst quality, nsfw",
          "LCM", 1024, 576, 6, 1.0f, 1, 2 },
        { "twilight_shrine", "Twilight Mountain Shrine", "",
          "masterpiece, best quality, amazing quality, scenic anime background, stone staircase lined with glowing red torii gates ascending misty pine mountain at twilight, warm stone lanterns, purple-orange dusk sky, fireflies, no humans",
          "lowres, people, human, text, watermark, blurry, worst quality, nsfw",
          "LCM", 768, 1024, 6, 1.0f, 1, 2 },
        { "fullbody_knight", "Full-Body Fantasy Paladin (Tall)", "",
          "masterpiece, best quality, amazing quality, 1girl, full body, silver and sapphire fantasy knight armor over white cape, holding ornate glowing sword, standing on sunlit stone castle terrace, clear blue sky, heroic anime key visual, fully clothed, sfw",
          "nsfw, bad eyes, deformed hands, bad anatomy, lowres, blurry, text, watermark, worst quality",
          "LCM", 832, 1216, 6, 1.0f, 1, 2 },
        { "cozy_bookshop", "Magical Antique Bookshop", "",
          "masterpiece, best quality, 2d anime illustration, cel shaded anime art, 1girl, scholar coat and scarf, warm glowing floating lanterns among towering antique wooden bookshelves, holding open illustrated grimoire, rich cozy colors, fully clothed, sfw",
          "photorealistic, realistic, 3d, doll, uncanny, nsfw, bad eyes, deformed hands, lowres, blurry, worst quality",
          "Euler a", 512, 768, 20, 7.0f, 1, 2 },
        { "starry_observatory", "16:9 Starry Observatory Rooftop", "",
          "masterpiece, best quality, amazing quality, wide shot, 1girl standing beside brass telescope on hilltop observatory balcony under sweeping Milky Way galaxy and shooting stars, gentle night breeze, serene Makoto Shinkai anime atmosphere, fully clothed, sfw",
          "nsfw, bad eyes, deformed hands, lowres, blurry, text, watermark, worst quality",
          "LCM", 1024, 576, 6, 1.0f, 1, 2 },
        { "ultra_square_kv", "Ultra Square Key Visual (1024px)", "",
          "masterpiece, best quality, amazing quality, 1girl, expressive detailed anime eyes, dynamic flowing hair and ribbons, sunlit stained glass cathedral hall with drifting light particles, ultra detailed anime movie key visual, fully clothed, sfw",
          "nsfw, bad eyes, deformed hands, lowres, blurry, text, watermark, worst quality",
          "LCM", 1024, 1024, 8, 1.0f, 0, 2 },
    };
    for (size_t i = 0; i < sizeof(DEFAULTS) / sizeof(DEFAULTS[0]); i++) {
        char fp[320];
        snprintf(fp, sizeof(fp), "%s/%s.cfg", g_presets_dir, DEFAULTS[i].id);
        if (!path_exists_local(fp)) presets_save(&DEFAULTS[i], NULL, 0);
    }
    write_all_file(marker, "2\n", 2);
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
    char marker[320];
    snprintf(marker, sizeof(marker), "%s/.seeded_v1", g_chars_dir);
    if (path_exists_local(marker)) return;

    static const CharacterItem DEFAULTS[] = {
        // Vocaloid (concise Danbooru tags — Illustrious already knows default outfits!)
        { "hatsune_miku", "Hatsune Miku", "Vocaloid",
          "1girl, solo, hatsune miku, aqua eyes, closed mouth, smile, sfw",
          "lowres, bad eyes, nsfw" },
        { "kagamine_rin", "Kagamine Rin", "Vocaloid",
          "1girl, kagamine rin, short blonde hair, big white hair ribbon, blue eyes, white sailor top, black shorts, cheerful smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "megurine_luka", "Megurine Luka", "Vocaloid",
          "1girl, megurine luka, long wavy pink hair, blue eyes, black sleeveless dress with gold trim, elegant smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        // Anime Classics & Modern Hits
        { "frieren", "Frieren", "Frieren",
          "1girl, frieren, long white twin tails, elf ears, green eyes, white and gold mage coat, striped scarf, wooden mage staff, calm gentle expression, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "fern_frieren", "Fern", "Frieren",
          "1girl, fern (sousou no frieren), long straight purple hair, purple eyes, butterfly hair ornament, white dress, long black mage coat, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "rem_rezero", "Rem", "Re:Zero",
          "1girl, rem (re:zero), short blue hair covering one eye, bright blue eyes, hair ribbon, classic black and white maid uniform, gentle smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "emilia_rezero", "Emilia", "Re:Zero",
          "1girl, emilia (re:zero), long silver hair, purple eyes, half-elf ears, white hair flower, white and violet fantasy outfit, gentle smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "saber_artoria", "Saber (Artoria)", "Fate",
          "1girl, artoria pendragon (fate), blonde hair in bun with ahoge, green eyes, blue dress with silver knight breastplate, noble expression, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "rin_tohsaka", "Rin Tohsaka", "Fate",
          "1girl, tohsaka rin, long black twin tails with black ribbons, aqua eyes, red turtleneck sweater with white cross, black pleated skirt, confident smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "asuka_langley", "Asuka Langley", "Evangelion",
          "1girl, souryuu asuka langley, long orange-red hair, blue eyes, red neural hair clips, school uniform shirt with ribbon, confident expression, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "rei_ayanami", "Rei Ayanami", "Evangelion",
          "1girl, ayanami rei, short pale blue hair, red eyes, white collared shirt and blue school uniform, serene calm expression, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "violet_evergarden", "Violet Evergarden", "Violet Evergarden",
          "1girl, violet evergarden, blonde braided hair with red ribbon, blue eyes, white dress, blue cropped jacket, emerald brooch, gentle expression, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "yor_forger", "Yor Forger", "Spy x Family",
          "1girl, yor forger, long straight black hair, red eyes, gold hairband, cozy red knit sweater, warm gentle smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "anya_forger", "Anya Forger", "Spy x Family",
          "1girl, anya forger, short pink hair, green eyes, black cone hair ornaments, black and gold Eden Academy school uniform, cheerful grin, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "makima_csm", "Makima", "Chainsaw Man",
          "1girl, makima (chainsaw man), long braided coral-red hair, yellow ringed eyes, crisp white dress shirt, black necktie, tan trench coat, calm smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "power_csm", "Power", "Chainsaw Man",
          "1girl, power (chainsaw man), long strawberry blonde hair, small red horns, cross-patterned eyes, white dress shirt, dark blue open jacket, energetic grin, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "megumin", "Megumin", "KonoSuba",
          "1girl, megumin, short brown hair, bright red eyes, large pointed wizard hat, crimson mage dress and red cape, wooden staff, confident smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "aqua_konosuba", "Aqua", "KonoSuba",
          "1girl, aqua (konosuba), long blue hair, blue eyes, loop hair ornament, blue and white fantasy adventurer outfit, cheerful expression, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "bocchi_hitori", "Hitori Gotoh (Bocchi)", "Bocchi the Rock",
          "1girl, gotoh hitori, long pink hair, aqua eyes, blue and yellow cube hair ornament, pink tracksuit jacket, holding electric guitar, shy smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "nezuko", "Nezuko Kamado", "Demon Slayer",
          "1girl, kamado nezuko, long wavy black hair with orange tips, bright pink eyes, pink hemp-leaf patterned kimono, dark brown haori coat, gentle eyes, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "zero_two", "Zero Two", "Darling in the Franxx",
          "1girl, zero two (darling in the franxx), long pink hair, small red horns, white headband, green eyes, red military jacket with white trim, playful smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "mai_sakurajima", "Mai Sakurajima", "Bunny Girl Senpai",
          "1girl, sakurajima mai, long straight dark hair, blue eyes, small bunny hairpin, school blazer over white shirt and ribbon, gentle smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "kurumi_tokisaki", "Kurumi Tokisaki", "Date A Live",
          "1girl, tokisaki kurumi, black twin tails, heterochromia red and gold clock eye, gothic red and black frilled dress, elegant smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        // VTubers (Hololive)
        { "usada_pekora", "Usada Pekora", "Hololive",
          "1girl, usada pekora, light blue braided twin tails with carrot ornaments, white bunny ears, red eyes, blue and white bunny-ear jacket, cheerful smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "gawr_gura", "Gawr Gura", "Hololive",
          "1girl, gawr gura, short silver-white hair with blue streaks, blue eyes, blue shark-themed hoodie jacket, cheerful bright smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "hoshimachi_suisei", "Hoshimachi Suisei", "Hololive",
          "1girl, hoshimachi suisei, blue hair side ponytail, star hair ornament, blue eyes, starry blue beret, plaid idol jacket and bow, sparkling smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "houshou_marine", "Houshou Marine", "Hololive",
          "1girl, houshou marine, long wavy crimson-red hair, heterochromia gold and red eyes, ornate pirate hat, red and white nautical jacket, cheerful wink, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        // Games (HoYoverse, NieR, FFVII, Nintendo)
        { "raiden_shogun", "Raiden Shogun", "Genshin Impact",
          "1girl, raiden shogun, long braided violet hair, purple eyes, floral hair ornament, ornate purple and gold kimono robe, serene regal expression, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "hu_tao", "Hu Tao", "Genshin Impact",
          "1girl, hu tao (genshin impact), long dark brown twin tails, red flower-shaped pupils, black mandarin hat with plum blossom, dark brown coat, playful smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "furina", "Furina", "Genshin Impact",
          "1girl, furina (genshin impact), short white hair with blue streaks, heterochromia blue eyes, miniature blue top hat, ornate royal blue and white vest, dramatic smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "ganyu", "Ganyu", "Genshin Impact",
          "1girl, ganyu (genshin impact), long pale blue hair, red-black curved horns, violet eyes, white and blue fantasy outfit with gold bell, calm gentle smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "firefly_hsr", "Firefly", "Honkai: Star Rail",
          "1girl, firefly (honkai: star rail), long silver hair with black headband, blue-pink eyes, cream blazer jacket, green pleated skirt, warm gentle smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "2b_nier", "2B", "NieR:Automata",
          "1girl, 2b (nier:automata), short silver bob hair, black headband, ornate gothic black dress with white embroidery and feather collar, katana, cool composed look, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "tifa_ff7", "Tifa Lockhart", "Final Fantasy VII",
          "1girl, tifa lockhart, long straight black hair, red eyes, white sporty top, black pleated skirt, suspenders, fingerless gloves, confident smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "aerith_ff7", "Aerith Gainsborough", "Final Fantasy VII",
          "1girl, aerith gainsborough, long brown braided hair with pink ribbon, emerald green eyes, pink dress, red bolero jacket, warm kind smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "cloud_ff7", "Cloud Strife", "Final Fantasy VII",
          "1boy, cloud strife, spiky blonde hair, bright blue mako eyes, dark navy sleeveless turtleneck, metal shoulder pauldron, buster sword on back, heroic expression, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "link_zelda", "Link (Champion)", "Zelda",
          "1boy, link, blonde hair, blue eyes, pointed Hylian ears, blue Champion tunic with white embroidery, leather belt, Master Sword and shield, heroic anime style, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
        { "zelda_princess", "Princess Zelda", "Zelda",
          "1girl, princess zelda, long golden blonde hair, green eyes, pointed Hylian ears, royal blue and gold fantasy tunic, serene gentle smile, fully clothed, sfw",
          "bad eyes, deformed hands, nsfw" },
    };
    for (size_t i = 0; i < sizeof(DEFAULTS) / sizeof(DEFAULTS[0]); i++) {
        char fp[320];
        snprintf(fp, sizeof(fp), "%s/%s.cfg", g_chars_dir, DEFAULTS[i].id);
        if (!path_exists_local(fp)) characters_save(&DEFAULTS[i], NULL, 0);
    }
    write_all_file(marker, "1\n", 2);
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
                     "arch=%s\nwidth=%u\nheight=%u\nsteps=%u\ncfg=%.2f\nsampler=%s\nupscale=%d\n"
                     "description=%s\nprompt=%s\nnegative=%s\n",
                     is_sd15 ? "sd15" : "sdxl", width, height, steps, (double)cfg,
                     (sampler && sampler[0]) ? sampler : "Euler a", upscale, desc, pr, ng);
    char fp[320];
    snprintf(fp, sizeof(fp), "%s/%s/preset.cfg", models_root, model_name);
    return write_all_file(fp, buf, (size_t)(n > 0 ? n : 0));
}
