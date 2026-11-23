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
#include <orbis/Http.h>
#include <orbis/Net.h>
#include <orbis/Ssl.h>
#include <orbis/Sysmodule.h>
#include <orbis/libkernel.h>
#define mkdir_p(p) mkdir(p, 0777)
#define rmdir_p(p) rmdir(p)
void glog(const char* fmt, ...);
#endif

#ifdef SDXL_CPU
static void glog(const char* fmt, ...) { (void)fmt; }
#endif

static char g_models_root[256] = "/data/garlic-diffusion/models";
static char g_repo_cfg[256] = "/data/garlic-diffusion/repo.cfg";
static char g_hf_repo[128] = "DJTomato/ps4-diffusion-models"; // Default public Hugging Face model catalog

static CatalogModel g_catalog[MODEL_DL_MAX_CATALOG];
static int g_catalog_count = 0;
static ModelDlStatus g_dl;

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
        { "WAI_Illustrious_v17_dmd2", "WAI-Illustrious v17 DMD2 (Fast SDXL)", "sdxl", 3716,
          "Latest WAI v17 anime · 6 steps CFG 1.0 (~50s Pro) · Standalone SDXL",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "waiNSFWIllustrious_v140_dmd2", "WAI-Illustrious v14 DMD2 (Fast SDXL)", "sdxl", 3716,
          "Sharp glowing anime · 6 steps CFG 1.0 (~50s Pro) · Standalone SDXL",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "Animagine_XL_4_dmd2", "Animagine XL 4.0 DMD2 (Fast SDXL)", "sdxl", 3716,
          "Clean modern anime · 6 steps CFG 1.0 (~50s Pro) · Standalone SDXL",
          "unet.psd1,clip_l.psd1,clip_g.psd1,vae.psd1,preset.cfg", "" },
        { "NoobAI_XL_1.1_dmd2", "NoobAI-XL 1.1 DMD2 (Fast SDXL)", "sdxl", 3716,
          "Crisp Danbooru anime · 6 steps CFG 1.0 (~50s Pro) · Standalone SDXL",
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
        { "DreamShaper_8", "DreamShaper 8 (2.5D / Fantasy SD 1.5)", "sd15", 1220,
          "Versatile 2.5D fantasy/illustration · 1.5 GiB VRAM · Base PS4 ready",
          "unet.psd1,clip_l.psd1,vae.psd1,preset.cfg", "" },
        { "Anything_V5", "Anything V5 (Cel-Shaded SD 1.5)", "sd15", 1220,
          "Crisp cel-shaded anime · 1.5 GiB VRAM · Auto 1080p RealESRGAN",
          "unet.psd1,clip_l.psd1,vae.psd1,preset.cfg", "" },
    };
    g_catalog_count = (int)(sizeof(BUILTIN) / sizeof(BUILTIN[0]));
    if (g_catalog_count > MODEL_DL_MAX_CATALOG) g_catalog_count = MODEL_DL_MAX_CATALOG;
    memcpy(g_catalog, BUILTIN, (size_t)g_catalog_count * sizeof(CatalogModel));
}

void model_dl_init(const char* models_root, const char* cfg_path) {
#ifndef SDXL_CPU
    if (!g_mu_ok) {
        scePthreadMutexInit(&g_dl_mu, NULL, "gdiff_dl");
        g_mu_ok = 1;
    }
#endif
    if (models_root && models_root[0]) snprintf(g_models_root, sizeof(g_models_root), "%s", models_root);
    if (cfg_path && cfg_path[0]) snprintf(g_repo_cfg, sizeof(g_repo_cfg), "%s", cfg_path);
    memset(&g_dl, 0, sizeof(g_dl));
    seed_builtin_catalog();

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

void model_dl_cancel(void) {
    dl_lock();
    if (g_dl.active) {
        g_dl.cancel_req = 1;
        snprintf(g_dl.stage, sizeof(g_dl.stage), "Cancelling download...");
    }
    dl_unlock();
}

// Parse a lightweight Hugging Face manifest.json array of model objects:
// [{"id":"...","title":"...","arch":"sdxl","size_mb":3530,"desc":"...","files":"unet.psd1,clip_l.psd1,..."}]
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
        if (*p == '\\' && p + 1 < end) { p++; out[n++] = *p++; }
        else out[n++] = *p++;
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
    char repo[128];
    model_dl_get_repo(repo, sizeof(repo));
    if (!repo[0]) return 0;
    char url[512], err[160] = {0};
    if (strncmp(repo, "http://", 7) == 0 || strncmp(repo, "https://", 8) == 0)
        snprintf(url, sizeof(url), "%s/manifest.json", repo);
    else
        snprintf(url, sizeof(url), "https://huggingface.co/%s/resolve/main/manifest.json", repo);

    char* buf = (char*)malloc(65536);
    if (!buf) return -1;
    int rc = http_fetch_url(url, NULL, buf, 65536, 0, err, sizeof(err));
    int n = 0;
    if (rc == 0) n = apply_manifest_json(buf);
    else glog("model_dl: manifest fetch failed (%s): %s", url, err);
    free(buf);
    return n;
}

int model_dl_start(const char* model_id, const char* repo_override, const char* files_override) {
    if (!gallery_valid_id(model_id)) return -1;
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
#endif
