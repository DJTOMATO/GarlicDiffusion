// Garlic Diffusion: SDXL Inference Engine (CLIP-L + OpenCLIP-bigG + SDXL UNet + Tiled VAE + Euler a)
// Builds for PS4 / PS4 Pro GCN GPU (default) or Windows/Host CPU verification (-DSDXL_CPU).

#include "sdxl.h"
#include "randomizer_pools.h"

#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#endif

#ifndef SDXL_CPU
#include <unistd.h>
#include <orbis/GnmDriver.h>
#include <orbis/libkernel.h>

int sceKernelIsNeoMode(void);
int sceKernelGetCpuTemperature(int* temp);
int sceKernelGetSocSensorTemperature(int sensor, int* temp);

#define SH_START_X 0x204
#define SH_NUM_THREAD_X 0x207
#define SH_PGM_LO 0x20C
#define SH_PGM_RSRC1 0x212
#define SH_RESOURCE_LIMITS 0x215
#define SH_STATIC_THREAD_MGMT_SE0 0x216
#define SH_STATIC_THREAD_MGMT_SE2 0x219
#define SH_TMPRING_SIZE 0x218
#define SH_USER_DATA_0 0x240
#define EV_CS_PARTIAL_FLUSH (7 | (4 << 8))
#define EV_BOTTOM_OF_PIPE_TS (0x28 | (5 << 8))
#define EV_CACHE_FLUSH_AND_INV_TS (0x14 | (5 << 8))
#define COHER_TCL1_KCACHE ((1u << 22) | (1u << 27))
#define COHER_L2_TCL1_KCACHE ((1u << 22) | (1u << 23) | (1u << 27))
#define EOP_WB_INV_L1_L2 (0x38u << 12)
#define LABEL_DONE 0x600DF00Du
#define DCB_BYTES (4u << 20)
#define KARGS_BYTES (4u << 20)
#define ALIGN_2MB (2ull << 20)
#endif

#define BUF_MAIN_BYTES (88ull << 20)
#define BUF_QKV_BYTES  (16ull << 20)
#define BUF_IM2COL_BYTES (64ull << 20)
#define BUF_SKIPS_BYTES (144ull << 20)

#ifndef SDXL_CPU
void glog(const char* fmt, ...);
#else
static void glog(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); printf("\n"); va_end(ap);
}
#endif

const char* sdxl_eco_name_ex(int eco_mode, int neo) {
    if (!neo) {
        switch (eco_mode) {
            case 0: return "Turbo (18CU)";
            case 1: return "Balanced (14CU)";
            case 2: return "Quiet (11CU)";
            default: return "Silent (7CU)";
        }
    }
    switch (eco_mode) {
        case 0: return "Turbo (36CU)";
        case 1: return "Balanced (25CU)";
        case 2: return "Quiet (20CU)";
        default: return "Silent (14CU)";
    }
}

const char* sdxl_eco_name(int eco_mode) { return sdxl_eco_name_ex(eco_mode, 1); }

// Direct libkernel temperature query (quietly skips if sandboxed)
int sdxl_read_temp_c(int* cpu_c, int* soc_c) {
    if (cpu_c) *cpu_c = 0;
    if (soc_c) *soc_c = 0;
#ifndef SDXL_CPU
    static int probed = 0, available = 0;
    if (probed && !available) return 0;
    int c = 0, s = 0;
    int rc_c = sceKernelGetCpuTemperature(&c);
    int rc_s = sceKernelGetSocSensorTemperature(0, &s);
    if (!probed) {
        probed = 1;
        available = (rc_c == 0 && c > 0) || (rc_s == 0 && s > 0);
        if (available) glog("temp sensor: cpu=%dC soc=%dC", c, s);
        else glog("temp sensor: unavailable in userland sandbox (using hardware CU power scaling)");
    }
    if (rc_c != 0) c = 0;
    if (rc_s != 0) s = 0;
    if (c <= 0 && s > 0) c = s;
    if (s <= 0 && c > 0) s = c;
    if (cpu_c) *cpu_c = c;
    if (soc_c) *soc_c = s;
    return s > 0 ? s : c;
#else
    return 0;
#endif
}

static double now_sec(const SdxlEngine* e) {
#ifndef SDXL_CPU
    return (double)sceKernelReadTsc() / (double)(e->tsc_freq ? e->tsc_freq : 1);
#else
    (void)e;
    return 0.0;
#endif
}

static void fail(SdxlEngine* e, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->error, sizeof(e->error), fmt, ap);
    va_end(ap);
#ifndef SDXL_CPU
    char line[300];
    snprintf(line, sizeof(line), "[gdiff] error: %s\n", e->error);
    sceKernelDebugOutText(0, line);
#else
    fprintf(stderr, "[gdiff] error: %s\n", e->error);
#endif
}

static uint64_t gpu_alloc(SdxlEngine* e, uint64_t size) {
    uint64_t off = (e->next + 255ull) & ~255ull;
    if (off + size > e->mem_size) return 0;
    e->next = off + size;
    return (uint64_t)(uintptr_t)(e->mem + off);
}

static const SdxlTensor* pack_find(const SdxlPack* p, const char* name) {
    int lo = 0, hi = (int)p->n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int c = strcmp(p->t[mid].name, name);
        if (c == 0) return &p->t[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return NULL;
}

static const SdxlTensor* pack_req(SdxlEngine* e, const SdxlPack* p, const char* name) {
    const SdxlTensor* t = pack_find(p, name);
    if (!t) fail(e, "missing tensor: %s", name);
    return t;
}

static uint64_t t_addr(const SdxlPack* p, const SdxlTensor* t) {
    return t ? (p->gpu_base + t->off) : 0;
}

static uint64_t t_scales(const SdxlPack* p, const SdxlTensor* t) {
    return (t && t->scales) ? (p->gpu_base + t->off + t->scales) : 0;
}

static int read_exact_fd(int fd, void* dst, size_t n) {
    uint8_t* p = (uint8_t*)dst;
    size_t done = 0;
    while (done < n) {
        size_t chunk = (n - done) > (32u << 20) ? (32u << 20) : (n - done);
        ssize_t r = read(fd, p + done, chunk);
        if (r <= 0) return -1;
        done += (size_t)r;
    }
    return 0;
}

// Fast (<1 ms) PSD1 completeness & header validator so incomplete/uploading models never hang or wipe active weights.
int sdxl_check_pack(const char* path, uint64_t* actual_out, uint64_t* expected_out) {
    if (actual_out) *actual_out = 0;
    if (expected_out) *expected_out = 0;
#ifdef _WIN32
    int fd = open(path, O_RDONLY | O_BINARY);
#else
    int fd = open(path, O_RDONLY);
#endif
    if (fd < 0) return -1;
    uint8_t hdr[24];
    if (read_exact_fd(fd, hdr, 24) != 0 || memcmp(hdr, "PSD1", 4) != 0) {
        close(fd);
        return -2;
    }
    uint32_t n = 0;
    uint64_t data_size = 0;
    memcpy(&n, hdr + 4, 4);
    memcpy(&data_size, hdr + 8, 8);
    if (n == 0 || n > 100000u || data_size == 0) {
        close(fd);
        return -2;
    }
    uint64_t header_end = 24ull + (uint64_t)n * sizeof(SdxlTensor);
    uint64_t data_start = (header_end + 255ull) & ~255ull;
    uint64_t expected_total = data_start + data_size;
#ifdef _WIN32
    int64_t fsz = _filelengthi64(fd);
#else
    off_t fsz = lseek(fd, 0, SEEK_END);
#endif
    close(fd);
    if (fsz < 0) return -1;
    uint64_t actual_data = ((uint64_t)fsz > data_start) ? ((uint64_t)fsz - data_start) : 0ull;
    if (actual_out) *actual_out = actual_data;
    if (expected_out) *expected_out = data_size;
    return ((uint64_t)fsz >= expected_total) ? 0 : -3;
}

static int probe_unet_is_sd15(const char* unet_path) {
#ifdef _WIN32
    int fd = open(unet_path, O_RDONLY | O_BINARY);
#else
    int fd = open(unet_path, O_RDONLY);
#endif
    if (fd < 0) return 0;
    uint8_t hdr[24];
    if (read_exact_fd(fd, hdr, 24) != 0 || memcmp(hdr, "PSD1", 4) != 0) { close(fd); return 0; }
    uint32_t n = 0;
    memcpy(&n, hdr + 4, 4);
    if (n == 0 || n > 100000u) { close(fd); return 0; }
    SdxlTensor* tbl = (SdxlTensor*)malloc((size_t)n * sizeof(SdxlTensor));
    if (!tbl) { close(fd); return 0; }
    int is_sd15 = 0;
    if (read_exact_fd(fd, tbl, (size_t)n * sizeof(SdxlTensor)) == 0) {
        int lo = 0, hi = (int)n - 1, found = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            int c = strcmp(tbl[mid].name, "add_embedding.linear_1.weight");
            if (c == 0) { found = 1; break; }
            if (c < 0) lo = mid + 1;
            else hi = mid - 1;
        }
        is_sd15 = found ? 0 : 1;
    }
    free(tbl);
    close(fd);
    return is_sd15;
}

static int pack_load(SdxlEngine* e, SdxlPack* p, const char* path,
                     SdxlProgressFn progress, void* user, float f0, float f1, const char* label) {
#ifdef _WIN32
    int fd = open(path, O_RDONLY | O_BINARY);
#else
    int fd = open(path, O_RDONLY);
#endif
    if (fd < 0) { fail(e, "cannot open %s", path); return -1; }
    uint8_t hdr[24];
    if (read_exact_fd(fd, hdr, 24) != 0 || memcmp(hdr, "PSD1", 4) != 0) {
        close(fd);
        fail(e, "bad PSD1 header: %s", path);
        return -1;
    }
    memcpy(&p->n, hdr + 4, 4);
    memcpy(&p->data_size, hdr + 8, 8);
    memcpy(&p->vae_scale, hdr + 16, 4);

    p->t = (SdxlTensor*)calloc(p->n, sizeof(SdxlTensor));
    if (!p->t) { close(fd); fail(e, "oom tensor table"); return -1; }
    _Static_assert(sizeof(SdxlTensor) == 136, "SdxlTensor size mismatch");
    if (read_exact_fd(fd, p->t, (size_t)p->n * sizeof(SdxlTensor)) != 0) {
        close(fd);
        fail(e, "truncated tensor table: %s", path);
        return -1;
    }
    uint64_t header_end = 24ull + (uint64_t)p->n * sizeof(SdxlTensor);
    uint64_t data_start = (header_end + 255ull) & ~255ull;
    if (lseek(fd, (off_t)data_start, SEEK_SET) < 0) {
        close(fd);
        fail(e, "seek failed: %s", path);
        return -1;
    }

    p->gpu_base = gpu_alloc(e, p->data_size);
    if (!p->gpu_base) {
        close(fd);
        fail(e, "out of GPU memory loading %s (%llu MiB)", path, (unsigned long long)(p->data_size >> 20));
        return -1;
    }

    uint64_t done = 0, chunk = 32ull << 20;
    uint8_t* dst = (uint8_t*)(uintptr_t)p->gpu_base;
    while (done < p->data_size) {
        uint64_t n = (p->data_size - done < chunk) ? (p->data_size - done) : chunk;
        ssize_t r = read(fd, dst + done, (size_t)n);
        if (r <= 0) break;
        done += (uint64_t)r;
        if (progress) progress(user, f0 + (f1 - f0) * ((float)done / (float)p->data_size), label);
    }
    close(fd);
    if (done != p->data_size) {
        fail(e, "incomplete %s (%llu / %llu)", path, (unsigned long long)done, (unsigned long long)p->data_size);
        return -1;
    }
    return 0;
}

// =================================================================================================
// BACKEND: PS4 GCN GPU vs HOST CPU (-DSDXL_CPU)
// =================================================================================================

#ifndef SDXL_CPU

static int reserve_garlic(SdxlEngine* e) {
    uint64_t top = (sceKernelGetDirectMemorySize() >> 20) & ~31ull;
    uint64_t cap = e->neo ? 4736 : 4544;
    if (top > cap) top = cap;
    for (uint64_t m = top; m >= 1536; m -= 32) {
        uint64_t size = m << 20;
        off_t off = 0;
        void* addr = NULL;
        if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), size, ALIGN_2MB, 3, &off)) continue;
        if (sceKernelMapDirectMemory(&addr, size, 0x33, 0, off, ALIGN_2MB)) {
            sceKernelReleaseDirectMemory(off, size);
            continue;
        }
        sceKernelMunmap(addr, size);
        sceKernelReleaseDirectMemory(off, size);
        size -= 64ull << 20;  // keep 64 MiB margin for system/dialogs
        if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), size, ALIGN_2MB, 3, &off)) return -1;
        if (sceKernelMapDirectMemory(&addr, size, 0x33, 0, off, ALIGN_2MB)) return -1;
        e->mem = (uint8_t*)addr;
        e->mem_phys_off = (int64_t)off;
        e->mem_size = size;
        return 0;
    }
    return -1;
}

static void put(SdxlEngine* e, uint32_t w) {
    if (e->ndw < e->dcb_cap) e->dcb[e->ndw] = w;
    e->ndw++;
}
static void pkt3(SdxlEngine* e, uint32_t op, int n, int compute) {
    put(e, (3u << 30) | ((uint32_t)(n - 1) << 16) | (op << 8) | (compute ? 2 : 0));
}
static void set_sh(SdxlEngine* e, uint32_t reg, int n, const uint32_t* v) {
    pkt3(e, 0x76, n + 1, 0);
    put(e, reg);
    for (int i = 0; i < n; i++) put(e, v[i]);
}
static void set_sh1(SdxlEngine* e, uint32_t reg, uint32_t a) { set_sh(e, reg, 1, &a); }
static void acquire(SdxlEngine* e, uint32_t cntl) {
    pkt3(e, 0x58, 6, 0);
    put(e, cntl); put(e, 0xFFFFFFFF); put(e, 0); put(e, 0); put(e, 0); put(e, 0x0A);
}
static void eop(SdxlEngine* e, uint32_t event, uint64_t addr, uint32_t data_sel, uint32_t value) {
    pkt3(e, 0x47, 5, 0);
    put(e, event);
    put(e, (uint32_t)addr);
    put(e, (uint32_t)((addr >> 32) & 0xFFFF) | (data_sel << 29));
    put(e, value);
    put(e, 0);
}

static void gpu_begin(SdxlEngine* e) {
    e->ndw = 0;
    e->kargs_used = 0;
    e->dispatch_count = 0;
    for (int i = 0; i < 256; i++) put(e, e->init_state[i]);
    // Symmetric checkerboard CU masks replicated across SH0 [15:0] and SH1 [31:16] on every SE:
    // spreads thermal load evenly across the entire APU die (no hot-side concentration, safe under CU harvesting).
    uint32_t mask = 0xFFFFFFFFu;          // Mode 0 Turbo:      100% CUs (36 Pro / 18 Base)
    if (e->eco_mode == 1) mask = 0x02DB02DBu;      // Mode 1 Balanced:   ~70% CUs (~26 Pro / ~13 Base)
    else if (e->eco_mode == 2) mask = 0x01550155u; // Mode 2 Quiet:      ~50% CUs (~18 Pro / ~9 Base)
    else if (e->eco_mode >= 3) mask = 0x02490249u; // Mode 3 Ultra-Cool: ~38% CUs (~14 Pro / ~7 Base)
    uint32_t se_pair[2] = {mask, mask};
    uint32_t zero3[3] = {0, 0, 0};
    set_sh1(e, SH_RESOURCE_LIMITS, 0);
    set_sh(e, SH_STATIC_THREAD_MGMT_SE0, 2, se_pair);
    if (e->neo) set_sh(e, SH_STATIC_THREAD_MGMT_SE2, 2, se_pair);
    set_sh1(e, SH_TMPRING_SIZE, 0);
    set_sh(e, SH_START_X, 3, zero3);
    acquire(e, COHER_L2_TCL1_KCACHE);
    e->stamps[0] = 0;
    e->stamps[1] = 0;
    eop(e, EV_BOTTOM_OF_PIPE_TS, (uint64_t)(uintptr_t)e->stamps, 3, 0);
}

static int gpu_flush(SdxlEngine* e) {
    if (e->dispatch_count == 0) return 0;
    eop(e, EV_CACHE_FLUSH_AND_INV_TS, (uint64_t)(uintptr_t)(e->stamps + 1), 3, 0);
    eop(e, EV_CACHE_FLUSH_AND_INV_TS | EOP_WB_INV_L1_L2, (uint64_t)(uintptr_t)e->label, 1, LABEL_DONE);
    while (e->ndw % 8) put(e, 0x80000000);
    if (e->ndw > e->dcb_cap || e->kargs_used > e->kargs_cap) {
        fail(e, "DCB/kargs overflow (%u dw, %u bytes)", e->ndw, e->kargs_used);
        return -1;
    }
    *e->label = 0;
    __asm__ volatile("sfence" ::: "memory");
    void* dcbs[1] = {e->dcb};
    uint32_t sizes[1] = {e->ndw * 4};
    int r = sceGnmSubmitCommandBuffers(1, dcbs, sizes, NULL, NULL);
    if (r) { fail(e, "sceGnmSubmitCommandBuffers -> 0x%08x", (unsigned)r); return -1; }
    sceGnmSubmitDone();
    uint64_t t0 = sceKernelReadTsc();
    int poll_n = 0;
    while (*e->label != LABEL_DONE) {
        uint64_t dt = sceKernelReadTsc() - t0;
        if (dt > e->tsc_freq * 45) { fail(e, "GPU timeout (>45s)"); return -1; }
        // Sleep immediately instead of burning CPU at 100% spin while GPU computes!
        sceKernelUsleep(1500);
        if (e->cancel_fn && (++poll_n & 7) == 0) e->cancel_fn(e->cancel_user);
    }
    __asm__ volatile("mfence" ::: "memory");
    __asm__ volatile("lfence" ::: "memory");
    if (e->stamps[1] > e->stamps[0]) {
        uint64_t dt_ticks = e->stamps[1] - e->stamps[0];
        if (dt_ticks < 4500000000ull) e->gpu_acc_sec += (double)dt_ticks / 100.0e6;
    }
    gpu_begin(e);
    return 0;
}

static uint64_t args_v(SdxlEngine* e, const char* spec, ...) {
    uint32_t off = (e->kargs_used + 63u) & ~63u;
    uint8_t* p = e->kargs + off;
    uint32_t pos = 0;
    va_list ap;
    va_start(ap, spec);
    for (const char* s = spec; *s; s++) {
        if (*s == 'p') {
            pos = (pos + 7u) & ~7u;
            uint64_t v = va_arg(ap, uint64_t);
            memcpy(p + pos, &v, 8);
            pos += 8;
        } else if (*s == 'u') {
            uint32_t v = va_arg(ap, uint32_t);
            memcpy(p + pos, &v, 4);
            pos += 4;
        } else {
            float v = (float)va_arg(ap, double);
            memcpy(p + pos, &v, 4);
            pos += 4;
        }
    }
    va_end(ap);
    e->kargs_used = off + pos;
    return (uint64_t)(uintptr_t)p;
}

static void dispatch(SdxlEngine* e, const SdxlKernel* k, uint64_t kargs, uint32_t groups, uint32_t threads) {
    if (groups == 0) return;
    uint32_t pgm[2] = {(uint32_t)(k->addr >> 8), (uint32_t)((k->addr >> 40) & 0xFF)};
    uint32_t rsrc[2] = {k->rsrc1, k->rsrc2};
    uint32_t nthr[3] = {threads, 1, 1};
    uint32_t user[6] = {0, 0, 0, 0, (uint32_t)kargs, (uint32_t)(kargs >> 32)};
    set_sh(e, SH_PGM_LO, 2, pgm);
    set_sh(e, SH_PGM_RSRC1, 2, rsrc);
    set_sh(e, SH_NUM_THREAD_X, 3, nthr);
    set_sh(e, SH_USER_DATA_0, (int)k->user_sgprs, user);
    pkt3(e, 0x15, 4, 1);
    put(e, groups); put(e, 1); put(e, 1); put(e, 1);
    pkt3(e, 0x46, 1, 0);
    put(e, EV_CS_PARTIAL_FLUSH);
    acquire(e, COHER_TCL1_KCACHE);
    e->dispatch_count++;
    // Auto-flush every 512 dispatches (DCB holds ~12k dispatches, KARGS ~32k)
    if (e->dispatch_count >= 512) gpu_flush(e);
}

static const SdxlKernel* find_k(SdxlEngine* e, const char* name) {
    for (int i = 0; i < e->nk; i++)
        if (!strcmp(e->k[i].name, name)) return &e->k[i];
    fail(e, "missing kernel: %s", name);
    return NULL;
}

static int load_kernels(SdxlEngine* e, const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fail(e, "cannot open %s", path); return -1; }
    off_t sz = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    uint8_t* d = (sz > 12) ? (uint8_t*)malloc((size_t)sz) : NULL;
    if (!d || read_exact_fd(fd, d, (size_t)sz) != 0) { free(d); close(fd); fail(e, "read %s failed", path); return -1; }
    close(fd);
    if (memcmp(d, "PKRN", 4) != 0) { free(d); fail(e, "bad PKRN magic"); return -1; }

    uint32_t n, text_size;
    memcpy(&n, d + 4, 4);
    memcpy(&text_size, d + 8, 4);
    uint64_t text = gpu_alloc(e, text_size);
    memcpy((void*)(uintptr_t)text, d + 12 + n * 52, text_size);
    e->nk = 0;
    for (uint32_t i = 0; i < n && i < 28; i++) {
        const uint8_t* r = d + 12 + i * 52;
        SdxlKernel* k = &e->k[e->nk++];
        memcpy(k->name, r, 32);
        k->name[31] = 0;
        uint32_t entry;
        memcpy(&entry, r + 32, 4);
        memcpy(&k->rsrc1, r + 36, 4);
        memcpy(&k->rsrc2, r + 40, 4);
        memcpy(&k->user_sgprs, r + 44, 4);
        memcpy(&k->kernarg_sgpr, r + 48, 4);
        k->addr = text + entry;
    }
    free(d);

    e->k_add          = find_k(e, "add_f32");
    e->k_copy         = find_k(e, "copy_f32");
    e->k_scale_add    = find_k(e, "scale_add_f32");
    e->k_gemm_q8_64   = find_k(e, "gemm_q8_64x64_t");
    e->k_gemm_q8_128x64 = find_k(e, "gemm_q8_128x64_t");
    e->k_gemm_q8_128  = find_k(e, "gemm_q8_128x128_t");
    e->k_gemm_f32     = find_k(e, "gemm_f32");
    e->k_im2col       = find_k(e, "im2col_3x3");
    e->k_gn        = find_k(e, "groupnorm_silu");
    e->k_ln        = find_k(e, "layernorm");
    e->k_add_temb  = find_k(e, "add_temb");
    e->k_concat    = find_k(e, "concat_c");
    e->k_up2x      = find_k(e, "upsample2x");
    e->k_geglu     = find_k(e, "geglu");
    e->k_act       = find_k(e, "act_inplace");
    e->k_clip_emb  = find_k(e, "clip_embed");
    e->k_attn64    = find_k(e, "attn_d64");
    e->k_attn40    = find_k(e, "attn_d40");
    e->k_attn80    = find_k(e, "attn_d80");
    e->k_attn160   = find_k(e, "attn_d160");
    e->k_attn512   = find_k(e, "attn_d512");
    e->k_prelu     = find_k(e, "prelu_inplace");
    e->k_lora_q8   = find_k(e, "lora_fuse_q8");
    e->k_lora_f32  = find_k(e, "lora_fuse_f32");
    return e->error[0] ? -1 : 0;
}

// GPU op wrappers
static void op_add(SdxlEngine* e, uint64_t a, uint64_t b, uint64_t out, uint32_t n) {
    dispatch(e, e->k_add, args_v(e, "pppu", a, b, out, n), (n + 255u) / 256u, 256);
}
static void op_copy(SdxlEngine* e, uint64_t src, uint64_t dst, uint32_t n) {
    dispatch(e, e->k_copy, args_v(e, "ppu", src, dst, n), (n + 255u) / 256u, 256);
}
static void op_scale_add(SdxlEngine* e, uint64_t a, uint64_t b, uint64_t out, float sa, float sb, uint32_t n) {
    dispatch(e, e->k_scale_add, args_v(e, "pppffu", a, b, out, (double)sa, (double)sb, n), (n + 255u) / 256u, 256);
}
static void op_gemm_q8_var(SdxlEngine* e, int var, uint64_t A, uint64_t Wq, uint64_t Wsc, uint64_t bias, uint64_t C,
                           uint32_t M, uint32_t N, uint32_t K, uint32_t add_res) {
    e->flops_acc += 2.0 * (double)M * (double)N * (double)K;
    if (var == 2) {
        uint32_t gx = (M + 127u) >> 7, gy = (N + 127u) >> 7;
        dispatch(e, e->k_gemm_q8_128, args_v(e, "pppppuuuu", A, Wq, Wsc, bias, C, M, N, K, add_res), gx * gy, 256);
    } else if (var == 1) {
        uint32_t gx = (M + 127u) >> 7, gy = (N + 63u) >> 6;
        dispatch(e, e->k_gemm_q8_128x64, args_v(e, "pppppuuuu", A, Wq, Wsc, bias, C, M, N, K, add_res), gx * gy, 256);
    } else {
        uint32_t gx = (M + 63u) >> 6, gy = (N + 63u) >> 6;
        dispatch(e, e->k_gemm_q8_64, args_v(e, "pppppuuuu", A, Wq, Wsc, bias, C, M, N, K, add_res), gx * gy, 256);
    }
}
static void op_gemm_q8(SdxlEngine* e, uint64_t A, uint64_t Wq, uint64_t Wsc, uint64_t bias, uint64_t C,
                       uint32_t M, uint32_t N, uint32_t K, uint32_t add_res) {
    int var = (M <= 64u) ? 0 : e->best_gemm;
    op_gemm_q8_var(e, var, A, Wq, Wsc, bias, C, M, N, K, add_res);
}
static void op_gemm_f32(SdxlEngine* e, uint64_t A, uint64_t W, uint64_t bias, uint64_t C,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t add_res) {
    uint32_t gx = (M + 63u) >> 6, gy = (N + 63u) >> 6;
    e->flops_acc += 2.0 * (double)M * (double)N * (double)K;
    dispatch(e, e->k_gemm_f32, args_v(e, "ppppuuuu", A, W, bias, C, M, N, K, add_res), gx * gy, 256);
}
static void op_im2col(SdxlEngine* e, uint64_t in, uint64_t out,
                      uint32_t H, uint32_t W, uint32_t C, uint32_t W_out, uint32_t stride, uint32_t row0, uint32_t n_rows) {
    dispatch(e, e->k_im2col, args_v(e, "ppuuuuuu", in, out, H, W, C, W_out, stride, row0), n_rows, 256);
}
static void op_gn(SdxlEngine* e, uint64_t x, uint64_t w, uint64_t b, uint64_t out,
                  uint32_t HW, uint32_t C, uint32_t G, float eps, uint32_t do_silu) {
    dispatch(e, e->k_gn, args_v(e, "ppppuuufu", x, w, b, out, HW, C, G, (double)eps, do_silu), G, 256);
}
static void op_ln(SdxlEngine* e, uint64_t x, uint64_t w, uint64_t b, uint64_t out, uint32_t M, uint32_t C, float eps) {
    dispatch(e, e->k_ln, args_v(e, "ppppuf", x, w, b, out, C, (double)eps), M, 256);
}
static void op_add_temb(SdxlEngine* e, uint64_t h, uint64_t temb, uint32_t HW, uint32_t C) {
    dispatch(e, e->k_add_temb, args_v(e, "ppuu", h, temb, HW, C), (HW * C + 255u) / 256u, 256);
}
static void op_concat(SdxlEngine* e, uint64_t a, uint64_t b, uint64_t out, uint32_t HW, uint32_t Ca, uint32_t Cb) {
    dispatch(e, e->k_concat, args_v(e, "pppuuu", a, b, out, HW, Ca, Cb), (HW * (Ca + Cb) + 255u) / 256u, 256);
}
static void op_up2x(SdxlEngine* e, uint64_t in, uint64_t out, uint32_t H, uint32_t W, uint32_t C) {
    uint64_t total = 4ull * H * W * C;
    dispatch(e, e->k_up2x, args_v(e, "ppuuu", in, out, H, W, C), (uint32_t)((total + 255ull) / 256ull), 256);
}
static void op_geglu(SdxlEngine* e, uint64_t in, uint64_t out, uint32_t M, uint32_t D) {
    uint64_t total = (uint64_t)M * D;
    dispatch(e, e->k_geglu, args_v(e, "ppuu", in, out, M, D), (uint32_t)((total + 255ull) / 256ull), 256);
}
static void op_act(SdxlEngine* e, uint64_t x, uint32_t n, uint32_t mode) {
    dispatch(e, e->k_act, args_v(e, "puu", x, n, mode), (n + 255u) / 256u, 256);
}
static void op_clip_emb(SdxlEngine* e, uint64_t tok_emb, uint64_t pos_emb, uint64_t tokens, uint64_t out,
                        uint32_t seq_len, uint32_t C) {
    dispatch(e, e->k_clip_emb, args_v(e, "ppppuu", tok_emb, pos_emb, tokens, out, seq_len, C),
             (seq_len * C + 255u) / 256u, 256);
}
static void op_attn64(SdxlEngine* e, uint64_t Q, uint64_t K, uint64_t V, uint64_t Out,
                      uint32_t M, uint32_t S, uint32_t H, float scale, uint32_t causal) {
    uint32_t q_blocks = (M + 15u) >> 4;
    e->flops_acc += 4.0 * (double)H * (double)M * (double)S * 64.0;
    dispatch(e, e->k_attn64, args_v(e, "ppppuuufu", Q, K, V, Out, M, S, H, (double)scale, causal), H * q_blocks, 256);
}
static void op_attn_hd(SdxlEngine* e, uint64_t Q, uint64_t K, uint64_t V, uint64_t Out,
                       uint32_t M, uint32_t S, uint32_t H, uint32_t D, float scale) {
    e->flops_acc += 4.0 * (double)H * (double)M * (double)S * (double)D;
    if (D == 40u) {
        uint32_t q_blocks = (M + 31u) >> 5;
        dispatch(e, e->k_attn40, args_v(e, "ppppuuuf", Q, K, V, Out, M, S, H, (double)scale), H * q_blocks, 256);
    } else if (D == 80u) {
        uint32_t q_blocks = (M + 15u) >> 4;
        dispatch(e, e->k_attn80, args_v(e, "ppppuuuf", Q, K, V, Out, M, S, H, (double)scale), H * q_blocks, 256);
    } else {
        uint32_t q_blocks = (M + 15u) >> 4;
        dispatch(e, e->k_attn160, args_v(e, "ppppuuuf", Q, K, V, Out, M, S, H, (double)scale), H * q_blocks, 256);
    }
}
static void op_attn512(SdxlEngine* e, uint64_t Q, uint64_t K, uint64_t V, uint64_t Out, uint32_t M, float scale) {
    dispatch(e, e->k_attn512, args_v(e, "ppppuf", Q, K, V, Out, M, (double)scale), M, 256);
}
static void op_prelu(SdxlEngine* e, uint64_t x, uint64_t w, uint32_t HW, uint32_t C) {
    uint64_t total = (uint64_t)HW * C;
    dispatch(e, e->k_prelu, args_v(e, "ppuu", x, w, HW, C), (uint32_t)((total + 255ull) / 256ull), 256);
}
static void op_lora_fuse_q8(SdxlEngine* e, uint64_t Wq, uint64_t Wsc, uint64_t Up, uint64_t Down,
                            uint32_t rows, uint32_t cols, uint32_t rank, float mult) {
    uint64_t total_blocks = (uint64_t)rows * (cols >> 5);
    uint32_t groups = (uint32_t)((total_blocks + 7ull) >> 3);
    dispatch(e, e->k_lora_q8, args_v(e, "ppppuuuf", Wq, Wsc, Up, Down, rows, cols, rank, (double)mult), groups, 256);
}
static void op_lora_fuse_f32(SdxlEngine* e, uint64_t W, uint64_t Up, uint64_t Down,
                             uint32_t rows, uint32_t cols, uint32_t rank, float mult) {
    uint64_t total = (uint64_t)rows * cols;
    uint32_t groups = (uint32_t)((total + 255ull) >> 8);
    dispatch(e, e->k_lora_f32, args_v(e, "pppuuuf", W, Up, Down, rows, cols, rank, (double)mult), groups, 256);
}

static float host_fp16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t ex = (h >> 10) & 31u;
    uint32_t m = h & 0x3FFu;
    if (ex == 0) {
        if (m == 0) { float z = 0.0f; return s ? -z : z; }
        float v = ldexpf((float)m, -24);
        return s ? -v : v;
    }
    uint32_t f = (ex == 31) ? (s | 0x7F800000u | (m << 13)) : (s | ((ex + 112u) << 23) | (m << 13));
    float r; memcpy(&r, &f, 4); return r;
}

#else // ---- HOST CPU REFERENCE BACKEND (Windows testing) ------------------------------------------

static float fp16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t ex = (h >> 10) & 31u;
    uint32_t m = h & 0x3FFu;
    uint32_t f;
    if (ex == 0) {
        if (m == 0) f = s;
        else {
            float v = ldexpf((float)m, -24);
            return s ? -v : v;
        }
    } else if (ex == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((ex + 112u) << 23) | (m << 13);
    }
    float r;
    memcpy(&r, &f, 4);
    return r;
}

static float host_fp16_to_f32(uint16_t h) { return fp16_to_f32(h); }

static uint16_t f32_to_fp16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000u) >> 13));
}

static inline float cpu_silu(float x) { return x / (1.0f + expf(-x)); }
static inline float cpu_quick_gelu(float x) { return x / (1.0f + expf(-1.702f * x)); }
static inline float cpu_gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }

static int gpu_flush(SdxlEngine* e) { (void)e; return 0; }
static void gpu_begin(SdxlEngine* e) { (void)e; }

static void op_add(SdxlEngine* e, uint64_t a, uint64_t b, uint64_t out, uint32_t n) {
    (void)e;
    const float *pa = (const float*)(uintptr_t)a, *pb = (const float*)(uintptr_t)b;
    float* po = (float*)(uintptr_t)out;
    for (uint32_t i = 0; i < n; i++) po[i] = pa[i] + pb[i];
}
static void op_copy(SdxlEngine* e, uint64_t src, uint64_t dst, uint32_t n) {
    (void)e;
    memcpy((void*)(uintptr_t)dst, (const void*)(uintptr_t)src, (size_t)n * 4);
}
static void op_scale_add(SdxlEngine* e, uint64_t a, uint64_t b, uint64_t out, float sa, float sb, uint32_t n) {
    (void)e;
    const float *pa = (const float*)(uintptr_t)a, *pb = (const float*)(uintptr_t)b;
    float* po = (float*)(uintptr_t)out;
    for (uint32_t i = 0; i < n; i++) po[i] = sa * pa[i] + sb * pb[i];
}
static void op_gemm_q8(SdxlEngine* e, uint64_t A, uint64_t Wq, uint64_t Wsc, uint64_t bias, uint64_t C,
                       uint32_t M, uint32_t N, uint32_t K, uint32_t add_res) {
    (void)e;
    const float* pA = (const float*)(uintptr_t)A;
    const int8_t* pW = (const int8_t*)(uintptr_t)Wq;
    const uint16_t* pS = (const uint16_t*)(uintptr_t)Wsc;
    const float* pB = (const float*)(uintptr_t)bias;
    float* pC = (float*)(uintptr_t)C;
    uint32_t nb = K >> 5;
    for (uint32_t m = 0; m < M; m++) {
        const float* ar = pA + (size_t)m * K;
        float* cr = pC + (size_t)m * N;
        for (uint32_t n = 0; n < N; n++) {
            const int8_t* wr = pW + (size_t)n * K;
            const uint16_t* sr = pS + (size_t)n * nb;
            float acc = 0.0f;
            for (uint32_t kb = 0; kb < nb; kb++) {
                float sc = fp16_to_f32(sr[kb]);
                float sub = 0.0f;
                const float* a32 = ar + (kb << 5);
                const int8_t* w32 = wr + (kb << 5);
                for (int j = 0; j < 32; j++) sub += a32[j] * (float)w32[j];
                acc += sub * sc;
            }
            if (pB) acc += pB[n];
            cr[n] = add_res ? (cr[n] + acc) : acc;
        }
    }
}
static void op_gemm_f32(SdxlEngine* e, uint64_t A, uint64_t W, uint64_t bias, uint64_t C,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t add_res) {
    (void)e;
    const float* pA = (const float*)(uintptr_t)A;
    const float* pW = (const float*)(uintptr_t)W;
    const float* pB = (const float*)(uintptr_t)bias;
    float* pC = (float*)(uintptr_t)C;
    for (uint32_t m = 0; m < M; m++) {
        const float* ar = pA + (size_t)m * K;
        float* cr = pC + (size_t)m * N;
        for (uint32_t n = 0; n < N; n++) {
            const float* wr = pW + (size_t)n * K;
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) acc += ar[k] * wr[k];
            if (pB) acc += pB[n];
            cr[n] = add_res ? (cr[n] + acc) : acc;
        }
    }
}
static void op_lora_fuse_q8(SdxlEngine* e, uint64_t Wq, uint64_t Wsc, uint64_t Up, uint64_t Down,
                            uint32_t rows, uint32_t cols, uint32_t rank, float mult) {
    (void)e;
    int8_t* pW = (int8_t*)(uintptr_t)Wq;
    uint16_t* pS = (uint16_t*)(uintptr_t)Wsc;
    const float* pU = (const float*)(uintptr_t)Up;
    const float* pD = (const float*)(uintptr_t)Down;
    uint32_t nb = cols >> 5;
    for (uint32_t row = 0; row < rows; row++) {
        const float* ur = pU + (size_t)row * rank;
        for (uint32_t kb = 0; kb < nb; kb++) {
            uint64_t bidx = (uint64_t)row * nb + kb;
            float sc_old = fp16_to_f32(pS[bidx]);
            float vals[32];
            float mx = 0.0f;
            for (uint32_t lane = 0; lane < 32; lane++) {
                uint32_t col = (kb << 5) + lane;
                float w_old = (float)pW[(size_t)row * cols + col] * sc_old;
                float delta = 0.0f;
                for (uint32_t r = 0; r < rank; r++) delta += ur[r] * pD[(size_t)r * cols + col];
                float wn = w_old + mult * delta;
                vals[lane] = wn;
                float aw = fabsf(wn);
                if (aw > mx) mx = aw;
            }
            uint16_t sc_h = f32_to_fp16(mx / 127.0f);
            pS[bidx] = sc_h;
            float sc_new = fp16_to_f32(sc_h);
            float inv = sc_new > 0.0f ? (1.0f / sc_new) : 0.0f;
            for (uint32_t lane = 0; lane < 32; lane++) {
                float qf = rintf(vals[lane] * inv);
                if (qf < -127.0f) qf = -127.0f;
                if (qf >  127.0f) qf =  127.0f;
                pW[(size_t)row * cols + (kb << 5) + lane] = (int8_t)((int)qf);
            }
        }
    }
}
static void op_lora_fuse_f32(SdxlEngine* e, uint64_t W, uint64_t Up, uint64_t Down,
                             uint32_t rows, uint32_t cols, uint32_t rank, float mult) {
    (void)e;
    float* pW = (float*)(uintptr_t)W;
    const float* pU = (const float*)(uintptr_t)Up;
    const float* pD = (const float*)(uintptr_t)Down;
    for (uint32_t row = 0; row < rows; row++) {
        const float* ur = pU + (size_t)row * rank;
        for (uint32_t col = 0; col < cols; col++) {
            float delta = 0.0f;
            for (uint32_t r = 0; r < rank; r++) delta += ur[r] * pD[(size_t)r * cols + col];
            pW[(size_t)row * cols + col] += mult * delta;
        }
    }
}
static void op_im2col(SdxlEngine* e, uint64_t in, uint64_t out,
                      uint32_t H, uint32_t W, uint32_t C, uint32_t W_out, uint32_t stride, uint32_t row0, uint32_t n_rows) {
    (void)e;
    const float* pi = (const float*)(uintptr_t)in;
    float* po = (float*)(uintptr_t)out;
    for (uint32_t r = 0; r < n_rows; r++) {
        uint32_t idx = row0 + r;
        uint32_t oy = idx / W_out, ox = idx % W_out;
        float* dst_row = po + (size_t)r * 9 * C;
        for (int ky = 0; ky < 3; ky++) {
            int iy = (int)(oy * stride) + ky - 1;
            for (int kx = 0; kx < 3; kx++) {
                int ix = (int)(ox * stride) + kx - 1;
                float* d = dst_row + (size_t)(ky * 3 + kx) * C;
                if (iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W)
                    memcpy(d, pi + ((size_t)iy * W + (size_t)ix) * C, C * 4);
                else
                    memset(d, 0, C * 4);
            }
        }
    }
}
static void op_gn(SdxlEngine* e, uint64_t x, uint64_t w, uint64_t b, uint64_t out,
                  uint32_t HW, uint32_t C, uint32_t G, float eps, uint32_t do_silu) {
    (void)e;
    const float *px = (const float*)(uintptr_t)x, *pw = (const float*)(uintptr_t)w, *pb = (const float*)(uintptr_t)b;
    float* po = (float*)(uintptr_t)out;
    uint32_t cg = C / G;
    for (uint32_t g = 0; g < G; g++) {
        uint32_t c0 = g * cg;
        double sum = 0.0;
        for (uint32_t hw = 0; hw < HW; hw++)
            for (uint32_t dc = 0; dc < cg; dc++) sum += px[(size_t)hw * C + c0 + dc];
        double mean = sum / (double)(HW * cg);
        double vs = 0.0;
        for (uint32_t hw = 0; hw < HW; hw++)
            for (uint32_t dc = 0; dc < cg; dc++) {
                double d = px[(size_t)hw * C + c0 + dc] - mean;
                vs += d * d;
            }
        float inv = 1.0f / sqrtf((float)(vs / (double)(HW * cg)) + eps);
        for (uint32_t hw = 0; hw < HW; hw++)
            for (uint32_t dc = 0; dc < cg; dc++) {
                uint32_t c = c0 + dc;
                float v = (px[(size_t)hw * C + c] - (float)mean) * inv * pw[c] + pb[c];
                if (do_silu) v = cpu_silu(v);
                po[(size_t)hw * C + c] = v;
            }
    }
}
static void op_ln(SdxlEngine* e, uint64_t x, uint64_t w, uint64_t b, uint64_t out, uint32_t M, uint32_t C, float eps) {
    (void)e;
    const float *px = (const float*)(uintptr_t)x, *pw = (const float*)(uintptr_t)w, *pb = (const float*)(uintptr_t)b;
    float* po = (float*)(uintptr_t)out;
    for (uint32_t m = 0; m < M; m++) {
        const float* r = px + (size_t)m * C;
        float* o = po + (size_t)m * C;
        double sum = 0.0;
        for (uint32_t c = 0; c < C; c++) sum += r[c];
        double mean = sum / (double)C;
        double vs = 0.0;
        for (uint32_t c = 0; c < C; c++) { double d = r[c] - mean; vs += d * d; }
        float inv = 1.0f / sqrtf((float)(vs / (double)C) + eps);
        for (uint32_t c = 0; c < C; c++) o[c] = (r[c] - (float)mean) * inv * pw[c] + pb[c];
    }
}
static void op_add_temb(SdxlEngine* e, uint64_t h, uint64_t temb, uint32_t HW, uint32_t C) {
    (void)e;
    float* ph = (float*)(uintptr_t)h;
    const float* pt = (const float*)(uintptr_t)temb;
    for (uint32_t hw = 0; hw < HW; hw++)
        for (uint32_t c = 0; c < C; c++) ph[(size_t)hw * C + c] += pt[c];
}
static void op_concat(SdxlEngine* e, uint64_t a, uint64_t b, uint64_t out, uint32_t HW, uint32_t Ca, uint32_t Cb) {
    (void)e;
    const float *pa = (const float*)(uintptr_t)a, *pb = (const float*)(uintptr_t)b;
    float* po = (float*)(uintptr_t)out;
    uint32_t Ct = Ca + Cb;
    for (uint32_t hw = 0; hw < HW; hw++) {
        memcpy(po + (size_t)hw * Ct, pa + (size_t)hw * Ca, Ca * 4);
        memcpy(po + (size_t)hw * Ct + Ca, pb + (size_t)hw * Cb, Cb * 4);
    }
}
static void op_up2x(SdxlEngine* e, uint64_t in, uint64_t out, uint32_t H, uint32_t W, uint32_t C) {
    (void)e;
    const float* pi = (const float*)(uintptr_t)in;
    float* po = (float*)(uintptr_t)out;
    uint32_t H2 = H * 2, W2 = W * 2;
    for (uint32_t oy = 0; oy < H2; oy++)
        for (uint32_t ox = 0; ox < W2; ox++)
            memcpy(po + ((size_t)oy * W2 + ox) * C, pi + ((size_t)(oy / 2) * W + ox / 2) * C, C * 4);
}
static void op_geglu(SdxlEngine* e, uint64_t in, uint64_t out, uint32_t M, uint32_t D) {
    (void)e;
    const float* pi = (const float*)(uintptr_t)in;
    float* po = (float*)(uintptr_t)out;
    for (uint32_t m = 0; m < M; m++) {
        const float* r = pi + (size_t)m * 2 * D;
        float* o = po + (size_t)m * D;
        for (uint32_t d = 0; d < D; d++) o[d] = r[d] * cpu_gelu(r[D + d]);
    }
}
static void op_act(SdxlEngine* e, uint64_t x, uint32_t n, uint32_t mode) {
    (void)e;
    float* px = (float*)(uintptr_t)x;
    for (uint32_t i = 0; i < n; i++)
        px[i] = (mode == 0) ? cpu_silu(px[i]) : (mode == 1 ? cpu_quick_gelu(px[i]) : cpu_gelu(px[i]));
}
static void op_clip_emb(SdxlEngine* e, uint64_t tok_emb, uint64_t pos_emb, uint64_t tokens, uint64_t out,
                        uint32_t seq_len, uint32_t C) {
    (void)e;
    const uint16_t* te = (const uint16_t*)(uintptr_t)tok_emb;
    const uint16_t* pe = (const uint16_t*)(uintptr_t)pos_emb;
    const uint32_t* tk = (const uint32_t*)(uintptr_t)tokens;
    float* po = (float*)(uintptr_t)out;
    for (uint32_t s = 0; s < seq_len; s++)
        for (uint32_t c = 0; c < C; c++)
            po[(size_t)s * C + c] = fp16_to_f32(te[(size_t)tk[s] * C + c]) + fp16_to_f32(pe[(size_t)s * C + c]);
}
static void op_attn64(SdxlEngine* e, uint64_t Q, uint64_t K, uint64_t V, uint64_t Out,
                      uint32_t M, uint32_t S, uint32_t H, float scale, uint32_t causal) {
    (void)e;
    const float *pQ = (const float*)(uintptr_t)Q, *pK = (const float*)(uintptr_t)K, *pV = (const float*)(uintptr_t)V;
    float* pO = (float*)(uintptr_t)Out;
    uint32_t stride = H * 64;
    for (uint32_t h = 0; h < H; h++) {
        uint32_t hoff = h * 64;
        for (uint32_t m = 0; m < M; m++) {
            const float* q = pQ + (size_t)m * stride + hoff;
            float* o = pO + (size_t)m * stride + hoff;
            float mx = -1e30f, ls = 0.0f;
            memset(o, 0, 64 * 4);
            uint32_t slim = causal ? (m + 1 < S ? m + 1 : S) : S;
            for (uint32_t s = 0; s < slim; s++) {
                const float* k = pK + (size_t)s * stride + hoff;
                const float* v = pV + (size_t)s * stride + hoff;
                float dot = 0.0f;
                for (int d = 0; d < 64; d++) dot += q[d] * k[d];
                float sc = dot * scale;
                float nm = sc > mx ? sc : mx;
                float a = expf(mx - nm), b = expf(sc - nm);
                for (int d = 0; d < 64; d++) o[d] = o[d] * a + v[d] * b;
                ls = ls * a + b;
                mx = nm;
            }
            for (int d = 0; d < 64; d++) o[d] /= ls;
        }
    }
}
static void op_attn_hd(SdxlEngine* e, uint64_t Q, uint64_t K, uint64_t V, uint64_t Out,
                       uint32_t M, uint32_t S, uint32_t H, uint32_t D, float scale) {
    (void)e;
    const float *pQ = (const float*)(uintptr_t)Q, *pK = (const float*)(uintptr_t)K, *pV = (const float*)(uintptr_t)V;
    float* pO = (float*)(uintptr_t)Out;
    uint32_t stride = H * D;
    for (uint32_t h = 0; h < H; h++) {
        uint32_t hoff = h * D;
        for (uint32_t m = 0; m < M; m++) {
            const float* q = pQ + (size_t)m * stride + hoff;
            float* o = pO + (size_t)m * stride + hoff;
            float mx = -1e30f, ls = 0.0f;
            memset(o, 0, D * 4);
            for (uint32_t s = 0; s < S; s++) {
                const float* k = pK + (size_t)s * stride + hoff;
                const float* v = pV + (size_t)s * stride + hoff;
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; d++) dot += q[d] * k[d];
                float sc = dot * scale;
                float nm = sc > mx ? sc : mx;
                float a = expf(mx - nm), b = expf(sc - nm);
                for (uint32_t d = 0; d < D; d++) o[d] = o[d] * a + v[d] * b;
                ls = ls * a + b;
                mx = nm;
            }
            for (uint32_t d = 0; d < D; d++) o[d] /= ls;
        }
    }
}
static void op_attn512(SdxlEngine* e, uint64_t Q, uint64_t K, uint64_t V, uint64_t Out, uint32_t M, float scale) {
    (void)e;
    const float *pQ = (const float*)(uintptr_t)Q, *pK = (const float*)(uintptr_t)K, *pV = (const float*)(uintptr_t)V;
    float* pO = (float*)(uintptr_t)Out;
    for (uint32_t m = 0; m < M; m++) {
        const float* q = pQ + (size_t)m * 512;
        float* o = pO + (size_t)m * 512;
        float mx = -1e30f, ls = 0.0f;
        memset(o, 0, 512 * 4);
        for (uint32_t s = 0; s < M; s++) {
            const float* k = pK + (size_t)s * 512;
            const float* v = pV + (size_t)s * 512;
            float dot = 0.0f;
            for (int d = 0; d < 512; d++) dot += q[d] * k[d];
            float sc = dot * scale;
            float nm = sc > mx ? sc : mx;
            float a = expf(mx - nm), b = expf(sc - nm);
            for (int d = 0; d < 512; d++) o[d] = o[d] * a + v[d] * b;
            ls = ls * a + b;
            mx = nm;
        }
        for (int d = 0; d < 512; d++) o[d] /= ls;
    }
}
static void op_prelu(SdxlEngine* e, uint64_t x, uint64_t w, uint32_t HW, uint32_t C) {
    (void)e;
    float* px = (float*)(uintptr_t)x;
    const float* pw = (const float*)(uintptr_t)w;
    for (uint32_t hw = 0; hw < HW; hw++)
        for (uint32_t c = 0; c < C; c++) {
            float v = px[(size_t)hw * C + c];
            if (v < 0.0f) px[(size_t)hw * C + c] = v * pw[c];
        }
}
#endif

// =================================================================================================
// HIGHER-LEVEL TENSOR OPS (shared between PS4 GPU and CPU)
// =================================================================================================

static void linear_op(SdxlEngine* e, const SdxlPack* p, const char* wname, const char* bname,
                      uint64_t A, uint64_t C, uint32_t M, uint32_t add_res) {
    const SdxlTensor* tw = pack_req(e, p, wname);
    if (!tw) return;
    const SdxlTensor* tb = bname ? pack_find(p, bname) : NULL;
    uint64_t ba = t_addr(p, tb);
    if (tw->fmt == 2)
        op_gemm_q8(e, A, t_addr(p, tw), t_scales(p, tw), ba, C, M, tw->rows, tw->cols, add_res);
    else
        op_gemm_f32(e, A, t_addr(p, tw), ba, C, M, tw->rows, tw->cols, add_res);
}

static void conv3x3_op(SdxlEngine* e, const SdxlPack* p, const char* wname, const char* bname,
                       uint64_t in, uint64_t out, uint32_t H, uint32_t W, uint32_t Cin, uint32_t Cout, uint32_t stride) {
    const SdxlTensor* tw = pack_req(e, p, wname);
    if (!tw) return;
    const SdxlTensor* tb = bname ? pack_find(p, bname) : NULL;
    uint64_t ba = t_addr(p, tb);
    uint32_t H_out = (stride == 2) ? (H / 2) : H;
    uint32_t W_out = (stride == 2) ? (W / 2) : W;
    uint32_t total_rows = H_out * W_out;
    uint32_t K = 9u * Cin;
    uint32_t max_rows = (uint32_t)(e->buf_im2col_bytes / ((uint64_t)K * 4ull));
    if (max_rows == 0) max_rows = 1;

    for (uint32_t r0 = 0; r0 < total_rows; r0 += max_rows) {
        uint32_t nr = (total_rows - r0 < max_rows) ? (total_rows - r0) : max_rows;
        op_im2col(e, in, e->buf_im2col, H, W, Cin, W_out, stride, r0, nr);
        uint64_t dst = out + (uint64_t)r0 * Cout * 4ull;
        if (tw->fmt == 2)
            op_gemm_q8(e, e->buf_im2col, t_addr(p, tw), t_scales(p, tw), ba, dst, nr, Cout, K, 0);
        else
            op_gemm_f32(e, e->buf_im2col, t_addr(p, tw), ba, dst, nr, Cout, K, 0);
    }
}

static void gn_op(SdxlEngine* e, const SdxlPack* p, const char* prefix,
                  uint64_t in, uint64_t out, uint32_t HW, uint32_t C, uint32_t G, float eps, uint32_t do_silu) {
    char wn[96], bn[96];
    snprintf(wn, sizeof(wn), "%s.weight", prefix);
    snprintf(bn, sizeof(bn), "%s.bias", prefix);
    const SdxlTensor* tw = pack_req(e, p, wn);
    const SdxlTensor* tb = pack_req(e, p, bn);
    if (!tw || !tb) return;
    op_gn(e, in, t_addr(p, tw), t_addr(p, tb), out, HW, C, G, eps, do_silu);
}

static void ln_op(SdxlEngine* e, const SdxlPack* p, const char* prefix,
                  uint64_t in, uint64_t out, uint32_t M, uint32_t C, float eps) {
    char wn[96], bn[96];
    snprintf(wn, sizeof(wn), "%s.weight", prefix);
    snprintf(bn, sizeof(bn), "%s.bias", prefix);
    const SdxlTensor* tw = pack_req(e, p, wn);
    const SdxlTensor* tb = pack_req(e, p, bn);
    if (!tw || !tb) return;
    op_ln(e, in, t_addr(p, tw), t_addr(p, tb), out, M, C, eps);
}

// =================================================================================================
// CLIP ENCODERS (CLIP-L + OpenCLIP-bigG)
// =================================================================================================

static int run_clip(SdxlEngine* e, const SdxlPack* p, const uint32_t tokens[CLIP_MAX_LEN], uint32_t eos_pos,
                    int n_layers, uint32_t C, uint32_t n_heads, uint32_t act_mode,
                    uint64_t penult_out, uint64_t pooled_out, int apply_final_ln) {
    memcpy((void*)(uintptr_t)e->tok_gpu, tokens, CLIP_MAX_LEN * 4);
    const SdxlTensor* te = pack_req(e, p, "text_model.embeddings.token_embedding.weight");
    const SdxlTensor* pe = pack_req(e, p, "text_model.embeddings.position_embedding.weight");
    if (!te || !pe) return -1;

    uint64_t x = e->buf_a, h = e->buf_b;
    op_clip_emb(e, t_addr(p, te), t_addr(p, pe), e->tok_gpu, x, CLIP_MAX_LEN, C);

    int run_layers = pooled_out ? n_layers : (n_layers - 1);
    for (int i = 0; i < run_layers; i++) {
        char nm[96], w[96], b[96];
        // Self-attention
        snprintf(nm, sizeof(nm), "text_model.encoder.layers.%d.layer_norm1", i);
        ln_op(e, p, nm, x, h, CLIP_MAX_LEN, C, 1e-5f);

        snprintf(w, sizeof(w), "text_model.encoder.layers.%d.self_attn.q_proj.weight", i);
        snprintf(b, sizeof(b), "text_model.encoder.layers.%d.self_attn.q_proj.bias", i);
        linear_op(e, p, w, b, h, e->buf_q, CLIP_MAX_LEN, 0);

        snprintf(w, sizeof(w), "text_model.encoder.layers.%d.self_attn.k_proj.weight", i);
        snprintf(b, sizeof(b), "text_model.encoder.layers.%d.self_attn.k_proj.bias", i);
        linear_op(e, p, w, b, h, e->buf_k, CLIP_MAX_LEN, 0);

        snprintf(w, sizeof(w), "text_model.encoder.layers.%d.self_attn.v_proj.weight", i);
        snprintf(b, sizeof(b), "text_model.encoder.layers.%d.self_attn.v_proj.bias", i);
        linear_op(e, p, w, b, h, e->buf_v, CLIP_MAX_LEN, 0);

        op_attn64(e, e->buf_q, e->buf_k, e->buf_v, h, CLIP_MAX_LEN, CLIP_MAX_LEN, n_heads, 0.125f, 1);

        snprintf(w, sizeof(w), "text_model.encoder.layers.%d.self_attn.out_proj.weight", i);
        snprintf(b, sizeof(b), "text_model.encoder.layers.%d.self_attn.out_proj.bias", i);
        linear_op(e, p, w, b, h, x, CLIP_MAX_LEN, 1);

        // MLP
        snprintf(nm, sizeof(nm), "text_model.encoder.layers.%d.layer_norm2", i);
        ln_op(e, p, nm, x, h, CLIP_MAX_LEN, C, 1e-5f);

        snprintf(w, sizeof(w), "text_model.encoder.layers.%d.mlp.fc1.weight", i);
        snprintf(b, sizeof(b), "text_model.encoder.layers.%d.mlp.fc1.bias", i);
        linear_op(e, p, w, b, h, e->buf_c, CLIP_MAX_LEN, 0);
        op_act(e, e->buf_c, CLIP_MAX_LEN * (4u * C), act_mode);

        snprintf(w, sizeof(w), "text_model.encoder.layers.%d.mlp.fc2.weight", i);
        snprintf(b, sizeof(b), "text_model.encoder.layers.%d.mlp.fc2.bias", i);
        linear_op(e, p, w, b, e->buf_c, x, CLIP_MAX_LEN, 1);

        if (i == n_layers - 2) {
            if (apply_final_ln) ln_op(e, p, "text_model.final_layer_norm", x, penult_out, CLIP_MAX_LEN, C, 1e-5f);
            else op_copy(e, x, penult_out, CLIP_MAX_LEN * C);
        }
    }

    if (pooled_out) {
        uint64_t eos_row = x + (uint64_t)eos_pos * C * 4ull;
        ln_op(e, p, "text_model.final_layer_norm", eos_row, h, 1, C, 1e-5f);
        linear_op(e, p, "text_projection.weight", NULL, h, pooled_out, 1, 0);
    }
    return gpu_flush(e);
}

// Apply Automatic1111 mean-preserving token weight multipliers to CLIP hidden states [CLIP_MAX_LEN, C]
static void apply_clip_token_weights(uint64_t buf_addr, uint32_t C, const float weights[CLIP_MAX_LEN]) {
    int has_w = 0;
    for (uint32_t s = 0; s < CLIP_MAX_LEN; s++) {
        if (fabsf(weights[s] - 1.0f) > 1e-4f) { has_w = 1; break; }
    }
    if (!has_w) return;
    float* z = (float*)(uintptr_t)buf_addr;
    size_t total = (size_t)CLIP_MAX_LEN * C;
    double sum0 = 0.0, sum1 = 0.0;
    for (size_t i = 0; i < total; i++) sum0 += z[i];
    for (uint32_t s = 0; s < CLIP_MAX_LEN; s++) {
        float w = weights[s];
        float* row = z + (size_t)s * C;
        for (uint32_t c = 0; c < C; c++) {
            row[c] *= w;
            sum1 += row[c];
        }
    }
    if (fabs(sum1) > 1e-6 && fabs(sum0) > 1e-6) {
        float rescale = (float)(sum0 / sum1);
        for (size_t i = 0; i < total; i++) z[i] *= rescale;
    }
}

static int encode_prompt(SdxlEngine* e, const char* text, int force_zero_empty,
                         uint64_t ctx_out, uint64_t pooled_out) {
    if (e->is_sd15) {
        // SD 1.5: CLIP-L with Clip Skip 2 (layer 10 + final_layer_norm), empty prompt encoded as BOS/EOS!
        uint32_t ids_l[CLIP_MAX_LEN];
        float w_l[CLIP_MAX_LEN];
        clip_tok_encode_weighted(&e->tok, text ? text : "", ids_l, w_l, CLIP_EOS);
        if (run_clip(e, &e->clip_l, ids_l, 0, 12, 768, 12, 1 /* quick_gelu */, ctx_out, 0, 1)) return -1;
        apply_clip_token_weights(ctx_out, 768, w_l);
        return 0;
    }
    if (force_zero_empty && (!text || !text[0])) {
        memset((void*)(uintptr_t)ctx_out, 0, CLIP_MAX_LEN * 2048u * 4u);
        memset((void*)(uintptr_t)pooled_out, 0, 1280u * 4u);
        return 0;
    }
    uint32_t ids_l[CLIP_MAX_LEN], ids_g[CLIP_MAX_LEN];
    float w_l[CLIP_MAX_LEN], w_g[CLIP_MAX_LEN];
    clip_tok_encode_weighted(&e->tok, text, ids_l, w_l, CLIP_EOS);
    uint32_t eos_g = clip_tok_encode_weighted(&e->tok, text, ids_g, w_g, 0);

    if (run_clip(e, &e->clip_l, ids_l, 0, 12, 768, 12, 1 /* quick_gelu */, e->tmp_cl, 0, 0)) return -1;
    if (run_clip(e, &e->clip_g, ids_g, eos_g, 32, 1280, 20, 2 /* gelu_exact */, e->tmp_cg, pooled_out, 0)) return -1;
    apply_clip_token_weights(e->tmp_cl, 768, w_l);
    apply_clip_token_weights(e->tmp_cg, 1280, w_g);
    op_concat(e, e->tmp_cl, e->tmp_cg, ctx_out, CLIP_MAX_LEN, 768, 1280);
    return gpu_flush(e);
}

// =================================================================================================
// SDXL UNET
// =================================================================================================

static void sinusoidal_embed(float val, int dim, float* out) {
    int half = dim / 2;
    float log10000 = logf(10000.0f);
    for (int i = 0; i < half; i++) {
        float freq = expf(-log10000 * (float)i / (float)half);
        float arg = val * freq;
        out[i] = cosf(arg);          // flip_sin_to_cos = True
        out[half + i] = sinf(arg);
    }
}

static void resnet_2d(SdxlEngine* e, const SdxlPack* p, const char* prefix,
                      uint64_t in, uint64_t out, uint32_t H, uint32_t W, uint32_t Cin, uint32_t Cout,
                      uint64_t temb_silu, float norm_eps) {
    uint32_t HW = H * W;
    char nm[96], w[96], b[96];

    snprintf(nm, sizeof(nm), "%s.norm1", prefix);
    gn_op(e, p, nm, in, e->buf_c, HW, Cin, 32, norm_eps, 1);

    snprintf(w, sizeof(w), "%s.conv1.weight", prefix);
    snprintf(b, sizeof(b), "%s.conv1.bias", prefix);
    conv3x3_op(e, p, w, b, e->buf_c, out, H, W, Cin, Cout, 1);

    if (temb_silu) {
        snprintf(w, sizeof(w), "%s.time_emb_proj.weight", prefix);
        snprintf(b, sizeof(b), "%s.time_emb_proj.bias", prefix);
        linear_op(e, p, w, b, temb_silu, e->tproj, 1, 0);
        op_add_temb(e, out, e->tproj, HW, Cout);
    }

    snprintf(nm, sizeof(nm), "%s.norm2", prefix);
    gn_op(e, p, nm, out, e->buf_c, HW, Cout, 32, norm_eps, 1);

    snprintf(w, sizeof(w), "%s.conv2.weight", prefix);
    snprintf(b, sizeof(b), "%s.conv2.bias", prefix);
    conv3x3_op(e, p, w, b, e->buf_c, out, H, W, Cout, Cout, 1);

    if (Cin != Cout) {
        snprintf(w, sizeof(w), "%s.conv_shortcut.weight", prefix);
        snprintf(b, sizeof(b), "%s.conv_shortcut.bias", prefix);
        linear_op(e, p, w, b, in, out, HW, 1);
    } else {
        op_add(e, out, in, out, HW * Cout);
    }
}

// Operates in-place on `x` (must be buf_a; uses buf_b as h, buf_c / buf_im2col / buf_q / buf_k / buf_v as scratch).
// Supports both SDXL (head_dim=64, H=C/64) and SD 1.5 (H=8, head_dim=C/8 -> 40, 80, 160).
static void spatial_transformer(SdxlEngine* e, const SdxlPack* p, const char* prefix,
                                uint64_t x, uint64_t ctx, uint32_t H, uint32_t W, uint32_t C, int n_blocks,
                                uint32_t n_heads) {
    uint32_t M = H * W;
    uint32_t hd = C / n_heads;
    float scale = 1.0f / sqrtf((float)hd);
    uint64_t h = e->buf_b;
    char nm[96], w[96], b[96];

    snprintf(nm, sizeof(nm), "%s.norm", prefix);
    gn_op(e, p, nm, x, e->buf_c, M, C, 32, 1e-6f, 0);

    snprintf(w, sizeof(w), "%s.proj_in.weight", prefix);
    snprintf(b, sizeof(b), "%s.proj_in.bias", prefix);
    linear_op(e, p, w, b, e->buf_c, h, M, 0);

    for (int i = 0; i < n_blocks; i++) {
        char tb[96];
        snprintf(tb, sizeof(tb), "%s.transformer_blocks.%d", prefix, i);

        // Self-attention
        snprintf(nm, sizeof(nm), "%s.norm1", tb);
        ln_op(e, p, nm, h, e->buf_c, M, C, 1e-5f);
        snprintf(w, sizeof(w), "%s.attn1.to_q.weight", tb);
        linear_op(e, p, w, NULL, e->buf_c, e->buf_q, M, 0);
        snprintf(w, sizeof(w), "%s.attn1.to_k.weight", tb);
        linear_op(e, p, w, NULL, e->buf_c, e->buf_k, M, 0);
        snprintf(w, sizeof(w), "%s.attn1.to_v.weight", tb);
        linear_op(e, p, w, NULL, e->buf_c, e->buf_v, M, 0);
        if (hd == 64u) op_attn64(e, e->buf_q, e->buf_k, e->buf_v, e->buf_c, M, M, n_heads, scale, 0);
        else op_attn_hd(e, e->buf_q, e->buf_k, e->buf_v, e->buf_c, M, M, n_heads, hd, scale);
        snprintf(w, sizeof(w), "%s.attn1.to_out.0.weight", tb);
        snprintf(b, sizeof(b), "%s.attn1.to_out.0.bias", tb);
        linear_op(e, p, w, b, e->buf_c, h, M, 1);

        // Cross-attention
        snprintf(nm, sizeof(nm), "%s.norm2", tb);
        ln_op(e, p, nm, h, e->buf_c, M, C, 1e-5f);
        snprintf(w, sizeof(w), "%s.attn2.to_q.weight", tb);
        linear_op(e, p, w, NULL, e->buf_c, e->buf_q, M, 0);
        snprintf(w, sizeof(w), "%s.attn2.to_k.weight", tb);
        linear_op(e, p, w, NULL, ctx, e->buf_k, CLIP_MAX_LEN, 0);
        snprintf(w, sizeof(w), "%s.attn2.to_v.weight", tb);
        linear_op(e, p, w, NULL, ctx, e->buf_v, CLIP_MAX_LEN, 0);
        if (hd == 64u) op_attn64(e, e->buf_q, e->buf_k, e->buf_v, e->buf_c, M, CLIP_MAX_LEN, n_heads, scale, 0);
        else op_attn_hd(e, e->buf_q, e->buf_k, e->buf_v, e->buf_c, M, CLIP_MAX_LEN, n_heads, hd, scale);
        snprintf(w, sizeof(w), "%s.attn2.to_out.0.weight", tb);
        snprintf(b, sizeof(b), "%s.attn2.to_out.0.bias", tb);
        linear_op(e, p, w, b, e->buf_c, h, M, 1);

        // GEGLU FFN (chunked over spatial rows so scratch never exceeds 16 MiB)
        snprintf(nm, sizeof(nm), "%s.norm3", tb);
        ln_op(e, p, nm, h, e->buf_c, M, C, 1e-5f);
        uint32_t mc_max = (C >= 1280u) ? 512u : 1024u;
        snprintf(w, sizeof(w), "%s.ff.net.0.proj.weight", tb);
        snprintf(b, sizeof(b), "%s.ff.net.0.proj.bias", tb);
        char w2[96], b2[96];
        snprintf(w2, sizeof(w2), "%s.ff.net.2.weight", tb);
        snprintf(b2, sizeof(b2), "%s.ff.net.2.bias", tb);
        for (uint32_t r0 = 0; r0 < M; r0 += mc_max) {
            uint32_t mc = (M - r0 < mc_max) ? (M - r0) : mc_max;
            linear_op(e, p, w, b, e->buf_c + (uint64_t)r0 * C * 4ull, e->buf_im2col, mc, 0);
            op_geglu(e, e->buf_im2col, e->buf_q, mc, 4u * C);
            linear_op(e, p, w2, b2, e->buf_q, h + (uint64_t)r0 * C * 4ull, mc, 1);
        }
    }

    snprintf(w, sizeof(w), "%s.proj_out.weight", prefix);
    snprintf(b, sizeof(b), "%s.proj_out.bias", prefix);
    linear_op(e, p, w, b, h, x, M, 1);
}

static void push_skip(SdxlEngine* e, uint64_t src, uint32_t HW, uint32_t C, uint64_t* skip_off) {
    uint64_t sz = ((uint64_t)HW * C * 4ull + 255ull) & ~255ull;
    if (*skip_off + sz > e->buf_skips_bytes) {
        fail(e, "skip_pool overflow (%llu > %llu)", (unsigned long long)(*skip_off + sz), (unsigned long long)e->buf_skips_bytes);
        return;
    }
    uint64_t dst = e->skip_pool + *skip_off;
    *skip_off += sz;
    op_copy(e, src, dst, HW * C);
    e->skip_ptr[e->n_skips] = dst;
    e->skip_c[e->n_skips] = C;
    e->n_skips++;
}

// Fine-grained inter-block thermal breather + responsive cancel checkpoint
static int block_breather(SdxlEngine* e) {
    if (e->eco_mode <= 0) {
        if (e->cancel_fn && e->cancel_fn(e->cancel_user)) { gpu_begin(e); return 1; }
        return 0;
    }
    if (gpu_flush(e)) return -1;
    if (e->cancel_fn && e->cancel_fn(e->cancel_user)) return 1;
#ifndef SDXL_CPU
    uint32_t us = (e->eco_mode == 1) ? 25000u : (e->eco_mode == 2 ? 80000u : 100000u);
    if (e->is_sd15) us >>= 1; // SD 1.5 blocks are ~4x lighter, keep breathers snappy!
    if (!e->neo) us >>= 1;    // Base PS4 (18 CUs, ~110W) runs cooler than Pro (36 CUs, ~160W)
    sceKernelUsleep(us);
    e->stats.cooldown_ms += (double)us / 1000.0;
#endif
    return 0;
}

// SD 1.5 UNet forward pass (860M params: 320/640/1280/1280, H=8 heads, 1 transformer block/attn)
static int unet_sd15_forward(SdxlEngine* e, uint64_t sample_in, float timestep,
                             uint64_t ctx, uint32_t img_h, uint32_t img_w, uint64_t eps_out) {
    const SdxlPack* p = &e->unet;
    uint32_t H = img_h / 8u, W = img_w / 8u;
    int br;

    // 1. Time embedding: sinusoidal(320) -> linear_1(1280) -> SiLU -> linear_2(1280) -> SiLU
    sinusoidal_embed(timestep, 320, (float*)(uintptr_t)e->t_raw);
    linear_op(e, p, "time_embedding.linear_1.weight", "time_embedding.linear_1.bias", e->t_raw, e->tproj, 1, 0);
    op_act(e, e->tproj, 1280, 0);
    linear_op(e, p, "time_embedding.linear_2.weight", "time_embedding.linear_2.bias", e->tproj, e->temb_silu, 1, 0);
    op_act(e, e->temb_silu, 1280, 0);

    e->n_skips = 0;
    uint64_t skip_off = 0;

    // conv_in -> buf_a (320)
    conv3x3_op(e, p, "conv_in.weight", "conv_in.bias", sample_in, e->buf_a, H, W, 4, 320, 1);
    push_skip(e, e->buf_a, H * W, 320, &skip_off);

    // down_blocks.0 (320): 2x (resnet + attn[H=8,D=40]) + downsample
    resnet_2d(e, p, "down_blocks.0.resnets.0", e->buf_a, e->buf_b, H, W, 320, 320, e->temb_silu, 1e-5f);
    op_copy(e, e->buf_b, e->buf_a, H * W * 320);
    spatial_transformer(e, p, "down_blocks.0.attentions.0", e->buf_a, ctx, H, W, 320, 1, 8);
    push_skip(e, e->buf_a, H * W, 320, &skip_off);
    resnet_2d(e, p, "down_blocks.0.resnets.1", e->buf_a, e->buf_b, H, W, 320, 320, e->temb_silu, 1e-5f);
    op_copy(e, e->buf_b, e->buf_a, H * W * 320);
    spatial_transformer(e, p, "down_blocks.0.attentions.1", e->buf_a, ctx, H, W, 320, 1, 8);
    push_skip(e, e->buf_a, H * W, 320, &skip_off);
    conv3x3_op(e, p, "down_blocks.0.downsamplers.0.conv.weight", "down_blocks.0.downsamplers.0.conv.bias",
               e->buf_a, e->buf_b, H, W, 320, 320, 2);
    H /= 2; W /= 2;
    push_skip(e, e->buf_b, H * W, 320, &skip_off);
    if ((br = block_breather(e)) != 0) return br;

    // down_blocks.1 (640): 2x (resnet + attn[H=8,D=80]) + downsample
    resnet_2d(e, p, "down_blocks.1.resnets.0", e->buf_b, e->buf_a, H, W, 320, 640, e->temb_silu, 1e-5f);
    spatial_transformer(e, p, "down_blocks.1.attentions.0", e->buf_a, ctx, H, W, 640, 1, 8);
    push_skip(e, e->buf_a, H * W, 640, &skip_off);
    resnet_2d(e, p, "down_blocks.1.resnets.1", e->buf_a, e->buf_b, H, W, 640, 640, e->temb_silu, 1e-5f);
    op_copy(e, e->buf_b, e->buf_a, H * W * 640);
    spatial_transformer(e, p, "down_blocks.1.attentions.1", e->buf_a, ctx, H, W, 640, 1, 8);
    push_skip(e, e->buf_a, H * W, 640, &skip_off);
    conv3x3_op(e, p, "down_blocks.1.downsamplers.0.conv.weight", "down_blocks.1.downsamplers.0.conv.bias",
               e->buf_a, e->buf_b, H, W, 640, 640, 2);
    H /= 2; W /= 2;
    push_skip(e, e->buf_b, H * W, 640, &skip_off);
    if ((br = block_breather(e)) != 0) return br;

    // down_blocks.2 (1280): 2x (resnet + attn[H=8,D=160]) + downsample
    resnet_2d(e, p, "down_blocks.2.resnets.0", e->buf_b, e->buf_a, H, W, 640, 1280, e->temb_silu, 1e-5f);
    spatial_transformer(e, p, "down_blocks.2.attentions.0", e->buf_a, ctx, H, W, 1280, 1, 8);
    push_skip(e, e->buf_a, H * W, 1280, &skip_off);
    resnet_2d(e, p, "down_blocks.2.resnets.1", e->buf_a, e->buf_b, H, W, 1280, 1280, e->temb_silu, 1e-5f);
    op_copy(e, e->buf_b, e->buf_a, H * W * 1280);
    spatial_transformer(e, p, "down_blocks.2.attentions.1", e->buf_a, ctx, H, W, 1280, 1, 8);
    push_skip(e, e->buf_a, H * W, 1280, &skip_off);
    conv3x3_op(e, p, "down_blocks.2.downsamplers.0.conv.weight", "down_blocks.2.downsamplers.0.conv.bias",
               e->buf_a, e->buf_b, H, W, 1280, 1280, 2);
    H /= 2; W /= 2;
    push_skip(e, e->buf_b, H * W, 1280, &skip_off);

    // down_blocks.3 (1280): 2x resnet (no attn, no downsample)
    resnet_2d(e, p, "down_blocks.3.resnets.0", e->buf_b, e->buf_a, H, W, 1280, 1280, e->temb_silu, 1e-5f);
    push_skip(e, e->buf_a, H * W, 1280, &skip_off);
    resnet_2d(e, p, "down_blocks.3.resnets.1", e->buf_a, e->buf_b, H, W, 1280, 1280, e->temb_silu, 1e-5f);
    op_copy(e, e->buf_b, e->buf_a, H * W * 1280);
    push_skip(e, e->buf_a, H * W, 1280, &skip_off);
    if ((br = block_breather(e)) != 0) return br;

    // mid_block (1280): resnet.0 -> attn[H=8,D=160] -> resnet.1
    resnet_2d(e, p, "mid_block.resnets.0", e->buf_a, e->buf_b, H, W, 1280, 1280, e->temb_silu, 1e-5f);
    op_copy(e, e->buf_b, e->buf_a, H * W * 1280);
    spatial_transformer(e, p, "mid_block.attentions.0", e->buf_a, ctx, H, W, 1280, 1, 8);
    resnet_2d(e, p, "mid_block.resnets.1", e->buf_a, e->buf_b, H, W, 1280, 1280, e->temb_silu, 1e-5f);
    op_copy(e, e->buf_b, e->buf_a, H * W * 1280);

    // up_blocks.0 (1280): 3x resnet (no attn) + upsample
    uint32_t cur_c = 1280;
    for (int i = 0; i < 3; i++) {
        char rn[96];
        int sk = --e->n_skips;
        op_concat(e, e->buf_a, e->skip_ptr[sk], e->buf_b, H * W, cur_c, e->skip_c[sk]);
        snprintf(rn, sizeof(rn), "up_blocks.0.resnets.%d", i);
        resnet_2d(e, p, rn, e->buf_b, e->buf_a, H, W, cur_c + e->skip_c[sk], 1280, e->temb_silu, 1e-5f);
        cur_c = 1280;
    }
    op_up2x(e, e->buf_a, e->buf_b, H, W, 1280);
    H *= 2; W *= 2;
    conv3x3_op(e, p, "up_blocks.0.upsamplers.0.conv.weight", "up_blocks.0.upsamplers.0.conv.bias",
               e->buf_b, e->buf_a, H, W, 1280, 1280, 1);
    if ((br = block_breather(e)) != 0) return br;

    // up_blocks.1 (1280): 3x (resnet + attn[H=8,D=160]) + upsample
    for (int i = 0; i < 3; i++) {
        char rn[96], an[96];
        int sk = --e->n_skips;
        op_concat(e, e->buf_a, e->skip_ptr[sk], e->buf_b, H * W, cur_c, e->skip_c[sk]);
        snprintf(rn, sizeof(rn), "up_blocks.1.resnets.%d", i);
        resnet_2d(e, p, rn, e->buf_b, e->buf_a, H, W, cur_c + e->skip_c[sk], 1280, e->temb_silu, 1e-5f);
        snprintf(an, sizeof(an), "up_blocks.1.attentions.%d", i);
        spatial_transformer(e, p, an, e->buf_a, ctx, H, W, 1280, 1, 8);
        cur_c = 1280;
    }
    op_up2x(e, e->buf_a, e->buf_b, H, W, 1280);
    H *= 2; W *= 2;
    conv3x3_op(e, p, "up_blocks.1.upsamplers.0.conv.weight", "up_blocks.1.upsamplers.0.conv.bias",
               e->buf_b, e->buf_a, H, W, 1280, 1280, 1);
    if ((br = block_breather(e)) != 0) return br;

    // up_blocks.2 (640): 3x (resnet + attn[H=8,D=80]) + upsample
    for (int i = 0; i < 3; i++) {
        char rn[96], an[96];
        int sk = --e->n_skips;
        op_concat(e, e->buf_a, e->skip_ptr[sk], e->buf_b, H * W, cur_c, e->skip_c[sk]);
        snprintf(rn, sizeof(rn), "up_blocks.2.resnets.%d", i);
        resnet_2d(e, p, rn, e->buf_b, e->buf_a, H, W, cur_c + e->skip_c[sk], 640, e->temb_silu, 1e-5f);
        snprintf(an, sizeof(an), "up_blocks.2.attentions.%d", i);
        spatial_transformer(e, p, an, e->buf_a, ctx, H, W, 640, 1, 8);
        cur_c = 640;
    }
    op_up2x(e, e->buf_a, e->buf_b, H, W, 640);
    H *= 2; W *= 2;
    conv3x3_op(e, p, "up_blocks.2.upsamplers.0.conv.weight", "up_blocks.2.upsamplers.0.conv.bias",
               e->buf_b, e->buf_a, H, W, 640, 640, 1);
    if ((br = block_breather(e)) != 0) return br;

    // up_blocks.3 (320): 3x (resnet + attn[H=8,D=40]), no upsample
    for (int i = 0; i < 3; i++) {
        char rn[96], an[96];
        int sk = --e->n_skips;
        op_concat(e, e->buf_a, e->skip_ptr[sk], e->buf_b, H * W, cur_c, e->skip_c[sk]);
        snprintf(rn, sizeof(rn), "up_blocks.3.resnets.%d", i);
        resnet_2d(e, p, rn, e->buf_b, e->buf_a, H, W, cur_c + e->skip_c[sk], 320, e->temb_silu, 1e-5f);
        snprintf(an, sizeof(an), "up_blocks.3.attentions.%d", i);
        spatial_transformer(e, p, an, e->buf_a, ctx, H, W, 320, 1, 8);
        cur_c = 320;
    }

    gn_op(e, p, "conv_norm_out", e->buf_a, e->buf_b, H * W, 320, 32, 1e-5f, 1);
    conv3x3_op(e, p, "conv_out.weight", "conv_out.bias", e->buf_b, eps_out, H, W, 320, 4, 1);
    return gpu_flush(e);
}

static int unet_forward(SdxlEngine* e, uint64_t sample_in, float timestep,
                        uint64_t ctx, uint64_t pooled, uint32_t img_h, uint32_t img_w, uint64_t eps_out) {
    if (e->is_sd15) return unet_sd15_forward(e, sample_in, timestep, ctx, img_h, img_w, eps_out);
    const SdxlPack* p = &e->unet;
    uint32_t H = img_h / 8u, W = img_w / 8u;
    int br;

    // 1. Compute temb = time_embedding(t) + add_embedding([pooled, time_ids_sin])
    sinusoidal_embed(timestep, 320, (float*)(uintptr_t)e->t_raw);
    float* ai = (float*)(uintptr_t)e->add_in;
    memcpy(ai, (const void*)(uintptr_t)pooled, 1280u * 4u);
    float tids[6] = {(float)img_h, (float)img_w, 0.0f, 0.0f, (float)img_h, (float)img_w};
    for (int i = 0; i < 6; i++) sinusoidal_embed(tids[i], 256, ai + 1280 + i * 256);

    linear_op(e, p, "time_embedding.linear_1.weight", "time_embedding.linear_1.bias", e->t_raw, e->tproj, 1, 0);
    op_act(e, e->tproj, 1280, 0);
    linear_op(e, p, "time_embedding.linear_2.weight", "time_embedding.linear_2.bias", e->tproj, e->temb, 1, 0);

    linear_op(e, p, "add_embedding.linear_1.weight", "add_embedding.linear_1.bias", e->add_in, e->tproj, 1, 0);
    op_act(e, e->tproj, 1280, 0);
    linear_op(e, p, "add_embedding.linear_2.weight", "add_embedding.linear_2.bias", e->tproj, e->aug_emb, 1, 0);

    op_add(e, e->temb, e->aug_emb, e->temb, 1280);
    op_copy(e, e->temb, e->temb_silu, 1280);
    op_act(e, e->temb_silu, 1280, 0);

    e->n_skips = 0;
    uint64_t skip_off = 0;

    // conv_in -> buf_a (320)
    conv3x3_op(e, p, "conv_in.weight", "conv_in.bias", sample_in, e->buf_a, H, W, 4, 320, 1);
    push_skip(e, e->buf_a, H * W, 320, &skip_off);

    // down_blocks.0: 2 resnets (320), downsample -> H/2, W/2
    resnet_2d(e, p, "down_blocks.0.resnets.0", e->buf_a, e->buf_b, H, W, 320, 320, e->temb_silu, 1e-5f);
    push_skip(e, e->buf_b, H * W, 320, &skip_off);
    resnet_2d(e, p, "down_blocks.0.resnets.1", e->buf_b, e->buf_a, H, W, 320, 320, e->temb_silu, 1e-5f);
    push_skip(e, e->buf_a, H * W, 320, &skip_off);
    conv3x3_op(e, p, "down_blocks.0.downsamplers.0.conv.weight", "down_blocks.0.downsamplers.0.conv.bias",
               e->buf_a, e->buf_b, H, W, 320, 320, 2);
    H /= 2; W /= 2;
    push_skip(e, e->buf_b, H * W, 320, &skip_off);
    if ((br = block_breather(e)) != 0) return br;

    // down_blocks.1: 2x (resnet + 2-block attn, 640), downsample -> H/2, W/2
    resnet_2d(e, p, "down_blocks.1.resnets.0", e->buf_b, e->buf_a, H, W, 320, 640, e->temb_silu, 1e-5f);
    spatial_transformer(e, p, "down_blocks.1.attentions.0", e->buf_a, ctx, H, W, 640, 2, 10);
    push_skip(e, e->buf_a, H * W, 640, &skip_off);
    resnet_2d(e, p, "down_blocks.1.resnets.1", e->buf_a, e->buf_b, H, W, 640, 640, e->temb_silu, 1e-5f);
    op_copy(e, e->buf_b, e->buf_a, H * W * 640);
    spatial_transformer(e, p, "down_blocks.1.attentions.1", e->buf_a, ctx, H, W, 640, 2, 10);
    push_skip(e, e->buf_a, H * W, 640, &skip_off);
    conv3x3_op(e, p, "down_blocks.1.downsamplers.0.conv.weight", "down_blocks.1.downsamplers.0.conv.bias",
               e->buf_a, e->buf_b, H, W, 640, 640, 2);
    H /= 2; W /= 2;
    push_skip(e, e->buf_b, H * W, 640, &skip_off);
    if ((br = block_breather(e)) != 0) return br;

    // down_blocks.2: 2x (resnet + 10-block attn, 1280), no downsample
    resnet_2d(e, p, "down_blocks.2.resnets.0", e->buf_b, e->buf_a, H, W, 640, 1280, e->temb_silu, 1e-5f);
    spatial_transformer(e, p, "down_blocks.2.attentions.0", e->buf_a, ctx, H, W, 1280, 10, 20);
    push_skip(e, e->buf_a, H * W, 1280, &skip_off);
    if ((br = block_breather(e)) != 0) return br;
    resnet_2d(e, p, "down_blocks.2.resnets.1", e->buf_a, e->buf_b, H, W, 1280, 1280, e->temb_silu, 1e-5f);
    op_copy(e, e->buf_b, e->buf_a, H * W * 1280);
    spatial_transformer(e, p, "down_blocks.2.attentions.1", e->buf_a, ctx, H, W, 1280, 10, 20);
    push_skip(e, e->buf_a, H * W, 1280, &skip_off);
    if ((br = block_breather(e)) != 0) return br;

    // mid_block: resnet0 -> 10-block attn -> resnet1 (1280)
    if (e->cancel_fn && e->cancel_fn(e->cancel_user)) { gpu_begin(e); return 1; }
    resnet_2d(e, p, "mid_block.resnets.0", e->buf_a, e->buf_b, H, W, 1280, 1280, e->temb_silu, 1e-5f);
    op_copy(e, e->buf_b, e->buf_a, H * W * 1280);
    spatial_transformer(e, p, "mid_block.attentions.0", e->buf_a, ctx, H, W, 1280, 10, 20);
    resnet_2d(e, p, "mid_block.resnets.1", e->buf_a, e->buf_b, H, W, 1280, 1280, e->temb_silu, 1e-5f);
    op_copy(e, e->buf_b, e->buf_a, H * W * 1280);
    if ((br = block_breather(e)) != 0) return br;

    // up_blocks.0 (1280, 3x resnet + 10-block attn) + upsample
    uint32_t cur_c = 1280;
    for (int i = 0; i < 3; i++) {
        char rn[96], an[96];
        int sk = --e->n_skips;
        op_concat(e, e->buf_a, e->skip_ptr[sk], e->buf_b, H * W, cur_c, e->skip_c[sk]);
        snprintf(rn, sizeof(rn), "up_blocks.0.resnets.%d", i);
        resnet_2d(e, p, rn, e->buf_b, e->buf_a, H, W, cur_c + e->skip_c[sk], 1280, e->temb_silu, 1e-5f);
        snprintf(an, sizeof(an), "up_blocks.0.attentions.%d", i);
        spatial_transformer(e, p, an, e->buf_a, ctx, H, W, 1280, 10, 20);
        cur_c = 1280;
        if ((br = block_breather(e)) != 0) return br;
    }
    op_up2x(e, e->buf_a, e->buf_b, H, W, 1280);
    H *= 2; W *= 2;
    conv3x3_op(e, p, "up_blocks.0.upsamplers.0.conv.weight", "up_blocks.0.upsamplers.0.conv.bias",
               e->buf_b, e->buf_a, H, W, 1280, 1280, 1);

    // up_blocks.1 (640, 3x resnet + 2-block attn) + upsample
    for (int i = 0; i < 3; i++) {
        char rn[96], an[96];
        int sk = --e->n_skips;
        op_concat(e, e->buf_a, e->skip_ptr[sk], e->buf_b, H * W, cur_c, e->skip_c[sk]);
        snprintf(rn, sizeof(rn), "up_blocks.1.resnets.%d", i);
        resnet_2d(e, p, rn, e->buf_b, e->buf_a, H, W, cur_c + e->skip_c[sk], 640, e->temb_silu, 1e-5f);
        snprintf(an, sizeof(an), "up_blocks.1.attentions.%d", i);
        spatial_transformer(e, p, an, e->buf_a, ctx, H, W, 640, 2, 10);
        cur_c = 640;
    }
    op_up2x(e, e->buf_a, e->buf_b, H, W, 640);
    H *= 2; W *= 2;
    conv3x3_op(e, p, "up_blocks.1.upsamplers.0.conv.weight", "up_blocks.1.upsamplers.0.conv.bias",
               e->buf_b, e->buf_a, H, W, 640, 640, 1);
    if ((br = block_breather(e)) != 0) return br;

    // up_blocks.2 (320, 3x resnet, no attn, no upsample)
    for (int i = 0; i < 3; i++) {
        char rn[96];
        int sk = --e->n_skips;
        op_concat(e, e->buf_a, e->skip_ptr[sk], e->buf_b, H * W, cur_c, e->skip_c[sk]);
        snprintf(rn, sizeof(rn), "up_blocks.2.resnets.%d", i);
        resnet_2d(e, p, rn, e->buf_b, e->buf_a, H, W, cur_c + e->skip_c[sk], 320, e->temb_silu, 1e-5f);
        cur_c = 320;
    }

    // conv_norm_out -> SiLU -> conv_out (320 -> 4)
    gn_op(e, p, "conv_norm_out", e->buf_a, e->buf_b, H * W, 320, 32, 1e-5f, 1);
    conv3x3_op(e, p, "conv_out.weight", "conv_out.bias", e->buf_b, eps_out, H, W, 320, 4, 1);
    return gpu_flush(e);
}

// =================================================================================================
// TILED SDXL VAE DECODER (32x32 latent tiles -> 256x256 RGB with 8-latent overlap feathering)
// =================================================================================================

static int vae_decode_tile(SdxlEngine* e, uint64_t lat_tile, uint32_t th, uint32_t tw, uint64_t rgb_out) {
    const SdxlPack* p = &e->vae;
    uint32_t H = th, W = tw;

    // post_quant_conv (1x1, 4->4) -> conv_in (3x3, 4->512)
    linear_op(e, p, "post_quant_conv.weight", "post_quant_conv.bias", lat_tile, e->buf_b, H * W, 0);
    conv3x3_op(e, p, "decoder.conv_in.weight", "decoder.conv_in.bias", e->buf_b, e->buf_a, H, W, 4, 512, 1);

    // mid_block: resnet0 -> attentions.0 (D=512) -> resnet1
    resnet_2d(e, p, "decoder.mid_block.resnets.0", e->buf_a, e->buf_b, H, W, 512, 512, 0, 1e-6f);
    gn_op(e, p, "decoder.mid_block.attentions.0.group_norm", e->buf_b, e->buf_c, H * W, 512, 32, 1e-6f, 0);
    linear_op(e, p, "decoder.mid_block.attentions.0.to_q.weight", "decoder.mid_block.attentions.0.to_q.bias", e->buf_c, e->buf_q, H * W, 0);
    linear_op(e, p, "decoder.mid_block.attentions.0.to_k.weight", "decoder.mid_block.attentions.0.to_k.bias", e->buf_c, e->buf_k, H * W, 0);
    linear_op(e, p, "decoder.mid_block.attentions.0.to_v.weight", "decoder.mid_block.attentions.0.to_v.bias", e->buf_c, e->buf_v, H * W, 0);
    op_attn512(e, e->buf_q, e->buf_k, e->buf_v, e->buf_c, H * W, 1.0f / sqrtf(512.0f));
    linear_op(e, p, "decoder.mid_block.attentions.0.to_out.0.weight", "decoder.mid_block.attentions.0.to_out.0.bias",
              e->buf_c, e->buf_b, H * W, 1);
    resnet_2d(e, p, "decoder.mid_block.resnets.1", e->buf_b, e->buf_a, H, W, 512, 512, 0, 1e-6f);
    if (gpu_flush(e)) return -1;

    static const uint32_t up_ch[4] = {512, 512, 256, 128};
    uint32_t cur_c = 512;
    for (int u = 0; u < 4; u++) {
        uint32_t out_c = up_ch[u];
        for (int r = 0; r < 3; r++) {
            char rn[96];
            snprintf(rn, sizeof(rn), "decoder.up_blocks.%d.resnets.%d", u, r);
            resnet_2d(e, p, rn, e->buf_a, e->buf_b, H, W, cur_c, out_c, 0, 1e-6f);
            op_copy(e, e->buf_b, e->buf_a, H * W * out_c);
            cur_c = out_c;
        }
        if (u < 3) {
            char uw[96], ub[96];
            op_up2x(e, e->buf_a, e->buf_b, H, W, cur_c);
            H *= 2; W *= 2;
            snprintf(uw, sizeof(uw), "decoder.up_blocks.%d.upsamplers.0.conv.weight", u);
            snprintf(ub, sizeof(ub), "decoder.up_blocks.%d.upsamplers.0.conv.bias", u);
            conv3x3_op(e, p, uw, ub, e->buf_b, e->buf_a, H, W, cur_c, cur_c, 1);
        }
        if (gpu_flush(e)) return -1;
    }

    gn_op(e, p, "decoder.conv_norm_out", e->buf_a, e->buf_b, H * W, 128, 32, 1e-6f, 1);
    conv3x3_op(e, p, "decoder.conv_out.weight", "decoder.conv_out.bias", e->buf_b, rgb_out, H, W, 128, 3, 1);
    return gpu_flush(e);
}

// =================================================================================================
// OPTIONAL TILED SDXL/SD1.5 VAE ENCODER (for on-demand Img2Img / Gallery "Fix Face")
// Uses encoder.* + quant_conv.* either bundled in vae.psd1 or loaded transiently from vae_enc.psd1
// =================================================================================================

static int file_can_open(const char* path);

static int find_vae_encoder_path(const SdxlEngine* e, char* out, size_t max) {
    char cand[4][320];
    int n = 0;
    if (e->model_dir[0]) {
        snprintf(cand[n++], sizeof(cand[0]), "%s/vae_enc.psd1", e->model_dir);
    }
    const char* arch_fn = e->is_sd15 ? "vae_enc_sd15.psd1" : "vae_enc_sdxl.psd1";
    snprintf(cand[n++], sizeof(cand[0]), "/data/garlic-diffusion/%s", arch_fn);
    snprintf(cand[n++], sizeof(cand[0]), "/data/garlic-diffusion/models/%s", arch_fn);
    snprintf(cand[n++], sizeof(cand[0]), "/app0/assets/%s", arch_fn);
    for (int i = 0; i < n; i++) {
        if (file_can_open(cand[i])) {
            if (out && max) snprintf(out, max, "%s", cand[i]);
            return 1;
        }
    }
    return 0;
}

int sdxl_has_vae_encoder(const SdxlEngine* e) {
    if (!e) return 0;
    if (e->vae.t && pack_find(&e->vae, "encoder.conv_in.weight") != NULL) return 1;
    return find_vae_encoder_path(e, NULL, 0);
}

static int vae_encode_tile(SdxlEngine* e, const SdxlPack* p, uint64_t rgb_in, uint32_t img_h, uint32_t img_w, uint64_t lat_out) {
    uint32_t H = img_h, W = img_w;
    conv3x3_op(e, p, "encoder.conv_in.weight", "encoder.conv_in.bias", rgb_in, e->buf_a, H, W, 3, 128, 1);

    static const uint32_t dn_ch[4] = {128, 256, 512, 512};
    uint32_t cur_c = 128;
    for (int d = 0; d < 4; d++) {
        uint32_t out_c = dn_ch[d];
        for (int r = 0; r < 2; r++) {
            char rn[96];
            snprintf(rn, sizeof(rn), "encoder.down_blocks.%d.resnets.%d", d, r);
            resnet_2d(e, p, rn, e->buf_a, e->buf_b, H, W, cur_c, out_c, 0, 1e-6f);
            op_copy(e, e->buf_b, e->buf_a, H * W * out_c);
            cur_c = out_c;
        }
        if (d < 3) {
            char dw[96], db[96];
            snprintf(dw, sizeof(dw), "encoder.down_blocks.%d.downsamplers.0.conv.weight", d);
            snprintf(db, sizeof(db), "encoder.down_blocks.%d.downsamplers.0.conv.bias", d);
            conv3x3_op(e, p, dw, db, e->buf_a, e->buf_b, H, W, cur_c, cur_c, 2);
            H /= 2; W /= 2;
            op_copy(e, e->buf_b, e->buf_a, H * W * cur_c);
        }
        if (gpu_flush(e)) return -1;
    }

    resnet_2d(e, p, "encoder.mid_block.resnets.0", e->buf_a, e->buf_b, H, W, 512, 512, 0, 1e-6f);
    gn_op(e, p, "encoder.mid_block.attentions.0.group_norm", e->buf_b, e->buf_c, H * W, 512, 32, 1e-6f, 0);
    linear_op(e, p, "encoder.mid_block.attentions.0.to_q.weight", "encoder.mid_block.attentions.0.to_q.bias", e->buf_c, e->buf_q, H * W, 0);
    linear_op(e, p, "encoder.mid_block.attentions.0.to_k.weight", "encoder.mid_block.attentions.0.to_k.bias", e->buf_c, e->buf_k, H * W, 0);
    linear_op(e, p, "encoder.mid_block.attentions.0.to_v.weight", "encoder.mid_block.attentions.0.to_v.bias", e->buf_c, e->buf_v, H * W, 0);
    op_attn512(e, e->buf_q, e->buf_k, e->buf_v, e->buf_c, H * W, 1.0f / sqrtf(512.0f));
    linear_op(e, p, "encoder.mid_block.attentions.0.to_out.0.weight", "encoder.mid_block.attentions.0.to_out.0.bias",
              e->buf_c, e->buf_b, H * W, 1);
    resnet_2d(e, p, "encoder.mid_block.resnets.1", e->buf_b, e->buf_a, H, W, 512, 512, 0, 1e-6f);
    gn_op(e, p, "encoder.conv_norm_out", e->buf_a, e->buf_b, H * W, 512, 32, 1e-6f, 1);
    conv3x3_op(e, p, "encoder.conv_out.weight", "encoder.conv_out.bias", e->buf_b, e->buf_a, H, W, 512, 8, 1);
    linear_op(e, p, "quant_conv.weight", "quant_conv.bias", e->buf_a, lat_out, H * W, 0);
    return gpu_flush(e);
}

static int vae_encode_tiled(SdxlEngine* e, const uint8_t* rgb_in, uint32_t img_h, uint32_t img_w, float* lat_out,
                            SdxlProgressFn progress, void* user) {
    const SdxlPack* enc_pack = NULL;
    SdxlPack tmp_pack;
    memset(&tmp_pack, 0, sizeof(tmp_pack));
    uint64_t saved_next = e->next;
    int loaded_tmp = 0;

    if (e->vae.t && pack_find(&e->vae, "encoder.conv_in.weight") != NULL) {
        enc_pack = &e->vae;
    } else {
        char ep[320] = {0};
        if (!find_vae_encoder_path(e, ep, sizeof(ep))) {
            fail(e, "VAE encoder not bundled (place vae_enc_sdxl.psd1 / vae_enc_sd15.psd1 in /data/garlic-diffusion/ or re-convert with --with-vae-encoder)");
            return -2;
        }
        if (progress) progress(user, 0.03f, "Loading optional VAE encoder...");
        if (pack_load(e, &tmp_pack, ep, NULL, NULL, 0, 0, NULL) != 0) {
            e->next = saved_next;
            free(tmp_pack.t);
            fail(e, "Not enough free GPU RAM for transient VAE encoder");
            return -1;
        }
        enc_pack = &tmp_pack;
        loaded_tmp = 1;
    }

    uint32_t LH = img_h / 8u, LW = img_w / 8u;
    float vae_scale = (e->vae.vae_scale > 0.0f) ? e->vae.vae_scale : (e->is_sd15 ? 0.18215f : 0.13025f);
    float* acc = (float*)calloc((size_t)LH * LW * 4u, sizeof(float));
    float* wgt = (float*)calloc((size_t)LH * LW, sizeof(float));
    if (!acc || !wgt) {
        free(acc); free(wgt);
        if (loaded_tmp) { e->next = saved_next; free(tmp_pack.t); }
        fail(e, "oom vae enc acc");
        return -1;
    }

    const uint32_t TILE = e->vae_tile ? e->vae_tile : 32u;
    const uint32_t OV = (TILE >= 32u) ? 8u : 6u;
    const uint32_t STRIDE = TILE - OV;
    float* rgb_tile = (float*)(uintptr_t)e->buf_q; // up to 256*256*3*4 = 768 KiB
    uint64_t lat8_gpu = e->lat_in;                 // up to 32*32*8*4 = 32 KiB
    const float* lat8 = (const float*)(uintptr_t)lat8_gpu;

    int rc = 0;
    for (uint32_t ty0 = 0; rc == 0 && ty0 < LH; ty0 += STRIDE) {
        uint32_t th = (LH - ty0 < TILE) ? (LH - ty0) : TILE;
        uint32_t y_off = (ty0 + th == LH && LH > TILE) ? (LH - TILE) : ty0;
        if (y_off + th > LH) th = LH - y_off;
        for (uint32_t tx0 = 0; rc == 0 && tx0 < LW; tx0 += STRIDE) {
            uint32_t tw = (LW - tx0 < TILE) ? (LW - tx0) : TILE;
            uint32_t x_off = (tx0 + tw == LW && LW > TILE) ? (LW - TILE) : tx0;
            if (x_off + tw > LW) tw = LW - x_off;

            uint32_t ph = th * 8u, pw = tw * 8u;
            for (uint32_t py = 0; py < ph; py++) {
                const uint8_t* src = rgb_in + ((size_t)(y_off * 8u + py) * img_w + x_off * 8u) * 3u;
                float* dst = rgb_tile + (size_t)py * pw * 3u;
                for (uint32_t px = 0; px < pw * 3u; px++) {
                    dst[px] = (float)src[px] / 127.5f - 1.0f;
                }
            }
            if (vae_encode_tile(e, enc_pack, e->buf_q, ph, pw, lat8_gpu) != 0) { rc = -1; break; }

            for (uint32_t ly = 0; ly < th; ly++) {
                float wy = 1.0f;
                if (y_off > 0 && ly < OV) wy = (float)(ly + 1u) / (float)(OV + 1u);
                if (y_off + th < LH && ly >= th - OV) {
                    float w2 = (float)(th - ly) / (float)(OV + 1u);
                    if (w2 < wy) wy = w2;
                }
                for (uint32_t lx = 0; lx < tw; lx++) {
                    float wx = 1.0f;
                    if (x_off > 0 && lx < OV) wx = (float)(lx + 1u) / (float)(OV + 1u);
                    if (x_off + tw < LW && lx >= tw - OV) {
                        float w2 = (float)(tw - lx) / (float)(OV + 1u);
                        if (w2 < wx) wx = w2;
                    }
                    float w = wx * wy;
                    size_t si = ((size_t)ly * tw + lx) * 8u;
                    size_t di = ((size_t)(y_off + ly) * LW + (x_off + lx));
                    for (int c = 0; c < 4; c++) acc[di * 4u + c] += lat8[si + c] * w;
                    wgt[di] += w;
                }
            }
            if (x_off + tw >= LW) break;
        }
        if (y_off + th >= LH) break;
    }

    if (rc == 0) {
        for (size_t i = 0; i < (size_t)LH * LW; i++) {
            float inv = wgt[i] > 1e-6f ? (vae_scale / wgt[i]) : vae_scale;
            for (int c = 0; c < 4; c++) lat_out[i * 4u + c] = acc[i * 4u + c] * inv;
        }
    }
    free(acc); free(wgt);
    if (loaded_tmp) {
        e->next = saved_next;
        free(tmp_pack.t);
    }
    return rc;
}

static int vae_decode_tiled(SdxlEngine* e, const float* lat, uint32_t img_h, uint32_t img_w, uint8_t* rgb_out,
                            SdxlProgressFn progress, void* user) {
    uint32_t LH = img_h / 8u, LW = img_w / 8u;
    float inv_scale = 1.0f / (e->vae.vae_scale > 0.0f ? e->vae.vae_scale : 0.13025f);
    float* acc = (float*)calloc((size_t)img_h * img_w * 3, sizeof(float));
    float* wgt = (float*)calloc((size_t)img_h * img_w, sizeof(float));
    if (!acc || !wgt) { free(acc); free(wgt); fail(e, "oom vae acc"); return -1; }

    const uint32_t TILE = e->vae_tile ? e->vae_tile : 32u;
    const uint32_t OV = (TILE >= 32u) ? 8u : 6u;
    const uint32_t STRIDE = TILE - OV;
    float* lat_tile = (float*)(uintptr_t)e->lat_in;
    float* rgb_tile = (float*)(uintptr_t)e->eps_c; // up to 256*256*3*4 = 768 KiB (fits in buf_q!)
    uint64_t rgb_gpu = e->buf_q;
    rgb_tile = (float*)(uintptr_t)rgb_gpu;

    int total_vt = 0, done_vt = 0;
    for (uint32_t ty0 = 0; ty0 < LH; ty0 += STRIDE) {
        uint32_t th = (LH - ty0 < TILE) ? (LH - ty0) : TILE;
        if (ty0 > 0 && th < TILE && LH >= TILE) { ty0 = LH - TILE; th = TILE; }
        for (uint32_t tx0 = 0; tx0 < LW; tx0 += STRIDE) {
            uint32_t tw = (LW - tx0 < TILE) ? (LW - tx0) : TILE;
            if (tx0 > 0 && tw < TILE && LW >= TILE) { tx0 = LW - TILE; tw = TILE; }
            total_vt++;
            if (tx0 + tw >= LW) break;
        }
        if (ty0 + th >= LH) break;
    }
    double vt0 = now_sec(e);

    for (uint32_t ty0 = 0; ty0 < LH; ty0 += STRIDE) {
        uint32_t th = (LH - ty0 < TILE) ? (LH - ty0) : TILE;
        if (ty0 > 0 && th < TILE && LH >= TILE) { ty0 = LH - TILE; th = TILE; }
        for (uint32_t tx0 = 0; tx0 < LW; tx0 += STRIDE) {
            uint32_t tw = (LW - tx0 < TILE) ? (LW - tx0) : TILE;
            if (tx0 > 0 && tw < TILE && LW >= TILE) { tx0 = LW - TILE; tw = TILE; }

            if (e->cancel_fn && e->cancel_fn(e->cancel_user)) {
                free(acc); free(wgt);
                return 1;
            }
            e->rt_phase = 3;
            e->rt_tiles_done = (uint32_t)done_vt;
            e->rt_tiles_total = (uint32_t)(total_vt > 0 ? total_vt : 1);
            e->rt_tile_avg_sec = (done_vt > 0) ? ((now_sec(e) - vt0) / (double)done_vt) : 0.0;
            if (progress) {
                char vmsg[96];
                snprintf(vmsg, sizeof(vmsg), "Decoding VAE (%d/%d tiles)", done_vt + 1, total_vt);
                progress(user, 0.90f + 0.08f * ((float)done_vt / (float)(total_vt ? total_vt : 1)), vmsg);
            }
            for (uint32_t y = 0; y < th; y++) {
                const float* src = lat + ((size_t)(ty0 + y) * LW + tx0) * 4;
                float* dst = lat_tile + (size_t)y * tw * 4;
                for (uint32_t x = 0; x < tw * 4; x++) dst[x] = src[x] * inv_scale;
            }
            if (vae_decode_tile(e, e->lat_in, th, tw, rgb_gpu)) {
                free(acc); free(wgt);
                return -1;
            }
            e->stats.vae_tiles++;
            done_vt++;
            int vbr = block_breather(e);
            if (vbr != 0) { free(acc); free(wgt); return vbr; }
            uint32_t ph = th * 8u, pw = tw * 8u;
            uint32_t py0 = ty0 * 8u, px0 = tx0 * 8u;
            uint32_t fov = OV * 8u;
            for (uint32_t y = 0; y < ph; y++) {
                float wy = 1.0f;
                if (ty0 > 0 && y < fov) wy = (float)(y + 1) / (float)(fov + 1);
                if (ty0 + th < LH && ph - 1 - y < fov) {
                    float w2 = (float)(ph - y) / (float)(fov + 1);
                    if (w2 < wy) wy = w2;
                }
                for (uint32_t x = 0; x < pw; x++) {
                    float wx = 1.0f;
                    if (tx0 > 0 && x < fov) wx = (float)(x + 1) / (float)(fov + 1);
                    if (tx0 + tw < LW && pw - 1 - x < fov) {
                        float w2 = (float)(pw - x) / (float)(fov + 1);
                        if (w2 < wx) wx = w2;
                    }
                    float wt = wx * wy;
                    size_t dst_i = (size_t)(py0 + y) * img_w + (px0 + x);
                    size_t src_i = ((size_t)y * pw + x) * 3;
                    acc[dst_i * 3 + 0] += rgb_tile[src_i + 0] * wt;
                    acc[dst_i * 3 + 1] += rgb_tile[src_i + 1] * wt;
                    acc[dst_i * 3 + 2] += rgb_tile[src_i + 2] * wt;
                    wgt[dst_i] += wt;
                }
            }
            if (tx0 + tw >= LW) break;
        }
        if (ty0 + th >= LH) break;
    }

    for (size_t i = 0; i < (size_t)img_h * img_w; i++) {
        float inv = wgt[i] > 0.0f ? (1.0f / wgt[i]) : 1.0f;
        for (int c = 0; c < 3; c++) {
            float v = (acc[i * 3 + c] * inv + 1.0f) * 0.5f;
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            rgb_out[i * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
        }
    }
    free(acc);
    free(wgt);
    return 0;
}

// =================================================================================================
// EULER ANCESTRAL SAMPLER + PRNG (xoshiro256** + Box-Muller)
// =================================================================================================

typedef struct { uint64_t s[4]; } Rng;

static uint64_t splitmix64(uint64_t* x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static void rng_seed(Rng* r, uint64_t seed) {
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&seed);
}
static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t rng_next(Rng* r) {
    uint64_t res = rotl64(r->s[1] * 5, 7) * 9;
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0]; r->s[3] ^= r->s[1]; r->s[1] ^= r->s[2]; r->s[0] ^= r->s[3];
    r->s[2] ^= t; r->s[3] = rotl64(r->s[3], 45);
    return res;
}
static float rng_u01(Rng* r) {
    return (float)((rng_next(r) >> 11) * (1.0 / 9007199254740992.0));
}
static void rng_randn(Rng* r, float* dst, size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        float u1 = rng_u01(r);
        if (u1 < 1e-7f) u1 = 1e-7f;
        float u2 = rng_u01(r);
        float mag = sqrtf(-2.0f * logf(u1));
        float ang = 6.28318530718f * u2;
        dst[i] = mag * cosf(ang);
        if (i + 1 < n) dst[i + 1] = mag * sinf(ang);
    }
}

static void build_sigmas(uint32_t steps, int sampler, float* timesteps_out, float* sigmas_out) {
    static double sig_all[1000];
    static double log_sig[1000];
    static int inited = 0;
    if (!inited) {
        double sb0 = sqrt(0.00085), sb1 = sqrt(0.012);
        double ac = 1.0;
        for (int i = 0; i < 1000; i++) {
            double sb = sb0 + (sb1 - sb0) * ((double)i / 999.0);
            double beta = sb * sb;
            ac *= (1.0 - beta);
            sig_all[i] = sqrt((1.0 - ac) / ac);
            log_sig[i] = log(sig_all[i]);
        }
        inited = 1;
    }

    // Use Karras et al. (2022) sigma schedule (rho = 7.0) for DPM++ 2M Karras (sampler == 3)
    // and for Euler a / Euler when steps >= 10, concentrating steps in the medium/low-sigma regime
    // where anime eyes, pupils, eyelashes, mouths, and fingers crystallize!
    if (sampler == 3 || (sampler != 2 && steps >= 10u)) {
        const double rho = 7.0;
        const double inv_rho = 1.0 / rho;
        const double s_max = sig_all[999];
        const double s_min = (steps <= 8u) ? sig_all[25] : sig_all[0]; // ~0.10 for 8-step, 0.0292 for >=9 steps
        const double smax_r = pow(s_max, inv_rho);
        const double smin_r = pow(s_min, inv_rho);
        for (uint32_t i = 0; i < steps; i++) {
            double u = (steps > 1u) ? ((double)i / (double)(steps - 1u)) : 0.0;
            double sig = pow(smax_r + u * (smin_r - smax_r), rho);
            double lsig = log(sig);
            // Continuous log-SNR interpolation (k-diffusion sigma_to_t)
            double t_cont = 0.0;
            if (lsig <= log_sig[0]) {
                t_cont = 0.0;
            } else if (lsig >= log_sig[999]) {
                t_cont = 999.0;
            } else {
                int lo = 0, hi = 998;
                while (lo <= hi) {
                    int mid = (lo + hi) >> 1;
                    if (log_sig[mid + 1] < lsig) lo = mid + 1;
                    else if (log_sig[mid] > lsig) hi = mid - 1;
                    else { lo = mid; break; }
                }
                double denom = log_sig[lo + 1] - log_sig[lo];
                double w = (denom > 1e-12) ? ((lsig - log_sig[lo]) / denom) : 0.0;
                t_cont = (double)lo + w;
            }
            timesteps_out[i] = (float)t_cont;
            sigmas_out[i] = (float)sig;
        }
    } else {
        // Trailing spacing starting at t=999 (used for LCM/DMD2 at any step count, and 4-8 step Hyper-SD Euler)
        double step = 1000.0 / (double)steps;
        for (uint32_t i = 0; i < steps; i++) {
            int t = (int)lround(999.0 - (double)i * step);
            if (t < 0) t = 0;
            if (t > 999) t = 999;
            timesteps_out[i] = (float)t;
            sigmas_out[i] = (float)sig_all[t];
        }
    }
    sigmas_out[steps] = 0.0f;
}

// =================================================================================================
// INIT / FREE / GENERATE
// =================================================================================================

int sdxl_init(SdxlEngine* e, const char* assets_dir, const char* model_dir,
              SdxlProgressFn progress, void* user) {
    memset(e, 0, sizeof(*e));
    if (model_dir && model_dir[0]) {
        const char* slash = strrchr(model_dir, '/');
        snprintf(e->model_name, sizeof(e->model_name), "%s", slash ? slash + 1 : model_dir);
    snprintf(e->model_dir, sizeof(e->model_dir), "%s", model_dir);
    e->applied_lora_spec[0] = 0;
    }

    char path[256];
    snprintf(path, sizeof(path), "%s/clip_tok.bin", assets_dir);
    if (clip_tok_load(&e->tok, path)) {
        fail(e, "cannot load CLIP tokenizer: %s", path);
        return -1;
    }

#ifndef SDXL_CPU
    e->tsc_freq = sceKernelGetTscFrequency();
    e->neo = (sceKernelIsNeoMode() == 1);
    if (reserve_garlic(e)) { fail(e, "could not reserve PS4 GPU memory"); return -1; }
    e->dcb = (uint32_t*)(uintptr_t)gpu_alloc(e, DCB_BYTES);
    e->dcb_cap = DCB_BYTES / 4u;
    e->kargs = (uint8_t*)(uintptr_t)gpu_alloc(e, KARGS_BYTES);
    e->kargs_cap = KARGS_BYTES;
    e->label = (volatile uint32_t*)(uintptr_t)gpu_alloc(e, 256);
    e->stamps = (volatile uint64_t*)(uintptr_t)gpu_alloc(e, 256);

    snprintf(path, sizeof(path), "%s/kernels.bin", assets_dir);
    if (load_kernels(e, path)) return -1;

    uint32_t* init = (uint32_t*)(uintptr_t)gpu_alloc(e, 1024);
    sceGnmDrawInitDefaultHardwareState350(init, 256);
    memcpy(e->init_state, init, sizeof(e->init_state));
    gpu_begin(e);

    // Quick GPU self-test: add_f32, gemm_q8 (non-aligned 70x135x64), attn_d64 (20x20x2 causal & non-causal)
    {
        uint64_t mark = e->next;
        uint64_t ta = gpu_alloc(e, 65536), tb = gpu_alloc(e, 65536), tc = gpu_alloc(e, 65536);
        uint64_t td = gpu_alloc(e, 65536), te = gpu_alloc(e, 65536);
        float *fa = (float*)(uintptr_t)ta, *fb = (float*)(uintptr_t)tb, *fc = (float*)(uintptr_t)tc;
        for (int i = 0; i < 256; i++) { fa[i] = (float)i; fb[i] = 100.0f; fc[i] = 0.0f; }
        op_add(e, ta, tb, tc, 256);
        if (gpu_flush(e)) return -1;
        for (int i = 0; i < 256; i++) {
            if (fabsf(fc[i] - ((float)i + 100.0f)) > 1e-3f) {
                fail(e, "GPU self-test add_f32 failed at %d (got %f)", i, fc[i]);
                return -1;
            }
        }
        // Test ALL THREE gemm_q8 variants: M=70, N=135, K=64
        const uint32_t M = 70, N = 135, K = 64;
        int8_t* wq = (int8_t*)(uintptr_t)tb;
        uint16_t* wsc = (uint16_t*)(uintptr_t)td;
        float* bias = (float*)(uintptr_t)te;
        for (uint32_t i = 0; i < M * K; i++) fa[i] = ((float)((int)(i % 19) - 9)) * 0.1f;
        for (uint32_t i = 0; i < N * K; i++) wq[i] = (int8_t)(((int)((i * 7) % 21) - 10) * 5);
        for (uint32_t i = 0; i < N * (K / 32); i++) wsc[i] = 0x3800u; // fp16 0.5
        for (uint32_t i = 0; i < N; i++) bias[i] = (float)i * 0.01f;
        float max_gerr = 0.0f;
        for (int var = 0; var < 3; var++) {
            memset(fc, 0, M * N * 4);
            op_gemm_q8_var(e, var, ta, tb, td, te, tc, M, N, K, 0);
            if (gpu_flush(e)) return -1;
            for (uint32_t m = 0; m < M; m++) {
                for (uint32_t n = 0; n < N; n++) {
                    float ref = bias[n];
                    for (uint32_t k = 0; k < K; k++) ref += fa[m * K + k] * (float)wq[n * K + k] * 0.5f;
                    float err = fabsf(fc[m * N + n] - ref);
                    if (err > max_gerr) max_gerr = err;
                }
            }
        }
        if (max_gerr > 1e-2f) {
            fail(e, "GPU self-test gemm_q8 (3 variants) max_err=%.4f", max_gerr);
            return -1;
        }
        // Test attn_d64: M=20, S=20, H=2 (non-causal + causal)
        const uint32_t AM = 20, AH = 2;
        float *pQ = (float*)(uintptr_t)ta, *pK = (float*)(uintptr_t)tb, *pV = (float*)(uintptr_t)tc, *pO = (float*)(uintptr_t)td;
        for (uint32_t i = 0; i < AM * AH * 64; i++) {
            pQ[i] = sinf((float)i * 0.13f) * 0.5f;
            pK[i] = cosf((float)i * 0.17f) * 0.5f;
            pV[i] = sinf((float)i * 0.23f) * 0.5f;
        }
        float max_aerr = 0.0f;
        for (uint32_t causal = 0; causal <= 1; causal++) {
            memset(pO, 0, AM * AH * 64 * 4);
            op_attn64(e, ta, tb, tc, td, AM, AM, AH, 0.125f, causal);
            if (gpu_flush(e)) return -1;
            for (uint32_t h = 0; h < AH; h++) {
                for (uint32_t m = 0; m < AM; m++) {
                    uint32_t slim = causal ? (m + 1) : AM;
                    float scs[20], mx = -1e30f, ls = 0.0f;
                    for (uint32_t s = 0; s < slim; s++) {
                        float dot = 0.0f;
                        for (int d = 0; d < 64; d++)
                            dot += pQ[(m * AH + h) * 64 + d] * pK[(s * AH + h) * 64 + d];
                        scs[s] = dot * 0.125f;
                        if (scs[s] > mx) mx = scs[s];
                    }
                    for (uint32_t s = 0; s < slim; s++) { scs[s] = expf(scs[s] - mx); ls += scs[s]; }
                    for (int d = 0; d < 64; d++) {
                        float ref = 0.0f;
                        for (uint32_t s = 0; s < slim; s++) ref += (scs[s] / ls) * pV[(s * AH + h) * 64 + d];
                        float err = fabsf(pO[(m * AH + h) * 64 + d] - ref);
                        if (err > max_aerr) max_aerr = err;
                    }
                }
            }
        }
        if (max_aerr > 1e-3f) {
            fail(e, "GPU self-test attn_d64 max_err=%.5f", max_aerr);
            return -1;
        }
        // Test attn_hd for SD 1.5 head dims (D=40, 80, 160)
        static const uint32_t sd15_dims[3] = {40, 80, 160};
        float max_hderr = 0.0f;
        for (int di = 0; di < 3; di++) {
            uint32_t HD = sd15_dims[di], HM = 18, HH = 2;
            float sc_hd = 1.0f / sqrtf((float)HD);
            for (uint32_t i = 0; i < HM * HH * HD; i++) {
                pQ[i] = sinf((float)i * 0.11f) * 0.4f;
                pK[i] = cosf((float)i * 0.19f) * 0.4f;
                pV[i] = sinf((float)i * 0.27f) * 0.4f;
            }
            memset(pO, 0, HM * HH * HD * 4);
            op_attn_hd(e, ta, tb, tc, td, HM, HM, HH, HD, sc_hd);
            if (gpu_flush(e)) return -1;
            for (uint32_t h = 0; h < HH; h++) {
                for (uint32_t m = 0; m < HM; m++) {
                    float scs[18], mx = -1e30f, ls = 0.0f;
                    for (uint32_t s = 0; s < HM; s++) {
                        float dot = 0.0f;
                        for (uint32_t d = 0; d < HD; d++)
                            dot += pQ[(m * HH + h) * HD + d] * pK[(s * HH + h) * HD + d];
                        scs[s] = dot * sc_hd;
                        if (scs[s] > mx) mx = scs[s];
                    }
                    for (uint32_t s = 0; s < HM; s++) { scs[s] = expf(scs[s] - mx); ls += scs[s]; }
                    for (uint32_t d = 0; d < HD; d++) {
                        float ref = 0.0f;
                        for (uint32_t s = 0; s < HM; s++) ref += (scs[s] / ls) * pV[(s * HH + h) * HD + d];
                        float err = fabsf(pO[(m * HH + h) * HD + d] - ref);
                        if (err > max_hderr) max_hderr = err;
                    }
                }
            }
        }
        if (max_hderr > 1e-3f) {
            fail(e, "GPU self-test attn_hd (D=40/80/160) max_err=%.5f", max_hderr);
            return -1;
        }
        e->next = mark;
        glog("GPU self-test PASS: add_f32 OK, gemm_q8=%.5f, attn_d64=%.5f, attn_hd(SD15)=%.5f",
             (double)max_gerr, (double)max_aerr, (double)max_hderr);
    }
#else
    e->mem_size = (model_dir && model_dir[0]) ? (4500ull << 20) : (512ull << 20);
    e->mem = (uint8_t*)malloc((size_t)e->mem_size);
    if (!e->mem) { fail(e, "CPU test OOM"); return -1; }
#endif

    // Base PS4 / Slim (<4500 MiB GPU pool) uses compact scratch (325 MiB vs 493 MiB, saving 168 MiB!)
    // so full 3.7 GiB SDXL models fit comfortably on both Base PS4 and PS4 Pro Neo!
    int compact_mem = (!e->neo || e->mem_size < (4500ull << 20));
    e->buf_main_bytes   = compact_mem ? (48ull << 20) : BUF_MAIN_BYTES;
    e->buf_im2col_bytes = compact_mem ? (36ull << 20) : BUF_IM2COL_BYTES;
    e->buf_skips_bytes  = compact_mem ? (84ull << 20) : BUF_SKIPS_BYTES;
    e->vae_tile         = compact_mem ? 24u : 32u;

    // Allocate scratch buffers first so any OOM is caught before reading GBs of weights
    e->buf_a      = gpu_alloc(e, e->buf_main_bytes);
    e->buf_b      = gpu_alloc(e, e->buf_main_bytes);
    e->buf_c      = gpu_alloc(e, e->buf_main_bytes);
    e->buf_q      = gpu_alloc(e, BUF_QKV_BYTES);
    e->buf_k      = gpu_alloc(e, BUF_QKV_BYTES);
    e->buf_v      = gpu_alloc(e, BUF_QKV_BYTES);
    e->buf_im2col = gpu_alloc(e, e->buf_im2col_bytes);
    e->skip_pool  = gpu_alloc(e, e->buf_skips_bytes);
    e->ctx_cond   = gpu_alloc(e, CLIP_MAX_LEN * 2048ull * 4ull);
    e->ctx_uncond = gpu_alloc(e, CLIP_MAX_LEN * 2048ull * 4ull);
    e->pool_cond  = gpu_alloc(e, 1280ull * 4ull);
    e->pool_uncond= gpu_alloc(e, 1280ull * 4ull);
    e->tmp_cl     = gpu_alloc(e, CLIP_MAX_LEN * 768ull * 4ull);
    e->tmp_cg     = gpu_alloc(e, CLIP_MAX_LEN * 1280ull * 4ull);
    e->tok_gpu    = gpu_alloc(e, CLIP_MAX_LEN * 4ull);
    e->t_raw      = gpu_alloc(e, 320ull * 4ull);
    e->temb       = gpu_alloc(e, 1280ull * 4ull);
    e->temb_silu  = gpu_alloc(e, 1280ull * 4ull);
    e->tproj      = gpu_alloc(e, 1280ull * 4ull);
    e->add_in     = gpu_alloc(e, 2816ull * 4ull);
    e->aug_emb    = gpu_alloc(e, 1280ull * 4ull);
    uint64_t lat_cap = compact_mem ? 16384ull : 24576ull; // >= 21,504 latents for 1024x1344 / 1344x1024 on PS4 Pro Neo!
    e->lat_in     = gpu_alloc(e, lat_cap * 4ull * 4ull);
    e->eps_u      = gpu_alloc(e, lat_cap * 4ull * 4ull);
    e->eps_c      = gpu_alloc(e, lat_cap * 4ull * 4ull);
    if (!e->eps_c) { fail(e, "not enough GPU memory for scratch"); return -1; }

    // Load optional built-in RealESRGAN Anime v3 upscaler (2.4 MiB) before marking scratch_bytes
    snprintf(path, sizeof(path), "%s/upscaler.psd1", assets_dir);
    {
#ifdef _WIN32
        int ufd = open(path, O_RDONLY | O_BINARY);
#else
        int ufd = open(path, O_RDONLY);
#endif
        if (ufd >= 0) {
            close(ufd);
            if (pack_load(e, &e->upscaler, path, NULL, NULL, 0, 0, NULL) == 0) {
                e->has_upscaler = 1;
                glog("upscaler ready: RealESRGAN Anime v3 (4x -> 1080p, %.2f MiB GPU)",
                     (double)e->upscaler.data_size / 1048576.0);
            } else {
                e->error[0] = 0;
            }
        }
    }
    e->scratch_bytes = e->next;

#ifndef SDXL_CPU
    // Auto-tune all 3 transposed-LDS gemm_q8 variants on real SDXL shape (2304x1280x1280) + bench attn_d64
    {
        static const char* vnames[3] = {"64x64_t", "128x64_t", "128x128_t"};
        double best_ms = 1e9, best_tf = 0.0;
        int best_v = 0;
        double v_ms[3], v_tf[3];
        for (int v = 0; v < 3; v++) {
            // Warmup + timed run
            op_gemm_q8_var(e, v, e->buf_a, e->buf_b, e->buf_q, 0, e->buf_c, 2304, 1280, 1280, 0);
            gpu_flush(e);
            e->gpu_acc_sec = 0.0;
            op_gemm_q8_var(e, v, e->buf_a, e->buf_b, e->buf_q, 0, e->buf_c, 2304, 1280, 1280, 0);
            gpu_flush(e);
            v_ms[v] = e->gpu_acc_sec * 1000.0;
            v_tf[v] = (v_ms[v] > 0.001) ? ((2.0 * 2304.0 * 1280.0 * 1280.0) / (e->gpu_acc_sec * 1e12)) : 0.0;
            if (v_ms[v] < best_ms) { best_ms = v_ms[v]; best_tf = v_tf[v]; best_v = v; }
        }
        e->best_gemm = best_v;
        e->stats.bench_gemm_tflops = best_tf;

        e->gpu_acc_sec = 0.0;
        op_attn64(e, e->buf_q, e->buf_k, e->buf_v, e->buf_c, 2304, 2304, 10, 0.125f, 0);
        gpu_flush(e);
        double at_ms = e->gpu_acc_sec * 1000.0;
        e->stats.bench_attn_ms = at_ms;
        glog("GPU tune gemm_q8(2304x1280x1280): 64x64=%.2fms(%.2fTF) 128x64=%.2fms(%.2fTF) 128x128=%.2fms(%.2fTF) -> %s; attn=%.2fms",
             v_ms[0], v_tf[0], v_ms[1], v_tf[1], v_ms[2], v_tf[2], vnames[best_v], at_ms);

        // Micro-bench all 4 hardware CU power modes (~25ms total) to verify live CU scaling on this console
        for (int m = 0; m < 4; m++) {
            e->eco_mode = m;
            gpu_begin(e);
            e->gpu_acc_sec = 0.0;
            op_gemm_q8_var(e, best_v, e->buf_a, e->buf_b, e->buf_q, 0, e->buf_c, 2304, 1280, 1280, 0);
            gpu_flush(e);
            e->stats.bench_mode_tf[m] = (e->gpu_acc_sec > 1e-6) ? ((2.0 * 2304.0 * 1280.0 * 1280.0) / (e->gpu_acc_sec * 1e12)) : 0.0;
        }
        e->eco_mode = 0;
        gpu_begin(e);
        glog("GPU CU power modes: Turbo=%.2fTF Bal=%.2fTF Quiet=%.2fTF Ultra=%.2fTF",
             e->stats.bench_mode_tf[0], e->stats.bench_mode_tf[1],
             e->stats.bench_mode_tf[2], e->stats.bench_mode_tf[3]);
    }
#endif

    if (!model_dir || !model_dir[0]) return 0;
    return sdxl_reload_model(e, model_dir, progress, user);
}

static int file_can_open(const char* path) {
#ifdef _WIN32
    int fd = open(path, O_RDONLY | O_BINARY);
#else
    int fd = open(path, O_RDONLY);
#endif
    if (fd < 0) return 0;
    close(fd);
    return 1;
}

static int resolve_pack_path(const char* model_dir, const char* fname, char* out, size_t max, int allow_shared, int want_sd15, int log_share) {
    snprintf(out, max, "%s/%s", model_dir, fname);
    if (file_can_open(out)) {
        if (!allow_shared || sdxl_check_pack(out, NULL, NULL) == 0) return 0;
        // Local file exists but is incomplete/corrupt: do not silently mask it with a sibling!
        return 0;
    }
    if (!allow_shared) return -1;

    // Fallback: share CLIP/VAE weights ONLY from a complete sibling matching the same architecture (SD15 vs SDXL)!
    char parent[256];
    snprintf(parent, sizeof(parent), "%s", model_dir);
    char* sl = strrchr(parent, '/');
#ifdef _WIN32
    char* bsl = strrchr(parent, '\\');
    if (bsl && (!sl || bsl > sl)) sl = bsl;
#endif
    if (!sl) return -1;
    *sl = 0;

    if (!want_sd15) {
        snprintf(out, max, "%s/waiNSFWIllustrious_v140/%s", parent, fname);
        if (sdxl_check_pack(out, NULL, NULL) == 0) {
            if (log_share) glog("sharing %s from waiNSFWIllustrious_v140", fname);
            return 0;
        }
    }
    DIR* d = opendir(parent);
    if (!d) return -1;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char probe_g[320];
        snprintf(probe_g, sizeof(probe_g), "%s/%s/clip_g.psd1", parent, ent->d_name);
        int sib_is_sdxl = file_can_open(probe_g);
        if (want_sd15 && sib_is_sdxl) continue;
        if (!want_sd15 && !sib_is_sdxl) continue;
        snprintf(out, max, "%s/%s/%s", parent, ent->d_name, fname);
        if (sdxl_check_pack(out, NULL, NULL) == 0) {
            if (log_share) glog("sharing %s from %s", fname, ent->d_name);
            closedir(d);
            return 0;
        }
    }
    closedir(d);
    return -1;
}

static int validate_pack_or_fail(SdxlEngine* e, const char* path, const char* label) {
    uint64_t act = 0, exp = 0;
    int rc = sdxl_check_pack(path, &act, &exp);
    if (rc == 0) return 0;
    if (rc == -3) {
        fail(e, "incomplete %s (%llu / %llu MiB) - still uploading?",
             path, (unsigned long long)(act >> 20), (unsigned long long)(exp >> 20));
    } else if (rc == -2) {
        fail(e, "corrupt/bad PSD1 header: %s", path);
    } else {
        fail(e, "missing %s: %s", label, path);
    }
    return -1;
}

int sdxl_reload_model(SdxlEngine* e, const char* model_dir,
                      SdxlProgressFn progress, void* user) {
    // Phase 1: Pre-flight check ALL required .psd1 files BEFORE freeing the currently active model!
    char p_unet[320], p_cl[320], p_cg[320] = {0}, p_vae[320];
    if (resolve_pack_path(model_dir, "unet.psd1", p_unet, sizeof(p_unet), 0, 0, 0) != 0) {
        fail(e, "missing unet.psd1 in %s", model_dir);
        return -2;
    }
    if (validate_pack_or_fail(e, p_unet, "UNet") != 0) return -2;

    int next_sd15 = probe_unet_is_sd15(p_unet);
    if (resolve_pack_path(model_dir, "clip_l.psd1", p_cl, sizeof(p_cl), 1, next_sd15, 0) != 0) {
        fail(e, "missing clip_l.psd1 for %s", model_dir);
        return -2;
    }
    if (validate_pack_or_fail(e, p_cl, "CLIP-L") != 0) return -2;

    if (!next_sd15) {
        if (resolve_pack_path(model_dir, "clip_g.psd1", p_cg, sizeof(p_cg), 1, 0, 0) != 0) {
            fail(e, "missing clip_g.psd1 for %s", model_dir);
            return -2;
        }
        if (validate_pack_or_fail(e, p_cg, "OpenCLIP-G") != 0) return -2;
    }
    if (resolve_pack_path(model_dir, "vae.psd1", p_vae, sizeof(p_vae), 1, next_sd15, 0) != 0) {
        fail(e, "missing vae.psd1 for %s", model_dir);
        return -2;
    }
    if (validate_pack_or_fail(e, p_vae, "VAE") != 0) return -2;

    // Phase 2: All files validated complete — swap weights cleanly in GPU memory
    free(e->clip_l.t); memset(&e->clip_l, 0, sizeof(e->clip_l));
    free(e->clip_g.t); memset(&e->clip_g, 0, sizeof(e->clip_g));
    free(e->vae.t);    memset(&e->vae, 0, sizeof(e->vae));
    free(e->unet.t);   memset(&e->unet, 0, sizeof(e->unet));
    e->error[0] = 0;
    e->is_sd15 = 0;
    e->next = e->scratch_bytes;

    const char* slash = strrchr(model_dir, '/');
#ifdef _WIN32
    const char* bsl = strrchr(model_dir, '\\');
    if (bsl && (!slash || bsl > slash)) slash = bsl;
#endif
    snprintf(e->model_name, sizeof(e->model_name), "%s", slash ? slash + 1 : model_dir);
    snprintf(e->model_dir, sizeof(e->model_dir), "%s", model_dir);
    e->applied_lora_spec[0] = 0;

    // Log any shared pack resolutions once during real load
    resolve_pack_path(model_dir, "clip_l.psd1", p_cl, sizeof(p_cl), 1, next_sd15, 1);
    if (!next_sd15) resolve_pack_path(model_dir, "clip_g.psd1", p_cg, sizeof(p_cg), 1, 0, 1);
    resolve_pack_path(model_dir, "vae.psd1", p_vae, sizeof(p_vae), 1, next_sd15, 1);

    uint64_t w0 = e->next;
    double tl0 = now_sec(e);
    if (pack_load(e, &e->unet, p_unet, progress, user, 0.00f, 0.65f, "Loading UNet")) goto load_fail;
    e->is_sd15 = (pack_find(&e->unet, "add_embedding.linear_1.weight") == NULL) ? 1 : 0;

    if (pack_load(e, &e->clip_l, p_cl, progress, user, 0.65f, 0.72f, "Loading CLIP-L")) goto load_fail;
    if (!e->is_sd15) {
        if (pack_load(e, &e->clip_g, p_cg, progress, user, 0.72f, 0.92f, "Loading OpenCLIP-G")) goto load_fail;
    }
    if (pack_load(e, &e->vae, p_vae, progress, user, 0.92f, 1.00f, "Loading VAE")) goto load_fail;
    e->weights_bytes = e->next - w0;
    double tl_sec = now_sec(e) - tl0;
    glog("weights loaded (%s, %s): %u MiB weights + %u MiB scratch = %u / %u MiB GPU in %.2fs (%.0f MiB/s)",
         e->model_name, e->is_sd15 ? "SD1.5" : "SDXL",
         (unsigned)(e->weights_bytes >> 20), (unsigned)(e->scratch_bytes >> 20),
         (unsigned)(e->next >> 20), (unsigned)(e->mem_size >> 20),
         tl_sec, tl_sec > 0.01 ? ((double)(e->weights_bytes >> 20) / tl_sec) : 0.0);
    return 0;

load_fail:
    free(e->clip_l.t); memset(&e->clip_l, 0, sizeof(e->clip_l));
    free(e->clip_g.t); memset(&e->clip_g, 0, sizeof(e->clip_g));
    free(e->vae.t);    memset(&e->vae, 0, sizeof(e->vae));
    free(e->unet.t);   memset(&e->unet, 0, sizeof(e->unet));
    e->next = e->scratch_bytes;
    e->model_name[0] = 0;
    return -1;
}

void sdxl_free(SdxlEngine* e) {
    clip_tok_free(&e->tok);
    free(e->clip_l.t);
    free(e->clip_g.t);
    free(e->unet.t);
    free(e->vae.t);
    free(e->upscaler.t);
#ifdef SDXL_CPU
    free(e->mem);
#else
    if (e->mem && e->mem_size) {
        sceKernelMunmap(e->mem, e->mem_size);
        sceKernelReleaseDirectMemory((off_t)e->mem_phys_off, e->mem_size);
    }
#endif
    memset(e, 0, sizeof(*e));
}

// =================================================================================================
// NATIVE .SAFETENSORS LORA ENGINE (Kohya & Diffusers · UNet + CLIP-L + OpenCLIP-G · PS4 GPU Fusion)
// =================================================================================================

typedef struct {
    char name[192];
    uint8_t dtype;      // 0=F32, 1=F16, 2=BF16, 3=F8_E4M3
    uint8_t ndim;
    uint32_t shape[4];
    uint64_t off_a, off_b;
} StEntry;

static int cmp_st_entry(const void* a, const void* b) {
    return strcmp(((const StEntry*)a)->name, ((const StEntry*)b)->name);
}

static const StEntry* find_st_entry(const StEntry* arr, int n, const char* name) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int c = strcmp(arr[mid].name, name);
        if (c == 0) return &arr[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return NULL;
}

static float fp8_e4m3_to_f32(uint8_t b) {
    uint32_t s = (uint32_t)(b & 0x80u) << 24;
    uint32_t e = (b >> 3) & 0x0Fu;
    uint32_t m = b & 0x07u;
    if (e == 0) {
        if (m == 0) { float z = 0.0f; return s ? -z : z; }
        float v = ldexpf((float)m, -9);
        return s ? -v : v;
    }
    if (e == 15 && m == 7) return 0.0f; // NaN -> 0
    uint32_t f = s | ((e + 120u) << 23) | (m << 20);
    float r; memcpy(&r, &f, 4); return r;
}

static int st_parse_header(const char* json, size_t jlen, StEntry** out_entries, int* out_n) {
    int cap = 512, cnt = 0;
    StEntry* list = (StEntry*)malloc((size_t)cap * sizeof(StEntry));
    if (!list) return -1;

    const char* p = json;
    const char* end = json + jlen;
    while (p < end && *p != '{') p++;
    if (p < end) p++;

    while (p < end) {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ',')) p++;
        if (p >= end || *p == '}') break;
        if (*p != '"') break;
        p++;
        char key[192];
        size_t kl = 0;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) p++;
            if (kl + 1 < sizeof(key)) key[kl++] = *p;
            p++;
        }
        key[kl] = 0;
        if (p < end && *p == '"') p++;
        while (p < end && (*p == ' ' || *p == ':' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
        if (p >= end || *p != '{') break;

        // Find matching '}' for this tensor object (skipping nested strings/brackets)
        const char* obj_start = p;
        int depth = 0, in_str = 0;
        while (p < end) {
            if (in_str) {
                if (*p == '\\' && p + 1 < end) p++;
                else if (*p == '"') in_str = 0;
            } else {
                if (*p == '"') in_str = 1;
                else if (*p == '{') depth++;
                else if (*p == '}') { depth--; if (depth == 0) { p++; break; } }
            }
            p++;
        }
        const char* obj_end = p;
        if (strcmp(key, "__metadata__") == 0) continue;

        StEntry e;
        memset(&e, 0, sizeof(e));
        snprintf(e.name, sizeof(e.name), "%s", key);
        e.dtype = 255;

        // Parse dtype
        const char* dp = strstr(obj_start, "\"dtype\"");
        if (dp && dp < obj_end) {
            dp += 7;
            while (dp < obj_end && *dp != '"') dp++;
            if (dp < obj_end) {
                dp++;
                if (strncmp(dp, "F32", 3) == 0) e.dtype = 0;
                else if (strncmp(dp, "F16", 3) == 0) e.dtype = 1;
                else if (strncmp(dp, "BF16", 4) == 0) e.dtype = 2;
                else if (strncmp(dp, "F8_E4M3", 7) == 0) e.dtype = 3;
                else if (strncmp(dp, "I64", 3) == 0) e.dtype = 4;
                else if (strncmp(dp, "I32", 3) == 0) e.dtype = 5;
            }
        }
        // Parse shape
        const char* sp = strstr(obj_start, "\"shape\"");
        if (sp && sp < obj_end) {
            sp += 7;
            while (sp < obj_end && *sp != '[') sp++;
            if (sp < obj_end) {
                sp++;
                while (sp < obj_end && *sp != ']' && e.ndim < 4) {
                    while (sp < obj_end && (*sp == ' ' || *sp == ',' || *sp == '\n' || *sp == '\r' || *sp == '\t')) sp++;
                    if (sp >= obj_end || *sp == ']') break;
                    e.shape[e.ndim++] = (uint32_t)strtoul(sp, (char**)&sp, 10);
                }
            }
        }
        // Parse data_offsets
        const char* op = strstr(obj_start, "\"data_offsets\"");
        if (op && op < obj_end) {
            op += 14;
            while (op < obj_end && *op != '[') op++;
            if (op < obj_end) {
                op++;
                e.off_a = strtoull(op, (char**)&op, 10);
                while (op < obj_end && (*op == ' ' || *op == ',' || *op == '\n' || *op == '\r' || *op == '\t')) op++;
                e.off_b = strtoull(op, (char**)&op, 10);
            }
        }
        if (e.dtype != 255 && e.off_b >= e.off_a) {
            if (cnt >= cap) {
                cap *= 2;
                StEntry* nl = (StEntry*)realloc(list, (size_t)cap * sizeof(StEntry));
                if (!nl) { free(list); return -1; }
                list = nl;
            }
            list[cnt++] = e;
        }
    }
    if (cnt > 1) qsort(list, (size_t)cnt, sizeof(StEntry), cmp_st_entry);
    *out_entries = list;
    *out_n = cnt;
    return 0;
}

static int st_read_f32(int fd, uint64_t data_base, const StEntry* e, float* dst, size_t expected_elems) {
    uint64_t raw_bytes = e->off_b - e->off_a;
    size_t bpe = (e->dtype == 0 || e->dtype == 5) ? 4 : ((e->dtype == 4) ? 8 : ((e->dtype == 3) ? 1 : 2));
    if (raw_bytes != expected_elems * bpe) return -1;
    if (lseek(fd, (off_t)(data_base + e->off_a), SEEK_SET) < 0) return -1;
    if (e->dtype == 0) {
        return read_exact_fd(fd, dst, raw_bytes);
    }
    uint8_t* tmp = (uint8_t*)malloc((size_t)raw_bytes);
    if (!tmp) return -1;
    if (read_exact_fd(fd, tmp, (size_t)raw_bytes) != 0) { free(tmp); return -1; }
    if (e->dtype == 1) {
        const uint16_t* h = (const uint16_t*)tmp;
        for (size_t i = 0; i < expected_elems; i++) dst[i] = host_fp16_to_f32(h[i]);
    } else if (e->dtype == 2) {
        const uint16_t* bf = (const uint16_t*)tmp;
        for (size_t i = 0; i < expected_elems; i++) {
            uint32_t u = ((uint32_t)bf[i]) << 16;
            float f; memcpy(&f, &u, 4); dst[i] = f;
        }
    } else if (e->dtype == 4) {
        const int64_t* iv = (const int64_t*)tmp;
        for (size_t i = 0; i < expected_elems; i++) dst[i] = (float)iv[i];
    } else if (e->dtype == 5) {
        const int32_t* iv = (const int32_t*)tmp;
        for (size_t i = 0; i < expected_elems; i++) dst[i] = (float)iv[i];
    } else {
        for (size_t i = 0; i < expected_elems; i++) dst[i] = fp8_e4m3_to_f32(tmp[i]);
    }
    free(tmp);
    return 0;
}

static const StEntry* find_lora_pair(const StEntry* st, int n_st, const char* prefix,
                                     const StEntry** out_up, const StEntry** out_alpha) {
    char k_down[192], k_up[192], k_alpha[192];
    static const char* const DOWN_SFX[] = { ".lora_down.weight", ".lora.down.weight", ".lora_A.weight", "_lora.down.weight" };
    static const char* const UP_SFX[]   = { ".lora_up.weight",   ".lora.up.weight",   ".lora_B.weight", "_lora.up.weight" };
    for (int s = 0; s < 4; s++) {
        snprintf(k_down, sizeof(k_down), "%s%s", prefix, DOWN_SFX[s]);
        const StEntry* d = find_st_entry(st, n_st, k_down);
        if (d) {
            snprintf(k_up, sizeof(k_up), "%s%s", prefix, UP_SFX[s]);
            const StEntry* u = find_st_entry(st, n_st, k_up);
            if (u) {
                snprintf(k_alpha, sizeof(k_alpha), "%s.alpha", prefix);
                *out_up = u;
                *out_alpha = find_st_entry(st, n_st, k_alpha);
                return d;
            }
        }
    }
    return NULL;
}

static int fuse_lora_pack(SdxlEngine* e, SdxlPack* pack, int fd, uint64_t data_base,
                          const StEntry* st, int n_st, const char* const* pfx_list, int n_pfx, float scale) {
    if (!pack->t || pack->n == 0) return 0;
    int fused = 0;
    float* gpu_up = (float*)(uintptr_t)e->buf_a;
    float* gpu_dn = (float*)(uintptr_t)e->buf_b;
    size_t max_floats = (size_t)(e->buf_main_bytes / sizeof(float));

    for (uint32_t ti = 0; ti < pack->n; ti++) {
        const SdxlTensor* t = &pack->t[ti];
        if (t->cols == 0 || (t->fmt != 0 && t->fmt != 2)) continue;
        size_t nl = strlen(t->name);
        if (nl <= 7 || strcmp(t->name + nl - 7, ".weight") != 0) continue;

        char under[128], raw_stem[128];
        size_t sl = nl - 7;
        if (sl >= sizeof(under)) continue;
        memcpy(raw_stem, t->name, sl);
        raw_stem[sl] = 0;
        for (size_t i = 0; i < sl; i++) under[i] = (raw_stem[i] == '.') ? '_' : raw_stem[i];
        under[sl] = 0;

        const StEntry *ed = NULL, *eu = NULL, *ea = NULL;
        for (int pi = 0; pi < n_pfx && !ed; pi++) {
            char cand[192];
            int use_dots = (strchr(pfx_list[pi], '.') != NULL);
            snprintf(cand, sizeof(cand), "%s%s", pfx_list[pi], use_dots ? raw_stem : under);
            ed = find_lora_pair(st, n_st, cand, &eu, &ea);
        }
        if (!ed || !eu || ed->ndim < 2 || eu->ndim < 2) continue;

        uint32_t rank = ed->shape[0];
        if (rank == 0 || rank > 512 || eu->shape[1] != rank || eu->shape[0] != t->rows) continue;

        float alpha = (float)rank;
        if (ea) {
            float av = 0.0f;
            if (st_read_f32(fd, data_base, ea, &av, 1) == 0 && av > 0.0f) alpha = av;
        }
        float mult = scale * (alpha / (float)rank);
        if (fabsf(mult) < 1e-7f) continue;

        uint32_t eff_rank = rank;
        size_t dn_elems = (size_t)ed->shape[0] * ed->shape[1];
        for (int d = 2; d < ed->ndim; d++) dn_elems *= ed->shape[d];
        size_t up_elems = (size_t)eu->shape[0] * eu->shape[1];
        for (int d = 2; d < eu->ndim; d++) up_elems *= eu->shape[d];

        int is_conv_down3 = (ed->ndim == 4 && (ed->shape[2] > 1 || ed->shape[3] > 1) &&
                             (eu->ndim == 2 || (eu->shape[2] == 1 && eu->shape[3] == 1)));
        int is_conv_up3   = (eu->ndim == 4 && (eu->shape[2] > 1 || eu->shape[3] > 1) &&
                             (ed->ndim == 2 || (ed->shape[2] == 1 && ed->shape[3] == 1)));

        if (!is_conv_down3 && !is_conv_up3) {
            // Standard Linear [rank, cols] or 1x1 Conv2d [rank, cols, 1, 1]
            if (dn_elems != (size_t)rank * t->cols || up_elems != (size_t)t->rows * rank) continue;
            if (up_elems > max_floats || dn_elems > max_floats) continue;
            if (st_read_f32(fd, data_base, eu, gpu_up, up_elems) != 0) continue;
            if (st_read_f32(fd, data_base, ed, gpu_dn, dn_elems) != 0) continue;
        } else if (is_conv_down3) {
            // Conv2d down: [rank, cin, kh, kw] -> transpose to [rank, kh, kw, cin] matching PSD1 NHWC im2col layout!
            uint32_t cin = ed->shape[1], kh = ed->shape[2], kw = ed->shape[3];
            if ((uint64_t)cin * kh * kw != t->cols || up_elems != (size_t)t->rows * rank) continue;
            if (up_elems > max_floats || dn_elems > max_floats) continue;
            float* raw_dn = (float*)malloc(dn_elems * sizeof(float));
            if (!raw_dn) continue;
            if (st_read_f32(fd, data_base, eu, gpu_up, up_elems) != 0 ||
                st_read_f32(fd, data_base, ed, raw_dn, dn_elems) != 0) {
                free(raw_dn);
                continue;
            }
            size_t hw = (size_t)kh * kw;
            for (uint32_t r = 0; r < rank; r++) {
                const float* sr = raw_dn + (size_t)r * cin * hw;
                float* dr = gpu_dn + (size_t)r * hw * cin;
                for (uint32_t c = 0; c < cin; c++)
                    for (size_t p = 0; p < hw; p++)
                        dr[p * cin + c] = sr[(size_t)c * hw + p];
            }
            free(raw_dn);
        } else {
            // Conv2d up: [rows, rank, kh, kw], down: [rank, cin, 1, 1] -> expand to eff_rank = kh*kw*rank
            uint32_t cin = ed->shape[1], kh = eu->shape[2], kw = eu->shape[3];
            uint32_t hw = kh * kw;
            eff_rank = rank * hw;
            if ((uint64_t)cin * hw != t->cols) continue;
            if ((size_t)t->rows * eff_rank > max_floats || (size_t)eff_rank * t->cols > max_floats) continue;
            float* raw_up = (float*)malloc(up_elems * sizeof(float));
            float* raw_dn = (float*)malloc(dn_elems * sizeof(float));
            if (!raw_up || !raw_dn) { free(raw_up); free(raw_dn); continue; }
            if (st_read_f32(fd, data_base, eu, raw_up, up_elems) != 0 ||
                st_read_f32(fd, data_base, ed, raw_dn, dn_elems) != 0) {
                free(raw_up); free(raw_dn); continue;
            }
            // raw_up is [rows, rank, hw] -> reorder to [rows, hw, rank] = [rows, eff_rank]
            for (uint32_t o = 0; o < t->rows; o++) {
                for (uint32_t p = 0; p < hw; p++)
                    for (uint32_t r = 0; r < rank; r++)
                        gpu_up[(size_t)o * eff_rank + (size_t)p * rank + r] = raw_up[((size_t)o * rank + r) * hw + p];
            }
            memset(gpu_dn, 0, (size_t)eff_rank * t->cols * sizeof(float));
            for (uint32_t p = 0; p < hw; p++) {
                for (uint32_t r = 0; r < rank; r++) {
                    memcpy(gpu_dn + ((size_t)p * rank + r) * t->cols + (size_t)p * cin,
                           raw_dn + (size_t)r * cin, (size_t)cin * sizeof(float));
                }
            }
            free(raw_up);
            free(raw_dn);
        }

        // Synchronously fuse this tensor on PS4 GPU before overwriting buf_a/buf_b with next layer!
        if (t->fmt == 2) {
            op_lora_fuse_q8(e, t_addr(pack, t), t_scales(pack, t), e->buf_a, e->buf_b, t->rows, t->cols, eff_rank, mult);
        } else {
            op_lora_fuse_f32(e, t_addr(pack, t), e->buf_a, e->buf_b, t->rows, t->cols, eff_rank, mult);
        }
        if (gpu_flush(e) != 0) return -1;
        fused++;
    }
    return fused;
}

static int reload_pack_inplace(SdxlEngine* e, SdxlPack* p, const char* path) {
    if (!p->t || !p->gpu_base || !p->data_size) return -1;
#ifdef _WIN32
    int fd = open(path, O_RDONLY | O_BINARY);
#else
    int fd = open(path, O_RDONLY);
#endif
    if (fd < 0) { fail(e, "cannot reopen %s", path); return -1; }
    uint64_t header_end = 24ull + (uint64_t)p->n * sizeof(SdxlTensor);
    uint64_t data_start = (header_end + 255ull) & ~255ull;
    if (lseek(fd, (off_t)data_start, SEEK_SET) < 0) { close(fd); return -1; }
    int rc = read_exact_fd(fd, (void*)(uintptr_t)p->gpu_base, (size_t)p->data_size);
    close(fd);
    return rc;
}

static int apply_single_lora_file(SdxlEngine* e, const char* path, float scale) {
#ifdef _WIN32
    int fd = open(path, O_RDONLY | O_BINARY);
#else
    int fd = open(path, O_RDONLY);
#endif
    if (fd < 0) {
        glog("LoRA skipped (file not installed): %s", path);
        return 1; // Soft-skip missing optional LoRA so presets never break
    }
    uint64_t hlen = 0;
    if (read_exact_fd(fd, &hlen, 8) != 0 || hlen == 0 || hlen > (16ull << 20)) {
        close(fd);
        fail(e, "Invalid .safetensors header in %s", path);
        return -1;
    }
    char* jbuf = (char*)malloc((size_t)hlen + 1);
    if (!jbuf || read_exact_fd(fd, jbuf, (size_t)hlen) != 0) {
        free(jbuf); close(fd);
        fail(e, "Failed reading LoRA header: %s", path);
        return -1;
    }
    jbuf[hlen] = 0;
    StEntry* st = NULL;
    int n_st = 0;
    int prc = st_parse_header(jbuf, (size_t)hlen, &st, &n_st);
    free(jbuf);
    if (prc != 0 || !st || n_st == 0) {
        free(st); close(fd);
        fail(e, "No supported tensors in LoRA: %s", path);
        return -1;
    }

    // Strict architecture compatibility check BEFORE fusing any layer!
    // Prevents SD 1.5 LoRAs from partially fusing into SDXL CLIP-L / early ResNets (and vice versa).
    int has_sdxl_marker = 0, has_sd15_marker = 0;
    for (int i = 0; i < n_st; i++) {
        const char* k = st[i].name;
        if (strstr(k, "lora_te2_") || strstr(k, "text_encoder_2") || strstr(k, "down_blocks_2_attentions")) {
            has_sdxl_marker = 1;
            break;
        }
        if (st[i].ndim >= 2 && (strstr(k, "attn2_to_k.lora_down") || strstr(k, "attn2.to_k.lora_A"))) {
            if (st[i].shape[1] == 2048) has_sdxl_marker = 1;
            else if (st[i].shape[1] == 768) has_sd15_marker = 1;
        }
    }
    if ((e->is_sd15 && has_sdxl_marker) || (!e->is_sd15 && has_sd15_marker && !has_sdxl_marker)) {
        glog("LoRA blocked (incompatible arch %s for active %s model): '%s'",
             has_sdxl_marker ? "SDXL" : "SD1.5", e->is_sd15 ? "SD1.5" : "SDXL", path);
        free(st);
        close(fd);
        return 1; // Safely skip incompatible LoRA without corrupting weights!
    }

    uint64_t data_base = 8ull + hlen;
    double t0 = now_sec(e);
    int prev_eco = e->eco_mode;
    gpu_begin(e);

    static const char* const UNET_PFX[] = { "lora_unet_", "unet.", "unet_" };
    static const char* const CL_SD15[]  = { "lora_te_", "lora_te1_", "text_encoder." };
    static const char* const CL_SDXL[]  = { "lora_te1_", "lora_te_", "text_encoder." };
    static const char* const CG_SDXL[]  = { "lora_te2_", "text_encoder_2." };

    int fu = fuse_lora_pack(e, &e->unet, fd, data_base, st, n_st, UNET_PFX, 3, scale);
    int fl = (fu >= 0) ? fuse_lora_pack(e, &e->clip_l, fd, data_base, st, n_st, e->is_sd15 ? CL_SD15 : CL_SDXL, 3, scale) : -1;
    int fg = (fl >= 0 && !e->is_sd15) ? fuse_lora_pack(e, &e->clip_g, fd, data_base, st, n_st, CG_SDXL, 2, scale) : 0;

    e->eco_mode = prev_eco;
    free(st);
    close(fd);
    if (fu < 0 || fl < 0 || fg < 0) return -1;
    int tot = fu + fl + fg;
    if (tot == 0) {
        glog("LoRA skipped (0 matching %s layers): '%s'", e->is_sd15 ? "SD1.5" : "SDXL", path);
        return 1; // Soft-skip SD1.5 <-> SDXL cross-arch LoRA tags cleanly
    }
    glog("LoRA fused '%s' (scale=%.2f): %d layers (UNet=%d, CLIP-L=%d, CLIP-G=%d) in %.2fs",
         path, (double)scale, tot, fu, fl, fg, now_sec(e) - t0);
    return 0;
}

static void clean_lora_stem(const char* raw, char* out, size_t max) {
    while (*raw == ' ' || *raw == '\t') raw++;
    snprintf(out, max, "%s", raw);
    size_t n = strlen(out);
    while (n > 0 && (out[n - 1] == ' ' || out[n - 1] == '\t')) out[--n] = 0;
    if (n > 12 && strcmp(out + n - 12, ".safetensors") == 0) out[n - 12] = 0;
}

void sdxl_extract_loras(const char* raw_prompt, const char* ui_lora, float ui_strength,
                        char* clean_prompt, size_t clean_max, char* out_spec, size_t spec_max) {
    char names[6][80];
    float weights[6];
    int nl = 0;

    if (ui_lora && ui_lora[0] && fabsf(ui_strength) > 1e-4f) {
        char st[80];
        clean_lora_stem(ui_lora, st, sizeof(st));
        if (st[0] && strcmp(st, "none") != 0 && strcmp(st, "Off") != 0 &&
            !strchr(st, '/') && !strchr(st, '\\') && !strstr(st, "..")) {
            snprintf(names[nl], sizeof(names[0]), "%s", st);
            weights[nl] = ui_strength;
            nl++;
        }
    }

    size_t co = 0;
    const char* p = raw_prompt ? raw_prompt : "";
    while (*p) {
        if ((strncmp(p, "<lora:", 6) == 0 || strncmp(p, "<LORA:", 6) == 0)) {
            const char* gt = strchr(p + 6, '>');
            if (gt && (size_t)(gt - (p + 6)) < 120u) {
                char inner[128];
                size_t il = (size_t)(gt - (p + 6));
                memcpy(inner, p + 6, il);
                inner[il] = 0;
                char* col = strchr(inner, ':');
                float w = 0.80f;
                if (col) {
                    *col = 0;
                    char* col2 = strchr(col + 1, ':');
                    if (col2) *col2 = 0;
                    w = (float)atof(col + 1);
                }
                char st[80];
                clean_lora_stem(inner, st, sizeof(st));
                if (st[0] && fabsf(w) > 1e-4f && !strchr(st, '/') && !strchr(st, '\\') && !strstr(st, "..")) {
                    int dup = -1;
                    for (int i = 0; i < nl; i++) {
                        if (strcmp(names[i], st) == 0) { dup = i; break; }
                    }
                    if (dup >= 0) weights[dup] = w; // Prompt tag overrides UI slider if same LoRA
                    else if (nl < 6) {
                        snprintf(names[nl], sizeof(names[0]), "%s", st);
                        weights[nl] = w;
                        nl++;
                    }
                }
                p = gt + 1;
                while (*p == ' ' || *p == ',') p++;
                continue;
            }
        }
        if (clean_prompt && co + 1 < clean_max) clean_prompt[co++] = *p;
        p++;
    }
    if (clean_prompt && clean_max) {
        while (co > 0 && (clean_prompt[co - 1] == ' ' || clean_prompt[co - 1] == ',')) co--;
        clean_prompt[co] = 0;
    }
    if (out_spec && spec_max) {
        size_t so = 0;
        out_spec[0] = 0;
        for (int i = 0; i < nl && so + 96 < spec_max; i++) {
            so += (size_t)snprintf(out_spec + so, spec_max - so, "%s%s:%.2f", i ? "," : "", names[i], (double)weights[i]);
        }
    }
}

int sdxl_sync_loras(SdxlEngine* e, const char* loras_dir, const char* lora_spec,
                    SdxlProgressFn progress, void* user) {
    const char* want = lora_spec ? lora_spec : "";
    if (strcmp(e->applied_lora_spec, want) == 0) return 0; // Already fused at exact weights!

    // If a previous LoRA was fused into UNet/CLIP weights, restore pristine base weights first
    if (e->applied_lora_spec[0] != 0) {
        if (!e->model_dir[0]) { fail(e, "cannot un-apply LoRA without model_dir"); return -1; }
        if (progress) progress(user, 0.01f, "Restoring base weights for new LoRA...");
        char p_unet[320], p_cl[320], p_cg[320] = {0};
        if (resolve_pack_path(e->model_dir, "unet.psd1", p_unet, sizeof(p_unet), 0, 0, 0) != 0 ||
            resolve_pack_path(e->model_dir, "clip_l.psd1", p_cl, sizeof(p_cl), 1, e->is_sd15, 0) != 0 ||
            (!e->is_sd15 && resolve_pack_path(e->model_dir, "clip_g.psd1", p_cg, sizeof(p_cg), 1, 0, 0) != 0)) {
            fail(e, "missing base model files in %s", e->model_dir);
            return -1;
        }
        if (reload_pack_inplace(e, &e->unet, p_unet) != 0 ||
            reload_pack_inplace(e, &e->clip_l, p_cl) != 0 ||
            (!e->is_sd15 && reload_pack_inplace(e, &e->clip_g, p_cg) != 0)) {
            return -1;
        }
        e->applied_lora_spec[0] = 0;
        glog("restored clean base weights for %s", e->model_name);
    }

    if (!want[0]) return 0;

    // Parse comma-separated "stem:weight" items and fuse each on GPU
    char buf[384], actual_spec[384] = {0};
    size_t ao = 0;
    snprintf(buf, sizeof(buf), "%s", want);
    char* cur = buf;
    while (*cur) {
        char* comma = strchr(cur, ',');
        if (comma) *comma = 0;
        char* col = strchr(cur, ':');
        float w = 0.80f;
        if (col) { *col = 0; w = (float)atof(col + 1); }
        if (cur[0] && fabsf(w) > 1e-4f) {
            char msg[160], fpath[360];
            snprintf(msg, sizeof(msg), "Fusing LoRA %s (%.2f)...", cur, (double)w);
            if (progress) progress(user, 0.02f, msg);
            snprintf(fpath, sizeof(fpath), "%s/%s.safetensors", (loras_dir && loras_dir[0]) ? loras_dir : "/data/garlic-diffusion/loras", cur);
            if (!file_can_open(fpath)) {
                // Case-insensitive filename match inside loras_dir so `<lora:Hifumi:0.8>` matches cleanly!
                const char* ldir = (loras_dir && loras_dir[0]) ? loras_dir : "/data/garlic-diffusion/loras";
                DIR* d = opendir(ldir);
                if (d) {
                    struct dirent* ent;
                    while ((ent = readdir(d)) != NULL) {
                        size_t fl = strlen(ent->d_name);
                        if (fl <= 12 || strcmp(ent->d_name + fl - 12, ".safetensors") != 0) continue;
                        char stem[80];
                        clean_lora_stem(ent->d_name, stem, sizeof(stem));
                        size_t sl = strlen(stem), ql = strlen(cur);
                        int match = (sl == ql);
                        for (size_t k = 0; match && k < sl; k++) {
                            char a = stem[k], b = cur[k];
                            if (a >= 'A' && a <= 'Z') a += 32;
                            if (b >= 'A' && b <= 'Z') b += 32;
                            if (a != b) match = 0;
                        }
                        if (match) {
                            snprintf(fpath, sizeof(fpath), "%s/%s", ldir, ent->d_name);
                            break;
                        }
                    }
                    closedir(d);
                }
            }
            int arc = apply_single_lora_file(e, fpath, w);
            if (arc < 0) {
                // Restore clean weights if a stacked LoRA failed mid-way
                e->applied_lora_spec[0] = '!'; e->applied_lora_spec[1] = 0;
                char err_keep[256];
                snprintf(err_keep, sizeof(err_keep), "%s", e->error);
                sdxl_sync_loras(e, loras_dir, "", NULL, NULL);
                snprintf(e->error, sizeof(e->error), "%s", err_keep);
                return -1;
            }
            if (arc == 0 && ao + 96 < sizeof(actual_spec)) {
                ao += (size_t)snprintf(actual_spec + ao, sizeof(actual_spec) - ao, "%s%s:%.2f", ao ? "," : "", cur, (double)w);
            }
        }
        if (!comma) break;
        cur = comma + 1;
    }
    snprintf(e->applied_lora_spec, sizeof(e->applied_lora_spec), "%s", actual_spec);
    return 0;
}

// ---- Prompt Deduplicator, 1050-Word Wildcard Randomizer & Anti-Deformity Shield ----------------
static int ci_has(const char* hay, const char* needle) {
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

static uint64_t splitmix64_next(uint64_t* st) {
    uint64_t z = (*st += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void canonical_tag_key(const char* raw, char* key, size_t max) {
    while (*raw == ' ' || *raw == '\t' || *raw == '\r' || *raw == '\n') raw++;
    size_t len = strlen(raw);
    while (len > 0 && (raw[len - 1] == ' ' || raw[len - 1] == '\t' || raw[len - 1] == '\r' || raw[len - 1] == '\n')) len--;

    // Canonicalize <lora:stem:weight> by stem so duplicate LoRA tags collapse cleanly
    if (len > 7 && (strncmp(raw, "<lora:", 6) == 0 || strncmp(raw, "<LORA:", 6) == 0) && raw[len - 1] == '>') {
        size_t k = 0;
        const char* s = raw + 6;
        const char* e = raw + len - 1;
        key[k++] = '<';
        while (s < e && *s != ':' && k + 2 < max) {
            char c = *s++;
            if (c >= 'A' && c <= 'Z') c += 32;
            key[k++] = c;
        }
        if (k > 13 && strncmp(key + k - 12, ".safetensors", 12) == 0) k -= 12;
        key[k++] = '>';
        key[k] = 0;
        return;
    }

    // Strip outer (tag:1.2) wrapper so "(bad hands:1.3)" and "bad hands" share canonical key "bad hands"
    if (len >= 2 && raw[0] == '(' && raw[len - 1] == ')') {
        raw++;
        len -= 2;
        const char* col = NULL;
        for (size_t i = len; i > 0; i--) {
            if (raw[i - 1] == ':') { col = raw + i - 1; break; }
            if (!((raw[i - 1] >= '0' && raw[i - 1] <= '9') || raw[i - 1] == '.' || raw[i - 1] == '-' || raw[i - 1] == ' ')) break;
        }
        if (col && col > raw) len = (size_t)(col - raw);
    }

    size_t k = 0;
    int prev_sp = 1;
    for (size_t i = 0; i < len && k + 1 < max; i++) {
        char c = raw[i];
        if (c == ' ' || c == '\t' || c == '_') {
            if (!prev_sp) { key[k++] = ' '; prev_sp = 1; }
        } else {
            if (c >= 'A' && c <= 'Z') c += 32;
            key[k++] = c;
            prev_sp = 0;
        }
    }
    while (k > 0 && key[k - 1] == ' ') k--;
    key[k] = 0;
}

void sdxl_clean_prompt_dupes(const char* src, char* dst, size_t dst_max) {
    if (!dst || !dst_max) return;
    dst[0] = 0;
    if (!src || !src[0]) return;

    char keys[192][80];
    char toks[192][160];
    int n_toks = 0;

    // Pre-split any adjacent "<lora:...>" glued without commas into separate logical items
    char norm[2048];
    size_t no = 0;
    for (const char* p = src; *p && no + 3 < sizeof(norm); p++) {
        if ((strncmp(p, "<lora:", 6) == 0 || strncmp(p, "<LORA:", 6) == 0) && no > 0 && norm[no - 1] != ',') {
            norm[no++] = ',';
        }
        norm[no++] = *p;
        if (*p == '>' && p[1] && p[1] != ',') {
            norm[no++] = ',';
        }
    }
    norm[no] = 0;

    const char* p = norm;
    while (*p && n_toks < 192) {
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == '\r' || *p == '\n') p++;
        if (!*p) break;
        int paren = 0, angle = 0;
        const char* st = p;
        while (*p) {
            if (*p == '(') paren++;
            else if (*p == ')' && paren > 0) paren--;
            else if (*p == '<') angle++;
            else if (*p == '>' && angle > 0) angle--;
            else if (*p == ',' && paren == 0 && angle == 0) break;
            p++;
        }
        size_t tl = (size_t)(p - st);
        while (tl > 0 && (st[tl - 1] == ' ' || st[tl - 1] == '\t' || st[tl - 1] == '\r' || st[tl - 1] == '\n')) tl--;
        if (tl == 0) continue;
        if (tl >= sizeof(toks[0])) tl = sizeof(toks[0]) - 1;
        char cand[160], ckey[80];
        memcpy(cand, st, tl);
        cand[tl] = 0;
        canonical_tag_key(cand, ckey, sizeof(ckey));
        if (!ckey[0]) continue;

        int dup = -1;
        for (int i = 0; i < n_toks; i++) {
            if (strcmp(keys[i], ckey) == 0) { dup = i; break; }
        }
        if (dup >= 0) {
            // If new token has explicit weight "(...:1.3)" and old didn't, upgrade in place
            if (cand[0] == '(' && toks[dup][0] != '(') {
                snprintf(toks[dup], sizeof(toks[0]), "%s", cand);
            }
            continue;
        }
        snprintf(keys[n_toks], sizeof(keys[0]), "%s", ckey);
        snprintf(toks[n_toks], sizeof(toks[0]), "%s", cand);
        n_toks++;
    }

    size_t out_pos = 0;
    for (int i = 0; i < n_toks && out_pos + 4 < dst_max; i++) {
        size_t need = strlen(toks[i]) + (i ? 2u : 0u);
        if (out_pos + need + 1 >= dst_max) break;
        if (i) { dst[out_pos++] = ','; dst[out_pos++] = ' '; }
        memcpy(dst + out_pos, toks[i], strlen(toks[i]));
        out_pos += strlen(toks[i]);
    }
    dst[out_pos] = 0;
}

static const char* pick_pool(const char* const* arr, int n, uint64_t* rng) {
    if (!arr || n <= 0) return "";
    return arr[(size_t)(splitmix64_next(rng) % (uint64_t)n)];
}

static const char* pick_salted_pool(const char* const* arr, int n, uint64_t base_rng, uint64_t salt) {
    uint64_t sr = base_rng ^ (salt * 0x9E3779B97F4A7C15ull);
    (void)splitmix64_next(&sr);
    return pick_pool(arr, n, &sr);
}

static const char* pick_sfw_any(uint64_t* rng) {
    int tot = RND_POSE_N + RND_CAMERA_N + RND_EXPR_N + RND_SCENE_N + RND_LIGHT_N + RND_OUTFIT_N + RND_DETAIL_N;
    int idx = (int)(splitmix64_next(rng) % (uint64_t)tot);
    if (idx < RND_POSE_N) return RND_POSE[idx];
    idx -= RND_POSE_N;
    if (idx < RND_CAMERA_N) return RND_CAMERA[idx];
    idx -= RND_CAMERA_N;
    if (idx < RND_EXPR_N) return RND_EXPR[idx];
    idx -= RND_EXPR_N;
    if (idx < RND_SCENE_N) return RND_SCENE[idx];
    idx -= RND_SCENE_N;
    if (idx < RND_LIGHT_N) return RND_LIGHT[idx];
    idx -= RND_LIGHT_N;
    if (idx < RND_OUTFIT_N) return RND_OUTFIT[idx];
    idx -= RND_OUTFIT_N;
    return RND_DETAIL[idx];
}

static const char* pick_ecchi_any(uint64_t* rng, int allow_explicit) {
    int pose_n = allow_explicit ? RND_ECCHI_POSE_N : (RND_ECCHI_POSE_N - 10);
    int expr_n = allow_explicit ? RND_ECCHI_EXPR_N : (RND_ECCHI_EXPR_N - 18);
    int outf_n = allow_explicit ? RND_ECCHI_OUTFIT_N : (RND_ECCHI_OUTFIT_N - 30);
    int acc_n  = allow_explicit ? RND_ECCHI_ACCENT_N : (RND_ECCHI_ACCENT_N - 15);
    int tot = pose_n + expr_n + outf_n + RND_ECCHI_SCENE_N + acc_n;
    int idx = (int)(splitmix64_next(rng) % (uint64_t)tot);
    if (idx < pose_n) return RND_ECCHI_POSE[idx];
    idx -= pose_n;
    if (idx < expr_n) return RND_ECCHI_EXPR[idx];
    idx -= expr_n;
    if (idx < outf_n) return RND_ECCHI_OUTFIT[idx];
    idx -= outf_n;
    if (idx < RND_ECCHI_SCENE_N) return RND_ECCHI_SCENE[idx];
    idx -= RND_ECCHI_SCENE_N;
    return RND_ECCHI_ACCENT[idx];
}

void sdxl_expand_wildcards(const char* src, uint64_t seed, int nsfw_allow, uint32_t rand_mask,
                           char* dst, size_t dst_max) {
    if (!dst || !dst_max) return;
    gd_init_randomizer_pools();
    dst[0] = 0;
    uint64_t rng = seed ? seed : 0xCAFEBABE12345678ull;
    (void)splitmix64_next(&rng);

    char work[2048];
    snprintf(work, sizeof(work), "%s", src ? src : "");

    // Pass 1: Expand inline {choiceA|choiceB|choiceC} groups (supports nesting up to 4 levels)
    for (int pass = 0; pass < 4; pass++) {
        char* lb = NULL;
        char* rb = NULL;
        for (char* p = work; *p; p++) {
            if (*p == '{') lb = p;
            else if (*p == '}' && lb) { rb = p; break; }
        }
        if (!lb || !rb) break;
        const char* opts[64];
        size_t lens[64];
        int n_opt = 0;
        const char* cur = lb + 1;
        for (const char* q = cur; q <= rb && n_opt < 64; q++) {
            if (*q == '|' || q == rb) {
                opts[n_opt] = cur;
                lens[n_opt] = (size_t)(q - cur);
                n_opt++;
                cur = q + 1;
            }
        }
        int pick = n_opt > 0 ? (int)(splitmix64_next(&rng) % (uint64_t)n_opt) : 0;
        char next_buf[2048];
        size_t pre = (size_t)(lb - work);
        size_t mid = n_opt > 0 ? lens[pick] : 0;
        snprintf(next_buf, sizeof(next_buf), "%.*s%.*s%s",
                 (int)pre, work, (int)mid, n_opt > 0 ? opts[pick] : "", rb + 1);
        snprintf(work, sizeof(work), "%s", next_buf);
    }

    // Pass 2: Expand __pool__ wildcard tokens inside prompt text
    struct { const char* tok; int id; } map[] = {
        { "__pose__", 1 }, { "__camera__", 2 }, { "__expr__", 3 }, { "__scene__", 4 },
        { "__light__", 5 }, { "__outfit__", 6 }, { "__detail__", 7 }, { "__random__", 8 },
        { "__ecchi__", 9 }, { "__ecchi_pose__", 10 }, { "__ecchi_expr__", 11 },
        { "__ecchi_outfit__", 12 }, { "__ecchi_scene__", 13 }, { "__ecchi_accent__", 14 },
        { "__explicit__", 15 }
    };
    for (int guard = 0; guard < 24; guard++) {
        int hit = 0;
        for (size_t m = 0; m < sizeof(map) / sizeof(map[0]); m++) {
            char* pos = strstr(work, map[m].tok);
            if (!pos) continue;
            hit = 1;
            const char* rep = "";
            int allow_exp = nsfw_allow && ((rand_mask & RAND_EXPLICIT) != 0 || map[m].id == 15);
            uint64_t wrng = rng ^ ((uint64_t)(map[m].id * 131u + guard * 17u) * 0x9E3779B97F4A7C15ull);
            (void)splitmix64_next(&wrng);
            (void)splitmix64_next(&rng);
            switch (map[m].id) {
            case 1: rep = pick_pool(RND_POSE, RND_POSE_N, &wrng); break;
            case 2: rep = pick_pool(RND_CAMERA, RND_CAMERA_N, &wrng); break;
            case 3: rep = pick_pool(RND_EXPR, RND_EXPR_N, &wrng); break;
            case 4: rep = pick_pool(RND_SCENE, RND_SCENE_N, &wrng); break;
            case 5: rep = pick_pool(RND_LIGHT, RND_LIGHT_N, &wrng); break;
            case 6: rep = pick_pool(RND_OUTFIT, RND_OUTFIT_N, &wrng); break;
            case 7: rep = pick_pool(RND_DETAIL, RND_DETAIL_N, &wrng); break;
            case 8: rep = pick_sfw_any(&wrng); break;
            case 9: rep = nsfw_allow ? pick_ecchi_any(&wrng, allow_exp) : pick_sfw_any(&wrng); break;
            case 10: rep = nsfw_allow ? pick_pool(RND_ECCHI_POSE, allow_exp ? RND_ECCHI_POSE_N : (RND_ECCHI_POSE_N - 10), &wrng) : pick_pool(RND_POSE, RND_POSE_N, &wrng); break;
            case 11: rep = nsfw_allow ? pick_pool(RND_ECCHI_EXPR, allow_exp ? RND_ECCHI_EXPR_N : (RND_ECCHI_EXPR_N - 18), &wrng) : pick_pool(RND_EXPR, RND_EXPR_N, &wrng); break;
            case 12: rep = nsfw_allow ? pick_pool(RND_ECCHI_OUTFIT, allow_exp ? RND_ECCHI_OUTFIT_N : (RND_ECCHI_OUTFIT_N - 30), &wrng) : pick_pool(RND_OUTFIT, RND_OUTFIT_N, &wrng); break;
            case 13: rep = nsfw_allow ? pick_pool(RND_ECCHI_SCENE, RND_ECCHI_SCENE_N, &wrng) : pick_pool(RND_SCENE, RND_SCENE_N, &wrng); break;
            case 14: rep = nsfw_allow ? pick_pool(RND_ECCHI_ACCENT, allow_exp ? RND_ECCHI_ACCENT_N : (RND_ECCHI_ACCENT_N - 15), &wrng) : pick_pool(RND_DETAIL, RND_DETAIL_N, &wrng); break;
            case 15: rep = nsfw_allow ? pick_pool(RND_ECCHI_OUTFIT + (RND_ECCHI_OUTFIT_N - 30), 30, &wrng) : ""; break;
            }
            char next_buf[2048];
            size_t pre = (size_t)(pos - work);
            size_t tl = strlen(map[m].tok);
            snprintf(next_buf, sizeof(next_buf), "%.*s%s%s", (int)pre, work, rep, pos + tl);
            snprintf(work, sizeof(work), "%s", next_buf);
            break;
        }
        if (!hit) break;
    }

    // Pass 3: Append 1-click Batch Randomizer checkboxes (rand_mask) — each pool uses an independent salted stream!
    size_t wl = strlen(work);
    #define APPEND_RND(s) do { \
        const char* _s = (s); \
        if (_s && _s[0] && wl + strlen(_s) + 4 < sizeof(work)) { \
            wl += (size_t)snprintf(work + wl, sizeof(work) - wl, "%s%s", wl ? ", " : "", _s); \
        } \
    } while (0)
    #define PICK_SALTED(arr, cnt, salt) pick_salted_pool((arr), (cnt), rng, (uint64_t)(salt))

    if (rand_mask & RAND_CAMERA) APPEND_RND(PICK_SALTED(RND_CAMERA, RND_CAMERA_N, 0x101u));
    if (rand_mask & RAND_POSE)   APPEND_RND(PICK_SALTED(RND_POSE,   RND_POSE_N,   0x202u));
    if (rand_mask & RAND_EXPR)   APPEND_RND(PICK_SALTED(RND_EXPR,   RND_EXPR_N,   0x303u));
    if (rand_mask & RAND_OUTFIT) APPEND_RND(PICK_SALTED(RND_OUTFIT, RND_OUTFIT_N, 0x404u));
    if (rand_mask & RAND_SCENE)  APPEND_RND(PICK_SALTED(RND_SCENE,  RND_SCENE_N,  0x505u));
    if (rand_mask & RAND_LIGHT)  APPEND_RND(PICK_SALTED(RND_LIGHT,  RND_LIGHT_N,  0x606u));
    if (rand_mask & RAND_DETAIL) APPEND_RND(PICK_SALTED(RND_DETAIL, RND_DETAIL_N, 0x707u));

    if (nsfw_allow && (rand_mask & RAND_ECCHI)) {
        int exp_ok = (rand_mask & RAND_EXPLICIT) != 0;
        uint32_t sfw_sub = rand_mask & (RAND_POSE | RAND_EXPR | RAND_OUTFIT | RAND_SCENE);
        if (!sfw_sub) {
            APPEND_RND(PICK_SALTED(RND_ECCHI_POSE, exp_ok ? RND_ECCHI_POSE_N : (RND_ECCHI_POSE_N - 10), 0x801u));
            APPEND_RND(PICK_SALTED(RND_ECCHI_EXPR, exp_ok ? RND_ECCHI_EXPR_N : (RND_ECCHI_EXPR_N - 18), 0x802u));
            APPEND_RND(PICK_SALTED(RND_ECCHI_OUTFIT, exp_ok ? RND_ECCHI_OUTFIT_N : (RND_ECCHI_OUTFIT_N - 30), 0x803u));
            APPEND_RND(PICK_SALTED(RND_ECCHI_SCENE, RND_ECCHI_SCENE_N, 0x804u));
        } else {
            uint64_t er = rng ^ 0x8055AA3399CC1177ull;
            (void)splitmix64_next(&er);
            APPEND_RND(pick_ecchi_any(&er, exp_ok));
        }
        APPEND_RND(PICK_SALTED(RND_ECCHI_ACCENT, exp_ok ? RND_ECCHI_ACCENT_N : (RND_ECCHI_ACCENT_N - 15), 0x806u));
    }
    if (nsfw_allow && (rand_mask & RAND_EXPLICIT)) {
        APPEND_RND(PICK_SALTED(RND_ECCHI_OUTFIT + (RND_ECCHI_OUTFIT_N - 30), 30, 0x901u));
        APPEND_RND(PICK_SALTED(RND_ECCHI_ACCENT + (RND_ECCHI_ACCENT_N - 15), 15, 0x902u));
    }
    #undef PICK_SALTED
    #undef APPEND_RND

    sdxl_clean_prompt_dupes(work, dst, dst_max);
}

void sdxl_apply_quality_shield(const char* model_name, int is_sd15, int auto_quality, int nsfw_allow,
                               char* prompt, size_t pr_max, char* negative, size_t ng_max) {
    const char* m = model_name ? model_name : "";
    char pbuf[2048] = {0}, nbuf[1024] = {0};
    snprintf(pbuf, sizeof(pbuf), "%s", prompt ? prompt : "");
    snprintf(nbuf, sizeof(nbuf), "%s", negative ? negative : "");

    if (auto_quality) {
        int has_q = ci_has(pbuf, "masterpiece") || ci_has(pbuf, "best quality") || ci_has(pbuf, "high score");
        if (!has_q && pbuf[0]) {
            char merged[2048];
            if (!is_sd15 && ci_has(m, "animagine")) {
                // Animagine XL 4.0 expects character + framing FIRST and quality tags at the END
                snprintf(merged, sizeof(merged), "%s, masterpiece, high score, great score, absurdres", pbuf);
            } else if (!is_sd15 && ci_has(m, "noob")) {
                snprintf(merged, sizeof(merged), "masterpiece, best quality, newest, absurdres, highres, %s", pbuf);
            } else if (!is_sd15) {
                snprintf(merged, sizeof(merged), "masterpiece, best quality, amazing quality, very aesthetic, absurdres, %s", pbuf);
            } else if (ci_has(m, "dreamshaper") && (ci_has(pbuf, "1girl") || ci_has(pbuf, "1boy") || ci_has(pbuf, "miku"))) {
                snprintf(merged, sizeof(merged), "masterpiece, best quality, 2d anime illustration, cel shaded anime, %s", pbuf);
            } else {
                snprintf(merged, sizeof(merged), "masterpiece, best quality, highres, %s", pbuf);
            }
            snprintf(pbuf, sizeof(pbuf), "%s", merged);
        }
        // Ensure anti-deformity hand/finger/eye negative shield
        const char* shield = is_sd15
            ? "(worst quality, low quality:1.4), lowres, bad anatomy, bad hands, extra digits, fewer digits, fused fingers, bad eyes, asymmetrical eyes, blurry, watermark"
            : (ci_has(m, "animagine")
                ? "lowres, bad anatomy, bad hands, extra digits, fewer digits, missing fingers, bad eyes, worst quality, low quality, low score, bad score, average score, blurry"
                : (ci_has(m, "noob")
                    ? "worst quality, old, early, low quality, lowres, bad anatomy, bad hands, extra digits, fewer digits, fused fingers, bad eyes, asymmetrical eyes, blurry"
                    : "lowres, bad anatomy, bad hands, extra digits, fewer digits, fused fingers, bad feet, bad eyes, asymmetrical eyes, blurry, worst quality, low quality"));
        if (!ci_has(nbuf, "bad hands") || !ci_has(nbuf, "extra digits")) {
            char mn[1024];
            snprintf(mn, sizeof(mn), "%s%s%s", nbuf, nbuf[0] ? ", " : "", shield);
            snprintf(nbuf, sizeof(nbuf), "%s", mn);
        }
    }

    if (!nsfw_allow) {
        char mn[1024];
        snprintf(mn, sizeof(mn), "%s%s%s", nbuf, nbuf[0] ? ", " : "",
                 "nsfw, rating: questionable, rating: explicit, nude, naked, nipples, explicit");
        snprintf(nbuf, sizeof(nbuf), "%s", mn);
    }

    if (prompt && pr_max) sdxl_clean_prompt_dupes(pbuf, prompt, pr_max);
    if (negative && ng_max) sdxl_clean_prompt_dupes(nbuf, negative, ng_max);
}

void sdxl_load_speed_preset(const char* model_name, int is_sd15, ModelPreset* out) {
    memset(out, 0, sizeof(*out));
    const char* m = model_name ? model_name : "";
    if (is_sd15) {
        out->width = 512; out->height = 768; out->upscale = 0;
        if (ci_has(m, "8step") || ci_has(m, "hyper")) {
            out->steps = 6; out->cfg = 3.2f; out->sampler = 1; // Euler (~18s!)
            snprintf(out->description, sizeof(out->description), "Speed Mode · SD1.5 6-Step Euler (~18s)");
        } else {
            out->steps = 12; out->cfg = 6.0f; out->sampler = 3; // DPM++ 2M Karras (~32s!)
            snprintf(out->description, sizeof(out->description), "Speed Mode · SD1.5 12-Step DPM++ (~32s)");
        }
    } else {
        out->width = 768; out->height = 1024; out->upscale = 0;
        if (ci_has(m, "dmd")) {
            if (ci_has(m, "animagine")) {
                out->steps = 4; out->cfg = 1.0f; out->sampler = 2; // Animagine XL 4.0 DMD2 Speed (~31s!)
                snprintf(out->description, sizeof(out->description), "Speed Mode · Animagine DMD2 4-Step CFG 1.0 (~31s)");
            } else {
                out->steps = 5; out->cfg = 1.0f; out->sampler = 2; // LCM single-pass (~38s!)
                snprintf(out->description, sizeof(out->description), "Speed Mode · DMD2 5-Step CFG 1.0 (~38s)");
            }
        } else if (ci_has(m, "8step") || ci_has(m, "hyper")) {
            out->steps = 6; out->cfg = 3.0f; out->sampler = 1; // Euler
            snprintf(out->description, sizeof(out->description), "Speed Mode · Hyper-SDXL 6-Step Euler");
        } else {
            out->steps = 12; out->cfg = 5.5f; out->sampler = 3; // DPM++ 2M Karras
            snprintf(out->description, sizeof(out->description), "Speed Mode · SDXL 12-Step DPM++ Karras");
        }
    }
}

void sdxl_load_preset(const char* model_dir, const char* model_name, int is_sd15, ModelPreset* out) {
    memset(out, 0, sizeof(*out));
    // Authoritative Anti-Deformity Best-Quality Defaults per Architecture & Checkpoint Family
    if (is_sd15) {
        out->width = 512; out->height = 768;
        out->upscale = 1; // Auto-upscale SD 1.5 -> crisp 1080p!
        snprintf(out->prompt, sizeof(out->prompt),
                 "masterpiece, best quality, highres, 1girl, solo, silver hair, blue eyes, glowing sakura petals, night sky, anime style");
        snprintf(out->negative, sizeof(out->negative),
                 "(worst quality, low quality:1.4), lowres, bad anatomy, bad hands, extra digits, fewer digits, fused fingers, bad eyes, blurry, text, watermark");
        if (ci_has(model_name, "8step") || ci_has(model_name, "hyper")) {
            out->steps = 8; out->cfg = 3.5f; out->sampler = 1; // Euler
            snprintf(out->description, sizeof(out->description), "SD 1.5 Fast 8-step (Euler · CFG 3.5 · +1080p)");
        } else if (ci_has(model_name, "counterfeit")) {
            out->steps = 20; out->cfg = 6.5f; out->sampler = 3; // DPM++ 2M Karras
            snprintf(out->description, sizeof(out->description), "SD 1.5 Counterfeit V2.5 · DPM++ 2M Karras · CFG 6.5");
        } else if (ci_has(model_name, "ghostmix")) {
            out->steps = 20; out->cfg = 6.5f; out->sampler = 3; // DPM++ 2M Karras
            snprintf(out->description, sizeof(out->description), "SD 1.5 GhostMix v2.0 · DPM++ 2M Karras · CFG 6.5");
        } else if (ci_has(model_name, "dreamshaper")) {
            out->steps = 20; out->cfg = 6.5f; out->sampler = 3; // DPM++ 2M Karras
            snprintf(out->description, sizeof(out->description), "SD 1.5 DreamShaper 8 · DPM++ 2M Karras · CFG 6.5");
        } else if (ci_has(model_name, "anything")) {
            out->steps = 20; out->cfg = 7.0f; out->sampler = 3; // DPM++ 2M Karras
            snprintf(out->description, sizeof(out->description), "SD 1.5 Anything V5 · DPM++ 2M Karras · CFG 7.0");
        } else {
            out->steps = 20; out->cfg = 6.5f; out->sampler = 3; // DPM++ 2M Karras (MeinaMix V11 etc.)
            snprintf(out->description, sizeof(out->description), "SD 1.5 MeinaMix V11 · DPM++ 2M Karras · CFG 6.5");
        }
    } else {
        // Native 1024x1344 portrait for maximum SDXL / Illustrious eye, finger & lineart detail on PS4 Pro Neo
        // (automatically clamped to 768x1024 on Base PS4 by apply_model_preset)
        int is_illus = ci_has(model_name, "illustrious") || ci_has(model_name, "wai") ||
                       ci_has(model_name, "hassaku") || ci_has(model_name, "nova") ||
                       ci_has(model_name, "pencil") || ci_has(model_name, "walnut") ||
                       ci_has(model_name, "plant") || ci_has(model_name, "noob") ||
                       ci_has(model_name, "animagine");
        out->width = 1024; out->height = 1344;
        out->upscale = 0;
        if (ci_has(model_name, "animagine")) {
            snprintf(out->prompt, sizeof(out->prompt),
                     "1girl, solo, upper body, silver hair, blue eyes, gentle smile, sakura petals, night sky, masterpiece, high score, great score, absurdres");
            snprintf(out->negative, sizeof(out->negative),
                     "lowres, bad anatomy, bad hands, extra digits, fewer digits, missing fingers, bad eyes, worst quality, low quality, low score, bad score, average score, blurry");
        } else if (ci_has(model_name, "noob")) {
            snprintf(out->prompt, sizeof(out->prompt),
                     "masterpiece, best quality, newest, absurdres, highres, 1girl, solo, upper body, silver hair, blue eyes, gentle smile, sakura petals, night sky");
            snprintf(out->negative, sizeof(out->negative),
                     "worst quality, old, early, low quality, lowres, bad anatomy, bad hands, extra digits, fewer digits, fused fingers, bad eyes, asymmetrical eyes, blurry");
        } else {
            snprintf(out->prompt, sizeof(out->prompt),
                     "masterpiece, best quality, amazing quality, very aesthetic, absurdres, 1girl, solo, upper body, silver hair, blue eyes, gentle smile, sakura petals, night sky");
            snprintf(out->negative, sizeof(out->negative),
                     "lowres, bad anatomy, bad hands, extra digits, fewer digits, fused fingers, bad feet, bad eyes, asymmetrical eyes, blurry, worst quality, low quality");
        }
        if (ci_has(model_name, "dmd")) {
            if (ci_has(model_name, "animagine")) {
                out->steps = 5; out->cfg = 1.1f; out->sampler = 2; // Animagine XL 4.0 DMD2: low overshoot sweet spot
                snprintf(out->description, sizeof(out->description),
                         "Animagine XL 4.0 DMD2 · 1024x1344 Best Quality (LCM · 5 stp · CFG 1.1)");
            } else if (ci_has(model_name, "noob")) {
                out->steps = 6; out->cfg = 1.2f; out->sampler = 2; // NoobAI-XL 1.1 DMD2: crisp contrast without eye burn
                snprintf(out->description, sizeof(out->description),
                         "NoobAI-XL 1.1 DMD2 · 1024x1344 Best Quality (LCM · 6 stp · CFG 1.2)");
            } else {
                out->steps = 6; out->cfg = 1.4f; out->sampler = 2; // LCM + CFG 1.4 activates negative hand/eye guard!
                snprintf(out->description, sizeof(out->description),
                         is_illus ? "Illustrious DMD2 · 1024x1344 Best Quality (LCM · 6 stp · CFG 1.4)"
                                  : "SDXL DMD2 · 1024x1344 Best Quality (LCM · 6 stp · CFG 1.4)");
            }
        } else if (ci_has(model_name, "8step") || ci_has(model_name, "hyper")) {
            out->steps = 8; out->cfg = 3.5f; out->sampler = 1; // Euler
            snprintf(out->description, sizeof(out->description), "SDXL 8-step Hyper-SD · 1024x1344 (Euler · CFG 3.5)");
        } else if (ci_has(model_name, "lightning") || ci_has(model_name, "turbo")) {
            out->steps = 6; out->cfg = 2.0f; out->sampler = 1; // Euler
            snprintf(out->description, sizeof(out->description), "SDXL Native Fast · 1024x1344 (Euler · 6 steps)");
        } else if (ci_has(model_name, "animagine")) {
            out->steps = 20; out->cfg = 5.5f; out->sampler = 3;
            snprintf(out->description, sizeof(out->description), "SDXL Animagine XL 4.0 · DPM++ 2M Karras · CFG 5.5");
        } else if (ci_has(model_name, "noob")) {
            out->steps = 20; out->cfg = 5.5f; out->sampler = 3;
            snprintf(out->description, sizeof(out->description), "SDXL NoobAI-XL 1.1 · DPM++ 2M Karras · CFG 5.5");
        } else {
            out->steps = 20; out->cfg = 6.0f; out->sampler = 3; // DPM++ 2M Karras smooth 2nd-order
            snprintf(out->description, sizeof(out->description), "SDXL Full 20-Step · DPM++ 2M Karras · CFG 6.0");
        }
    }

    // Only allow <model_dir>/preset.cfg to override optimal settings if the user explicitly saved custom=1!
    if (!model_dir || !model_dir[0]) return;
    char cfg_path[320];
    snprintf(cfg_path, sizeof(cfg_path), "%s/preset.cfg", model_dir);
#ifdef _WIN32
    int fd = open(cfg_path, O_RDONLY | O_BINARY);
#else
    int fd = open(cfg_path, O_RDONLY);
#endif
    if (fd < 0) return;
    char buf[2048];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return;
    buf[n] = 0;
    if (!strstr(buf, "custom=1")) return; // Ignore legacy non-custom preset.cfg files that had stale CFG 1.0 / Euler a!

    char* line = buf;
    while (*line) {
        char* nl = strchr(line, '\n');
        if (nl) *nl = 0;
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\r' || line[len - 1] == ' ')) line[--len] = 0;
        char* eq = strchr(line, '=');
        if (eq && line[0] != '#') {
            *eq = 0;
            const char* k = line;
            const char* v = eq + 1;
            if (strcmp(k, "width") == 0) out->width = (uint32_t)atoi(v);
            else if (strcmp(k, "height") == 0) out->height = (uint32_t)atoi(v);
            else if (strcmp(k, "steps") == 0) out->steps = (uint32_t)atoi(v);
            else if (strcmp(k, "cfg") == 0) out->cfg = (float)atof(v);
            else if (strcmp(k, "upscale") == 0) out->upscale = atoi(v);
            else if (strcmp(k, "sampler") == 0) {
                if (ci_has(v, "dpm")) out->sampler = 3;
                else if (ci_has(v, "lcm")) out->sampler = 2;
                else if (strcmp(v, "Euler") == 0 || strcmp(v, "euler") == 0) out->sampler = 1;
                else out->sampler = 0;
            } else if (strcmp(k, "description") == 0) snprintf(out->description, sizeof(out->description), "%s", v);
            else if (strcmp(k, "prompt") == 0) snprintf(out->prompt, sizeof(out->prompt), "%s", v);
            else if (strcmp(k, "negative") == 0) snprintf(out->negative, sizeof(out->negative), "%s", v);
        }
        if (!nl) break;
        line = nl + 1;
    }
}

// ---- RealESRGAN Anime v3 GPU Upscaler (4x SRVGGNetCompact -> crisp area-resampled 1080p / 2x) ---
int sdxl_upscale_rgb(SdxlEngine* e, const uint8_t* src_rgb, uint32_t src_w, uint32_t src_h,
                     uint8_t** dst_rgb_out, uint32_t* dst_w_out, uint32_t* dst_h_out,
                     int upscale_mode, int eco_mode,
                     SdxlProgressFn progress, SdxlCancelFn cancel, void* user) {
    if (!e->has_upscaler || upscale_mode <= 0) return -1;

    // Compute exact target dimensions:
    //   mode 1 = 1080p Fit (capped at 1920 long edge)
    //   mode 2 = 2x HD (capped at 1920 long edge)
    //   mode 3 = 4K UHD Fit (short edge 2160, capped at 3840 long edge, max 4x RealESRGAN)
    uint32_t dw = src_w, dh = src_h;
    if (upscale_mode == 3) {
        double s = 2160.0 / (double)(src_w < src_h ? src_w : src_h);
        if (s < 1.0) s = 1.0;
        if (s > 4.0) s = 4.0;
        double lw = (double)src_w * s, lh = (double)src_h * s;
        double mx = lw > lh ? lw : lh;
        if (mx > 3840.0) { lw *= 3840.0 / mx; lh *= 3840.0 / mx; }
        dw = ((uint32_t)(lw + 1.0)) & ~1u;
        dh = ((uint32_t)(lh + 1.0)) & ~1u;
    } else if (upscale_mode == 1) {
        double s = 1080.0 / (double)(src_w < src_h ? src_w : src_h);
        if (s < 1.0) s = 1.0;
        if (s > 4.0) s = 4.0;
        double lw = (double)src_w * s, lh = (double)src_h * s;
        double mx = lw > lh ? lw : lh;
        if (mx > 1920.0) { lw *= 1920.0 / mx; lh *= 1920.0 / mx; }
        dw = ((uint32_t)(lw + 1.0)) & ~1u;
        dh = ((uint32_t)(lh + 1.0)) & ~1u;
    } else {
        double lw = (double)src_w * 2.0, lh = (double)src_h * 2.0;
        double mx = lw > lh ? lw : lh;
        if (mx > 1920.0) { lw *= 1920.0 / mx; lh *= 1920.0 / mx; }
        dw = ((uint32_t)(lw + 1.0)) & ~1u;
        dh = ((uint32_t)(lh + 1.0)) & ~1u;
    }
    if (dw < src_w) dw = src_w;
    if (dh < src_h) dh = src_h;

    const uint32_t TILE = 128, OV = 12, STRIDE = TILE - OV;
    float* acc = (float*)calloc((size_t)dw * dh * 3, sizeof(float));
    float* wgt = (float*)calloc((size_t)dw * dh, sizeof(float));
    float* tile4 = (float*)malloc((size_t)(TILE * 4) * (TILE * 4) * 3 * sizeof(float));
    uint8_t* out_rgb = (uint8_t*)malloc((size_t)dw * dh * 3);
    // Graceful fallback to 1080p if 4K CPU accumulator exceeds free flex RAM (e.g. Base PS4)
    if ((!acc || !wgt || !tile4 || !out_rgb) && upscale_mode == 3) {
        free(acc); free(wgt); free(tile4); free(out_rgb);
        glog("upscale 4K (%ux%u) OOM on flex heap -> falling back to 1080p", dw, dh);
        return sdxl_upscale_rgb(e, src_rgb, src_w, src_h, dst_rgb_out, dst_w_out, dst_h_out,
                                1, eco_mode, progress, cancel, user);
    }
    if (!acc || !wgt || !tile4 || !out_rgb) {
        free(acc); free(wgt); free(tile4); free(out_rgb);
        fail(e, "oom upscaler");
        return -1;
    }

    e->eco_mode = eco_mode;
    gpu_begin(e);
    e->cancel_fn = cancel;
    e->cancel_user = user;
    double tu0 = now_sec(e);

    const SdxlPack* p = &e->upscaler;
    const uint32_t W4 = src_w * 4u, H4 = src_h * 4u;
    const float sx_4 = (float)W4 / (float)dw;
    const float sy_4 = (float)H4 / (float)dh;
    const float rx = (sx_4 * 0.5f > 0.75f) ? (sx_4 * 0.5f) : 0.75f;
    const float ry = (sy_4 * 0.5f > 0.75f) ? (sy_4 * 0.5f) : 0.75f;

    // Count total tiles for progress
    int total_tiles = 0, done_tiles = 0;
    for (uint32_t ty0 = 0; ty0 < src_h; ty0 += STRIDE) {
        uint32_t th = (src_h - ty0 < TILE) ? (src_h - ty0) : TILE;
        if (ty0 > 0 && th < TILE && src_h >= TILE) { ty0 = src_h - TILE; th = TILE; }
        for (uint32_t tx0 = 0; tx0 < src_w; tx0 += STRIDE) {
            uint32_t tw = (src_w - tx0 < TILE) ? (src_w - tx0) : TILE;
            if (tx0 > 0 && tw < TILE && src_w >= TILE) { tx0 = src_w - TILE; tw = TILE; }
            total_tiles++;
            if (tx0 + tw >= src_w) break;
        }
        if (ty0 + th >= src_h) break;
    }

    for (uint32_t ty0 = 0; ty0 < src_h; ty0 += STRIDE) {
        uint32_t th = (src_h - ty0 < TILE) ? (src_h - ty0) : TILE;
        if (ty0 > 0 && th < TILE && src_h >= TILE) { ty0 = src_h - TILE; th = TILE; }
        for (uint32_t tx0 = 0; tx0 < src_w; tx0 += STRIDE) {
            uint32_t tw = (src_w - tx0 < TILE) ? (src_w - tx0) : TILE;
            if (tx0 > 0 && tw < TILE && src_w >= TILE) { tx0 = src_w - TILE; tw = TILE; }
            if (cancel && cancel(user)) {
                free(acc); free(wgt); free(tile4); free(out_rgb);
                e->cancel_fn = NULL;
                e->rt_phase = 0;
                return 1;
            }
            e->rt_phase = 4;
            e->rt_tiles_done = (uint32_t)done_tiles;
            e->rt_tiles_total = (uint32_t)(total_tiles > 0 ? total_tiles : 1);
            e->rt_tile_avg_sec = (done_tiles > 0) ? ((now_sec(e) - tu0) / (double)done_tiles) : 0.0;
            if (progress) {
                char msg[96];
                snprintf(msg, sizeof(msg), "Upscaling -> %ux%u (tile %d/%d)", dw, dh, done_tiles + 1, total_tiles);
                progress(user, 0.94f + 0.05f * ((float)done_tiles / (float)(total_tiles ? total_tiles : 1)), msg);
            }

            // Upload RGB tile [0..1] to buf_c
            float* in_f = (float*)(uintptr_t)e->buf_c;
            for (uint32_t y = 0; y < th; y++) {
                const uint8_t* sr = src_rgb + ((size_t)(ty0 + y) * src_w + tx0) * 3;
                float* dr = in_f + (size_t)y * tw * 3;
                for (uint32_t x = 0; x < tw * 3; x++) dr[x] = (float)sr[x] * (1.0f / 255.0f);
            }

            // SRVGGNetCompact: body.0 (3->64) + PReLU + 16x (64->64 + PReLU) + body.34 (64->48)
            conv3x3_op(e, p, "body.0.weight", "body.0.bias", e->buf_c, e->buf_a, th, tw, 3, 64, 1);
            const SdxlTensor* pw0 = pack_req(e, p, "body.1.weight");
            if (!pw0) { free(acc); free(wgt); free(tile4); free(out_rgb); return -1; }
            op_prelu(e, e->buf_a, t_addr(p, pw0), th * tw, 64);

            uint64_t cur = e->buf_a, nxt = e->buf_b;
            for (int l = 1; l <= 16; l++) {
                char wn[48], bn[48], pn[48];
                snprintf(wn, sizeof(wn), "body.%d.weight", l * 2);
                snprintf(bn, sizeof(bn), "body.%d.bias", l * 2);
                snprintf(pn, sizeof(pn), "body.%d.weight", l * 2 + 1);
                conv3x3_op(e, p, wn, bn, cur, nxt, th, tw, 64, 64, 1);
                const SdxlTensor* pw = pack_req(e, p, pn);
                if (!pw) { free(acc); free(wgt); free(tile4); free(out_rgb); return -1; }
                op_prelu(e, nxt, t_addr(p, pw), th * tw, 64);
                uint64_t tmp = cur; cur = nxt; nxt = tmp;
            }
            conv3x3_op(e, p, "body.34.weight", "body.34.bias", cur, nxt, th, tw, 64, 48, 1);
            if (gpu_flush(e)) { free(acc); free(wgt); free(tile4); free(out_rgb); return -1; }

            // PixelShuffle 4x + nearest skip into local tile4 [4*th, 4*tw, 3]
            const float* out48 = (const float*)(uintptr_t)nxt;
            uint32_t th4 = th * 4u, tw4 = tw * 4u;
            for (uint32_t y = 0; y < th; y++) {
                for (uint32_t x = 0; x < tw; x++) {
                    const float* base_rgb = in_f + ((size_t)y * tw + x) * 3;
                    const float* o48 = out48 + ((size_t)y * tw + x) * 48;
                    for (uint32_t sy = 0; sy < 4; sy++) {
                        float* drow = tile4 + ((size_t)(y * 4u + sy) * tw4 + x * 4u) * 3;
                        for (uint32_t sx = 0; sx < 4; sx++) {
                            uint32_t sub = sy * 4u + sx;
                            for (uint32_t c = 0; c < 3; c++) {
                                float v = base_rgb[c] + o48[c * 16u + sub];
                                if (v < 0.0f) v = 0.0f;
                                if (v > 1.0f) v = 1.0f;
                                drow[sx * 3 + c] = v;
                            }
                        }
                    }
                }
            }

            // Area-resample tile4 directly into dst accumulator with smooth overlap feathering
            uint32_t y4_0 = ty0 * 4u, x4_0 = tx0 * 4u;
            uint32_t y4_1 = y4_0 + th4, x4_1 = x4_0 + tw4;
            uint32_t fov4 = OV * 4u;
            int oy_start = (int)floorf(((float)y4_0 + 0.5f) / sy_4 - 0.5f);
            int oy_end   = (int)ceilf(((float)y4_1 - 0.5f) / sy_4 - 0.5f);
            int ox_start = (int)floorf(((float)x4_0 + 0.5f) / sx_4 - 0.5f);
            int ox_end   = (int)ceilf(((float)x4_1 - 0.5f) / sx_4 - 0.5f);
            if (oy_start < 0) oy_start = 0;
            if (ox_start < 0) ox_start = 0;
            if (oy_end > (int)dh) oy_end = (int)dh;
            if (ox_end > (int)dw) ox_end = (int)dw;

            for (int oy = oy_start; oy < oy_end; oy++) {
                float fy = ((float)oy + 0.5f) * sy_4 - 0.5f - (float)y4_0;
                if (fy < 0.0f || fy >= (float)th4) continue;
                float wy = 1.0f;
                if (ty0 > 0 && fy < (float)fov4) wy = (fy + 0.5f) / (float)fov4;
                if (ty0 + th < src_h && ((float)th4 - 1.0f - fy) < (float)fov4) {
                    float w2 = ((float)th4 - fy) / (float)fov4;
                    if (w2 < wy) wy = w2;
                }
                if (wy <= 0.0f) continue;
                int iy0 = (int)floorf(fy - ry), iy1 = (int)ceilf(fy + ry);
                if (iy0 < 0) iy0 = 0;
                if (iy1 >= (int)th4) iy1 = (int)th4 - 1;

                for (int ox = ox_start; ox < ox_end; ox++) {
                    float fx = ((float)ox + 0.5f) * sx_4 - 0.5f - (float)x4_0;
                    if (fx < 0.0f || fx >= (float)tw4) continue;
                    float wx = 1.0f;
                    if (tx0 > 0 && fx < (float)fov4) wx = (fx + 0.5f) / (float)fov4;
                    if (tx0 + tw < src_w && ((float)tw4 - 1.0f - fx) < (float)fov4) {
                        float w2 = ((float)tw4 - fx) / (float)fov4;
                        if (w2 < wx) wx = w2;
                    }
                    if (wx <= 0.0f) continue;
                    int ix0 = (int)floorf(fx - rx), ix1 = (int)ceilf(fx + rx);
                    if (ix0 < 0) ix0 = 0;
                    if (ix1 >= (int)tw4) ix1 = (int)tw4 - 1;

                    float rsum = 0, gsum = 0, bsum = 0, ksum = 0;
                    for (int iy = iy0; iy <= iy1; iy++) {
                        float dy = fabsf((float)iy - fy) / ry;
                        float ky = 1.0f - dy;
                        if (ky <= 0.0f) continue;
                        const float* tr = tile4 + (size_t)iy * tw4 * 3;
                        for (int ix = ix0; ix <= ix1; ix++) {
                            float dx = fabsf((float)ix - fx) / rx;
                            float kx = 1.0f - dx;
                            if (kx <= 0.0f) continue;
                            float k = kx * ky;
                            rsum += tr[ix * 3 + 0] * k;
                            gsum += tr[ix * 3 + 1] * k;
                            bsum += tr[ix * 3 + 2] * k;
                            ksum += k;
                        }
                    }
                    if (ksum > 0.0f) {
                        float wt = wx * wy;
                        size_t di = (size_t)oy * dw + (size_t)ox;
                        acc[di * 3 + 0] += (rsum / ksum) * wt;
                        acc[di * 3 + 1] += (gsum / ksum) * wt;
                        acc[di * 3 + 2] += (bsum / ksum) * wt;
                        wgt[di] += wt;
                    }
                }
            }
            done_tiles++;
            int ubr = block_breather(e);
            if (ubr != 0) { free(acc); free(wgt); free(tile4); free(out_rgb); e->cancel_fn = NULL; return ubr; }
            if (tx0 + tw >= src_w) break;
        }
        if (ty0 + th >= src_h) break;
    }

    for (size_t i = 0; i < (size_t)dw * dh; i++) {
        float inv = wgt[i] > 0.0f ? (1.0f / wgt[i]) : 1.0f;
        for (int c = 0; c < 3; c++) {
            float v = acc[i * 3 + c] * inv;
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            out_rgb[i * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
        }
    }
    free(acc); free(wgt); free(tile4);
    e->cancel_fn = NULL;
    e->cancel_user = NULL;
    e->rt_phase = 0;
    e->stats.upscale_ms = (now_sec(e) - tu0) * 1000.0;
    glog("upscale RealESRGAN Anime v3: %ux%u -> %ux%u (%d tiles) in %.2fs",
         src_w, src_h, dw, dh, done_tiles, e->stats.upscale_ms / 1000.0);
    *dst_rgb_out = out_rgb;
    *dst_w_out = dw;
    *dst_h_out = dh;
    return 0;
}

static int sdxl_generate_internal(SdxlEngine* e, const SdxlParams* p, const uint8_t* init_rgb, float denoise,
                                  uint8_t* rgb_out, SdxlProgressFn progress, SdxlCancelFn cancel, void* user) {
    uint32_t W = p->width, H = p->height;
    uint32_t max_dim = (e->is_sd15 || !e->neo) ? 1280u : 1344u;
    uint64_t max_lat = (e->is_sd15 || !e->neo) ? 12288ull : 21504ull;
    if (W < 256 || H < 256 || W > max_dim || H > max_dim || (W % 64) != 0 || (H % 64) != 0 ||
        (uint64_t)(W / 8u) * (H / 8u) > max_lat) {
        fail(e, "resolution must be multiple of 64 and <= %s (got %ux%u)",
             (e->is_sd15 || !e->neo) ? "768x1024" : "1024x1344", W, H);
        return -1;
    }
    uint32_t steps = p->steps;
    if (steps < 1) steps = 1;
    if (steps > 64) steps = 64;
    float cfg = p->cfg_scale;
    int eco = p->eco_mode;
    if (eco < 0) eco = 0;
    if (eco > 3) eco = 3;

    double saved_gemm = e->stats.bench_gemm_tflops;
    double saved_attn = e->stats.bench_attn_ms;
    double saved_modes[4];
    memcpy(saved_modes, e->stats.bench_mode_tf, sizeof(saved_modes));
    memset(&e->stats, 0, sizeof(e->stats));
    e->stats.bench_gemm_tflops = saved_gemm;
    e->stats.bench_attn_ms = saved_attn;
    memcpy(e->stats.bench_mode_tf, saved_modes, sizeof(saved_modes));
    e->stats.width = W; e->stats.height = H; e->stats.steps = steps; e->stats.cfg = cfg;
    // Apply active hardware CU mask BEFORE starting GPU work!
    e->eco_mode = eco;
    gpu_begin(e);
    e->cancel_fn = cancel;
    e->cancel_user = user;

    int cpu0 = 0, soc0 = 0;
    int t0_c = sdxl_read_temp_c(&cpu0, &soc0);
    e->stats.temp_start_c = t0_c;
    e->stats.temp_peak_c = t0_c;

    double t_start = now_sec(e);
    e->rt_phase = 1;
    e->rt_step_done = 0;
    e->rt_steps_total = steps;
    e->rt_tiles_done = 0;
    e->rt_tiles_total = 0;
    e->rt_step_avg_sec = 0.0;
    e->rt_tile_avg_sec = 0.0;

    if (progress) progress(user, 0.02f, "Encoding prompt (CLIP-L + OpenCLIP-G)");
    double tc0 = now_sec(e);
    if (encode_prompt(e, p->prompt, 0, e->ctx_cond, e->pool_cond)) { e->rt_phase = 0; e->cancel_fn = NULL; return -1; }
    e->stats.clip_l_ms = (now_sec(e) - tc0) * 1000.0;
    if (cfg > 1.001f) {
        double tc1 = now_sec(e);
        if (encode_prompt(e, p->negative, 1, e->ctx_uncond, e->pool_uncond)) { e->rt_phase = 0; e->cancel_fn = NULL; return -1; }
        e->stats.clip_g_ms = (now_sec(e) - tc1) * 1000.0;
    }
    if (cancel && cancel(user)) { e->rt_phase = 0; e->cancel_fn = NULL; return 1; }
    glog("clip: cond=%.1f ms, uncond=%.1f ms", e->stats.clip_l_ms, e->stats.clip_g_ms);

    float timesteps[64], sigmas[65];
    build_sigmas(steps, p->sampler, timesteps, sigmas);

    uint32_t LH = H / 8u, LW = W / 8u;
    size_t lat_n = (size_t)LH * LW * 4u;
    float* lat = (float*)malloc(lat_n * sizeof(float));
    float* noise = (float*)malloc(lat_n * sizeof(float));
    float* old_denoised = (float*)malloc(lat_n * sizeof(float));
    if (!lat || !noise || !old_denoised) {
        free(lat); free(noise); free(old_denoised);
        e->cancel_fn = NULL; fail(e, "oom latent"); return -1;
    }
    float old_h = 0.0f;
    int has_old_d = 0;

    Rng rng;
    rng_seed(&rng, p->seed ? p->seed : 42ull);
    uint32_t start_step = 0;
    if (init_rgb) {
        if (denoise < 0.08f) denoise = 0.08f;
        if (denoise > 1.0f) denoise = 1.0f;
        if (progress) progress(user, 0.04f, "Encoding face crop (VAE Encoder)...");
        int er = vae_encode_tiled(e, init_rgb, H, W, lat, progress, user);
        if (er != 0) {
            free(lat); free(noise); free(old_denoised);
            e->rt_phase = 0; e->cancel_fn = NULL;
            return er;
        }
        start_step = (uint32_t)floorf((1.0f - denoise) * (float)steps);
        if (start_step >= steps) start_step = steps - 1u;
        rng_randn(&rng, noise, lat_n);
        float s0 = sigmas[start_step];
        for (size_t i = 0; i < lat_n; i++) lat[i] += noise[i] * s0;
    } else {
        rng_randn(&rng, lat, lat_n);
        float init_sigma = sqrtf(sigmas[0] * sigmas[0] + 1.0f);
        for (size_t i = 0; i < lat_n; i++) lat[i] *= init_sigma;
    }

    e->gpu_acc_sec = 0.0;
    e->flops_acc = 0.0;
    double tu_start = now_sec(e);

    for (uint32_t s = start_step; s < steps; s++) {
        if (cancel && cancel(user)) { free(lat); free(noise); free(old_denoised); e->rt_phase = 0; e->cancel_fn = NULL; return 1; }
        e->rt_phase = 2;
        e->rt_step_done = s;
        e->rt_steps_total = steps;
        if (s > 0) e->rt_step_avg_sec = (now_sec(e) - tu_start) / (double)s;
        int cur_t = sdxl_read_temp_c(NULL, NULL);
        if (cur_t > e->stats.temp_peak_c) e->stats.temp_peak_c = cur_t;
        char msg[128];
        if (cur_t > 0) snprintf(msg, sizeof(msg), "Step %u/%u | %s | %dC", s + 1, steps, sdxl_eco_name_ex(eco, e->neo), cur_t);
        else snprintf(msg, sizeof(msg), "Step %u/%u | %s", s + 1, steps, sdxl_eco_name_ex(eco, e->neo));
        if (progress) progress(user, 0.05f + 0.85f * ((float)s / (float)steps), msg);

        float sigma = sigmas[s];
        float sigma_next = sigmas[s + 1];
        float c_in = 1.0f / sqrtf(sigma * sigma + 1.0f);
        float* lin = (float*)(uintptr_t)e->lat_in;
        for (size_t i = 0; i < lat_n; i++) lin[i] = lat[i] * c_in;

        double ts0 = now_sec(e);
        int ur0 = unet_forward(e, e->lat_in, timesteps[s], e->ctx_cond, e->pool_cond, H, W, e->eps_c);
        if (ur0 != 0) { free(lat); free(noise); free(old_denoised); e->rt_phase = 0; e->cancel_fn = NULL; return ur0; }
        double cond_ms = (now_sec(e) - ts0) * 1000.0;
        if (cancel && cancel(user)) { free(lat); free(noise); free(old_denoised); e->rt_phase = 0; e->cancel_fn = NULL; return 1; }

        const float* ec = (const float*)(uintptr_t)e->eps_c;
        const float* eu = NULL;
        double uncond_ms = 0.0;
        if (cfg > 1.001f) {
            double ts1 = now_sec(e);
            int ur1 = unet_forward(e, e->lat_in, timesteps[s], e->ctx_uncond, e->pool_uncond, H, W, e->eps_u);
            if (ur1 != 0) { free(lat); free(noise); free(old_denoised); e->rt_phase = 0; e->cancel_fn = NULL; return ur1; }
            uncond_ms = (now_sec(e) - ts1) * 1000.0;
            eu = (const float*)(uintptr_t)e->eps_u;
        }

        if (s == 0 || s == steps - 1 || ((s + 1) % 4 == 0)) {
            glog("  step %2u/%u: cond=%.0f ms, uncond=%.0f ms", s + 1, steps, cond_ms, uncond_ms);
        }

        // For LCM / DMD2 with CFG > 1.0 (negative prompt active), compute Lin et al. (2023) CFG variance rescale
        // (phi = 0.7) so negative prompts cleanly fix eyes/hands without over-saturating distilled models!
        float cfg_rescale = 1.0f;
        if (p->sampler == 2 && eu != NULL) {
            double sum_c = 0.0, sq_c = 0.0, sum_g = 0.0, sq_g = 0.0;
            for (size_t i = 0; i < lat_n; i++) {
                double vc = ec[i];
                double vg = eu[i] + cfg * (ec[i] - eu[i]);
                sum_c += vc; sq_c += vc * vc;
                sum_g += vg; sq_g += vg * vg;
            }
            double n_d = (double)lat_n;
            double var_c = (sq_c / n_d) - (sum_c / n_d) * (sum_c / n_d);
            double var_g = (sq_g / n_d) - (sum_g / n_d) * (sum_g / n_d);
            if (var_g > var_c && var_c > 1e-8) {
                float ratio = (float)sqrt(var_c / var_g);
                cfg_rescale = 0.7f * ratio + 0.3f;
            }
        }

        // Sampler update:
        //   0 = Euler a (Karras for >=10 steps)
        //   1 = Euler (Karras for >=10 steps)
        //   2 = LCM (Latent Consistency / DMD2 with variance-preserving late-step refinement)
        //   3 = DPM++ 2M Karras (2nd-order multistep ODE solver in log-SNR space)
        if (p->sampler == 3) {
            float h = (sigma_next > 0.0f && sigma > 0.0f) ? (logf(sigma) - logf(sigma_next)) : 0.0f;
            float alpha_ratio = (sigma > 0.0f) ? (sigma_next / sigma) : 0.0f;
            float c_d = 1.0f - alpha_ratio;
            float r = (has_old_d && h > 1e-6f) ? (old_h / h) : 0.0f;
            float inv_2r = (r > 1e-6f) ? (0.5f / r) : 0.0f;
            for (size_t i = 0; i < lat_n; i++) {
                float eps = eu ? (eu[i] + cfg * (ec[i] - eu[i])) : ec[i];
                float d = lat[i] - sigma * eps;
                if (sigma_next <= 0.0f) {
                    lat[i] = d;
                } else {
                    float d_corr = has_old_d ? ((1.0f + inv_2r) * d - inv_2r * old_denoised[i]) : d;
                    lat[i] = alpha_ratio * lat[i] + c_d * d_corr;
                }
                old_denoised[i] = d;
            }
            old_h = h;
            has_old_d = 1;
        } else if (p->sampler == 2) {
            if (sigma_next > 0.0f) rng_randn(&rng, noise, lat_n);
            // Taper stochasticity gamma on final detail-forming step(s) when steps >= 5 while keeping exact sigma_next^2 variance!
            float gamma = 1.0f;
            if (steps >= 5u && sigma_next > 0.0f && sigma_next < 1.15f) {
                gamma = (s + 2u >= steps) ? 0.45f : 0.75f;
            }
            float c_eps = sigma_next * sqrtf(1.0f - gamma * gamma);
            float c_z   = sigma_next * gamma;
            for (size_t i = 0; i < lat_n; i++) {
                float eps = eu ? ((eu[i] + cfg * (ec[i] - eu[i])) * cfg_rescale) : ec[i];
                float x0 = lat[i] - sigma * eps;
                lat[i] = (sigma_next > 0.0f) ? (x0 + c_eps * eps + c_z * noise[i]) : x0;
            }
        } else {
            float sigma_up = 0.0f;
            if (p->sampler == 0 && sigma_next > 0.0f && sigma > 0.0f) {
                float r = (sigma_next * sigma_next * (sigma * sigma - sigma_next * sigma_next)) / (sigma * sigma);
                sigma_up = r > 0.0f ? sqrtf(r) : 0.0f;
                if (sigma_up > sigma_next) sigma_up = sigma_next;
            }
            float sd2 = sigma_next * sigma_next - sigma_up * sigma_up;
            float sigma_down = sd2 > 0.0f ? sqrtf(sd2) : 0.0f;
            float dt = sigma_down - sigma;

            if (sigma_up > 0.0f) rng_randn(&rng, noise, lat_n);
            for (size_t i = 0; i < lat_n; i++) {
                float eps = eu ? (eu[i] + cfg * (ec[i] - eu[i])) : ec[i];
                lat[i] += eps * dt;
                if (sigma_up > 0.0f) lat[i] += noise[i] * sigma_up;
            }
        }
    }

    e->stats.unet_total_ms = (now_sec(e) - tu_start) * 1000.0;
    e->stats.unet_gpu_ms = e->gpu_acc_sec * 1000.0;
    e->stats.total_gflops = e->flops_acc * 1e-9;
    e->stats.effective_tflops = (e->gpu_acc_sec > 0.001) ? (e->flops_acc / (e->gpu_acc_sec * 1e12)) : 0.0;

    e->rt_phase = 3;
    e->rt_tiles_done = 0;
    e->rt_tiles_total = 0;
    e->rt_tile_avg_sec = 0.0;
    if (progress) progress(user, 0.90f, "Decoding image (VAE)");
    e->gpu_acc_sec = 0.0;
    double tv0 = now_sec(e);
    int vr = vae_decode_tiled(e, lat, H, W, rgb_out, progress, user);
    e->stats.vae_ms = (now_sec(e) - tv0) * 1000.0;
    e->stats.vae_gpu_ms = e->gpu_acc_sec * 1000.0;
    free(lat);
    free(noise);
    free(old_denoised);
    e->rt_phase = 0;
    e->cancel_fn = NULL;
    e->cancel_user = NULL;
    if (vr != 0) return vr;

    e->last_gen_sec = now_sec(e) - t_start;
    e->stats.total_sec = e->last_gen_sec;
    e->stats.temp_end_c = sdxl_read_temp_c(NULL, NULL);
    if (e->stats.temp_end_c > e->stats.temp_peak_c) e->stats.temp_peak_c = e->stats.temp_end_c;

    static const char* SMP_LOG[4] = {"EulerA", "Euler", "LCM", "DPM2MKarras"};
    glog("stats %s %ux%u %usteps cfg=%.1f %s %s | total=%.2fs | unet=%.2fs (gpu=%.2fs, %.0fms/step, %.2f TFLOPS, %.0f GFLOPs) | vae=%.2fs (%d tiles) | cool=%.2fs",
         e->is_sd15 ? "SD1.5" : "SDXL", W, H, steps, (double)cfg,
         SMP_LOG[(p->sampler >= 0 && p->sampler <= 3) ? p->sampler : 0], sdxl_eco_name_ex(eco, e->neo),
         e->stats.total_sec,
         e->stats.unet_total_ms / 1000.0, e->stats.unet_gpu_ms / 1000.0,
         e->stats.unet_total_ms / (double)steps, e->stats.effective_tflops, e->stats.total_gflops,
         e->stats.vae_ms / 1000.0, e->stats.vae_tiles, e->stats.cooldown_ms / 1000.0);

    if (progress) progress(user, 1.00f, "Done");
    return 0;
}

int sdxl_generate(SdxlEngine* e, const SdxlParams* p, uint8_t* rgb_out,
                  SdxlProgressFn progress, SdxlCancelFn cancel, void* user) {
    return sdxl_generate_internal(e, p, NULL, 1.0f, rgb_out, progress, cancel, user);
}

int sdxl_img2img(SdxlEngine* e, const SdxlParams* p, const uint8_t* init_rgb, float denoise,
                 uint8_t* rgb_out, SdxlProgressFn progress, SdxlCancelFn cancel, void* user) {
    if (!init_rgb) { fail(e, "img2img requires init_rgb"); return -1; }
    return sdxl_generate_internal(e, p, init_rgb, denoise, rgb_out, progress, cancel, user);
}
