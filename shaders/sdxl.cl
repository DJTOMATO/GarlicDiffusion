// Garlic Diffusion v1.2: Ultra-optimized OpenCL C 1.2 kernels for SDXL on PS4 / PS4 Pro GCN
// - Three auto-tuned transposed-LDS Q8_0 GEMM variants using contiguous float2 (ds_read_b64) loads:
//     gemm_q8_64x64_t   (BM=64,  BN=64,  BK=32, 4x4/thr, <=56 VGPRs, 4 waves/SIMD)
//     gemm_q8_128x64_t  (BM=128, BN=64,  BK=16, 8x4/thr, 12.5 KiB LDS)
//     gemm_q8_128x128_t (BM=128, BN=128, BK=16, 8x8/thr, 16.5 KiB LDS, 8 FMAs/LDS-read)
// - Transposed-LDS float2 gemm_f32 (64x64x32) for fast VAE Decode
// - 16-lane ds_swizzle FlashAttention-style tiled attn_d64 (28 VGPRs, 0 spills)

typedef unsigned int uint;
typedef unsigned long ulong;
typedef signed char schar;
typedef _Float16 half_t;
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float4 __attribute__((ext_vector_type(4)));

#define LID __builtin_amdgcn_workitem_id_x()
#define GID __builtin_amdgcn_workgroup_id_x()
#define BARRIER()                                                    \
    do {                                                             \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");       \
        __builtin_amdgcn_s_barrier();                                \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");       \
    } while (0)
#define LOG2E 1.44269504f

static inline float rsq(float x) { return __builtin_amdgcn_rsqf(x); }
static inline float exp_(float x) { return __builtin_exp2f(x * LOG2E); }
static inline float silu_(float x) { return x / (1.0f + exp_(-x)); }
static inline float quick_gelu_(float x) { return x / (1.0f + exp_(-1.702f * x)); }
static inline float erff_(float x) {
    float ax = x < 0.0f ? -x : x;
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float p = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    float y = 1.0f - p * exp_(-ax * ax);
    return x < 0.0f ? -y : y;
}
static inline float gelu_exact_(float x) { return 0.5f * x * (1.0f + erff_(x * 0.70710678f)); }

#define SWIZZLE_XOR_ADD(v, MASK) do { \
    union { float f; int i; } _u; \
    _u.f = (v); \
    _u.i = __builtin_amdgcn_ds_swizzle(_u.i, (MASK)); \
    (v) += _u.f; \
} while (0)

static inline float sum8_wave(float v) {
    SWIZZLE_XOR_ADD(v, 0x041F);
    SWIZZLE_XOR_ADD(v, 0x081F);
    SWIZZLE_XOR_ADD(v, 0x101F);
    return v;
}

static inline float sum16_wave(float v) {
    SWIZZLE_XOR_ADD(v, 0x041F);
    SWIZZLE_XOR_ADD(v, 0x081F);
    SWIZZLE_XOR_ADD(v, 0x101F);
    SWIZZLE_XOR_ADD(v, 0x201F);
    return v;
}

static float group_sum256(local float* red, float v) {
    uint t = LID;
    red[t] = v;
    BARRIER();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        BARRIER();
    }
    float r = red[0];
    BARRIER();
    return r;
}

// 1. Elementwise add
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void add_f32(global const float* a, global const float* b, global float* out, uint n) {
    uint i = GID * 256 + LID;
    if (i < n) out[i] = a[i] + b[i];
}

// 1b. Copy float32
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void copy_f32(global const float* src, global float* dst, uint n) {
    uint i = GID * 256 + LID;
    if (i < n) dst[i] = src[i];
}

// 2. Linear combination
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void scale_add_f32(global const float* a, global const float* b, global float* out, float sa, float sb, uint n) {
    uint i = GID * 256 + LID;
    if (i < n) out[i] = sa * a[i] + sb * b[i];
}

// ---- GEMM Q8_0 VARIANT A: 64x64x32 TRANSPOSED LDS (As[32][68], Bs[32][68], 4x4/thread, float2 ds_read_b64) ----
// groups = ((M + 63) / 64) * ((N + 63) / 64), threads = 256.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void gemm_q8_64x64_t(global const float* A, global const schar* Wq, global const half_t* Wsc,
                     global const float* bias, global float* C,
                     uint M, uint N, uint K, uint add_res) {
    local float As[32 * 68];
    local float Bs[32 * 68];

    uint grid_n = (N + 63u) >> 6;
    uint bm = (GID / grid_n) << 6;
    uint bn = (GID % grid_n) << 6;
    uint tid = LID;
    uint ty = tid >> 4;
    uint tx = tid & 15u;

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    uint k_blocks = K >> 5;
    uint k_u32 = K >> 2;
    global const float4* A4 = (global const float4*)A;
    global const uint* Wq4 = (global const uint*)Wq;
    uint lr = tid >> 3;
    uint lc4 = tid & 7u;
    uint lc = lc4 << 2;

    for (uint kb = 0; kb < k_blocks; kb++) {
        uint k4 = (kb << 3) + lc4;
        #pragma unroll
        for (uint p = 0; p < 2; p++) {
            uint r = lr + (p << 5);
            uint gm = bm + r;
            float4 v = (gm < M) ? A4[(ulong)gm * k_u32 + k4] : (float4)(0.0f);
            As[(lc + 0) * 68 + r] = v.x;
            As[(lc + 1) * 68 + r] = v.y;
            As[(lc + 2) * 68 + r] = v.z;
            As[(lc + 3) * 68 + r] = v.w;
        }
        #pragma unroll
        for (uint p = 0; p < 2; p++) {
            uint r = lr + (p << 5);
            uint gn = bn + r;
            if (gn < N) {
                float sc = (float)Wsc[(ulong)gn * k_blocks + kb];
                uint w4 = Wq4[(ulong)gn * k_u32 + k4];
                Bs[(lc + 0) * 68 + r] = (float)((int)(w4 << 24) >> 24) * sc;
                Bs[(lc + 1) * 68 + r] = (float)((int)(w4 << 16) >> 24) * sc;
                Bs[(lc + 2) * 68 + r] = (float)((int)(w4 << 8) >> 24) * sc;
                Bs[(lc + 3) * 68 + r] = (float)((int)w4 >> 24) * sc;
            } else {
                Bs[(lc + 0) * 68 + r] = 0.0f;
                Bs[(lc + 1) * 68 + r] = 0.0f;
                Bs[(lc + 2) * 68 + r] = 0.0f;
                Bs[(lc + 3) * 68 + r] = 0.0f;
            }
        }
        BARRIER();

        uint r0 = ty << 2;
        uint c0 = tx << 2;
        #pragma unroll 4
        for (uint kk = 0; kk < 32; kk++) {
            local const float2* ap = (local const float2*)(As + kk * 68 + r0);
            local const float2* bp = (local const float2*)(Bs + kk * 68 + c0);
            float2 a01 = ap[0], a23 = ap[1];
            float2 b01 = bp[0], b23 = bp[1];
            acc[0][0] += a01.x * b01.x; acc[0][1] += a01.x * b01.y; acc[0][2] += a01.x * b23.x; acc[0][3] += a01.x * b23.y;
            acc[1][0] += a01.y * b01.x; acc[1][1] += a01.y * b01.y; acc[1][2] += a01.y * b23.x; acc[1][3] += a01.y * b23.y;
            acc[2][0] += a23.x * b01.x; acc[2][1] += a23.x * b01.y; acc[2][2] += a23.x * b23.x; acc[2][3] += a23.x * b23.y;
            acc[3][0] += a23.y * b01.x; acc[3][1] += a23.y * b01.y; acc[3][2] += a23.y * b23.x; acc[3][3] += a23.y * b23.y;
        }
        BARRIER();
    }

    uint r0 = bm + (ty << 2);
    uint c0 = bn + (tx << 2);
    #pragma unroll
    for (uint i = 0; i < 4; i++) {
        uint gm = r0 + i;
        if (gm >= M) continue;
        ulong coff = (ulong)gm * N + c0;
        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            uint gn = c0 + j;
            if (gn >= N) continue;
            float v = acc[i][j];
            if (bias) v += bias[gn];
            if (add_res) v += C[coff + j];
            C[coff + j] = v;
        }
    }
}

// ---- GEMM Q8_0 VARIANT B: 128x64x16 TRANSPOSED LDS (As[16][132], Bs[16][68], 8x4/thread, 12.5 KiB LDS) ----
// groups = ((M + 127) / 128) * ((N + 63) / 64), threads = 256.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void gemm_q8_128x64_t(global const float* A, global const schar* Wq, global const half_t* Wsc,
                      global const float* bias, global float* C,
                      uint M, uint N, uint K, uint add_res) {
    local float As[16 * 132];
    local float Bs[16 * 68];

    uint grid_n = (N + 63u) >> 6;
    uint bm = (GID / grid_n) << 7;
    uint bn = (GID % grid_n) << 6;
    uint tid = LID;
    uint ty = tid >> 4;  // 0..15 -> 8 rows (ty*8 + 0..7)
    uint tx = tid & 15u; // 0..15 -> 4 cols (tx*4 + 0..3)

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    uint k_blocks = K >> 5;
    uint k_u32 = K >> 2;
    global const float4* A4 = (global const float4*)A;
    global const uint* Wq4 = (global const uint*)Wq;

    uint lr = tid >> 2;   // 0..63
    uint lc4 = tid & 3u;  // 0..3 (4x4 = 16 elements along K!)
    uint lc = lc4 << 2;

    for (uint k0 = 0; k0 < K; k0 += 16u) {
        uint kb = k0 >> 5;
        uint k4 = (k0 >> 2) + lc4;
        // Load A [128, 16]: 2x float4 per thread (lr and lr+64)
        #pragma unroll
        for (uint p = 0; p < 2; p++) {
            uint r = lr + (p << 6);
            uint gm = bm + r;
            float4 v = (gm < M) ? A4[(ulong)gm * k_u32 + k4] : (float4)(0.0f);
            As[(lc + 0) * 132 + r] = v.x;
            As[(lc + 1) * 132 + r] = v.y;
            As[(lc + 2) * 132 + r] = v.z;
            As[(lc + 3) * 132 + r] = v.w;
        }
        // Load B [64, 16]: 1x packed uint per thread (lr)
        {
            uint r = lr;
            uint gn = bn + r;
            if (gn < N) {
                float sc = (float)Wsc[(ulong)gn * k_blocks + kb];
                uint w4 = Wq4[(ulong)gn * k_u32 + k4];
                Bs[(lc + 0) * 68 + r] = (float)((int)(w4 << 24) >> 24) * sc;
                Bs[(lc + 1) * 68 + r] = (float)((int)(w4 << 16) >> 24) * sc;
                Bs[(lc + 2) * 68 + r] = (float)((int)(w4 << 8) >> 24) * sc;
                Bs[(lc + 3) * 68 + r] = (float)((int)w4 >> 24) * sc;
            } else {
                Bs[(lc + 0) * 68 + r] = 0.0f;
                Bs[(lc + 1) * 68 + r] = 0.0f;
                Bs[(lc + 2) * 68 + r] = 0.0f;
                Bs[(lc + 3) * 68 + r] = 0.0f;
            }
        }
        BARRIER();

        uint r0 = ty << 3;
        uint c0 = tx << 2;
        #pragma unroll 4
        for (uint kk = 0; kk < 16; kk++) {
            local const float2* ap = (local const float2*)(As + kk * 132 + r0);
            local const float2* bp = (local const float2*)(Bs + kk * 68 + c0);
            float2 a01 = ap[0], a23 = ap[1], a45 = ap[2], a67 = ap[3];
            float2 b01 = bp[0], b23 = bp[1];
            acc[0][0] += a01.x * b01.x; acc[0][1] += a01.x * b01.y; acc[0][2] += a01.x * b23.x; acc[0][3] += a01.x * b23.y;
            acc[1][0] += a01.y * b01.x; acc[1][1] += a01.y * b01.y; acc[1][2] += a01.y * b23.x; acc[1][3] += a01.y * b23.y;
            acc[2][0] += a23.x * b01.x; acc[2][1] += a23.x * b01.y; acc[2][2] += a23.x * b23.x; acc[2][3] += a23.x * b23.y;
            acc[3][0] += a23.y * b01.x; acc[3][1] += a23.y * b01.y; acc[3][2] += a23.y * b23.x; acc[3][3] += a23.y * b23.y;
            acc[4][0] += a45.x * b01.x; acc[4][1] += a45.x * b01.y; acc[4][2] += a45.x * b23.x; acc[4][3] += a45.x * b23.y;
            acc[5][0] += a45.y * b01.x; acc[5][1] += a45.y * b01.y; acc[5][2] += a45.y * b23.x; acc[5][3] += a45.y * b23.y;
            acc[6][0] += a67.x * b01.x; acc[6][1] += a67.x * b01.y; acc[6][2] += a67.x * b23.x; acc[6][3] += a67.x * b23.y;
            acc[7][0] += a67.y * b01.x; acc[7][1] += a67.y * b01.y; acc[7][2] += a67.y * b23.x; acc[7][3] += a67.y * b23.y;
        }
        BARRIER();
    }

    uint r0 = bm + (ty << 3);
    uint c0 = bn + (tx << 2);
    #pragma unroll
    for (uint i = 0; i < 8; i++) {
        uint gm = r0 + i;
        if (gm >= M) continue;
        ulong coff = (ulong)gm * N + c0;
        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            uint gn = c0 + j;
            if (gn >= N) continue;
            float v = acc[i][j];
            if (bias) v += bias[gn];
            if (add_res) v += C[coff + j];
            C[coff + j] = v;
        }
    }
}

// ---- GEMM Q8_0 VARIANT C: 128x128x16 TRANSPOSED LDS (8x8 = 64 outputs/thread, 8 FMAs/LDS read!) ----
// groups = ((M + 127) / 128) * ((N + 127) / 128), threads = 256.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void gemm_q8_128x128_t(global const float* A, global const schar* Wq, global const half_t* Wsc,
                       global const float* bias, global float* C,
                       uint M, uint N, uint K, uint add_res) {
    local float As[16 * 132];
    local float Bs[16 * 132];

    uint grid_n = (N + 127u) >> 7;
    uint bm = (GID / grid_n) << 7;
    uint bn = (GID % grid_n) << 7;
    uint tid = LID;
    uint ty = tid >> 4;
    uint tx = tid & 15u;

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    uint k_blocks = K >> 5;
    uint k_u32 = K >> 2;
    global const float4* A4 = (global const float4*)A;
    global const uint* Wq4 = (global const uint*)Wq;

    uint lr = tid >> 2;   // 0..63
    uint lc4 = tid & 3u;  // 0..3
    uint lc = lc4 << 2;

    for (uint k0 = 0; k0 < K; k0 += 16u) {
        uint kb = k0 >> 5;
        uint k4 = (k0 >> 2) + lc4;
        #pragma unroll
        for (uint p = 0; p < 2; p++) {
            uint r = lr + (p << 6);
            uint gm = bm + r;
            float4 v = (gm < M) ? A4[(ulong)gm * k_u32 + k4] : (float4)(0.0f);
            As[(lc + 0) * 132 + r] = v.x;
            As[(lc + 1) * 132 + r] = v.y;
            As[(lc + 2) * 132 + r] = v.z;
            As[(lc + 3) * 132 + r] = v.w;
        }
        #pragma unroll
        for (uint p = 0; p < 2; p++) {
            uint r = lr + (p << 6);
            uint gn = bn + r;
            if (gn < N) {
                float sc = (float)Wsc[(ulong)gn * k_blocks + kb];
                uint w4 = Wq4[(ulong)gn * k_u32 + k4];
                Bs[(lc + 0) * 132 + r] = (float)((int)(w4 << 24) >> 24) * sc;
                Bs[(lc + 1) * 132 + r] = (float)((int)(w4 << 16) >> 24) * sc;
                Bs[(lc + 2) * 132 + r] = (float)((int)(w4 << 8) >> 24) * sc;
                Bs[(lc + 3) * 132 + r] = (float)((int)w4 >> 24) * sc;
            } else {
                Bs[(lc + 0) * 132 + r] = 0.0f;
                Bs[(lc + 1) * 132 + r] = 0.0f;
                Bs[(lc + 2) * 132 + r] = 0.0f;
                Bs[(lc + 3) * 132 + r] = 0.0f;
            }
        }
        BARRIER();

        uint r0 = ty << 3;
        uint c0 = tx << 3;
        #pragma unroll 2
        for (uint kk = 0; kk < 16; kk++) {
            local const float2* ap = (local const float2*)(As + kk * 132 + r0);
            local const float2* bp = (local const float2*)(Bs + kk * 132 + c0);
            float a[8], b[8];
            float2 t0 = ap[0], t1 = ap[1], t2 = ap[2], t3 = ap[3];
            a[0]=t0.x; a[1]=t0.y; a[2]=t1.x; a[3]=t1.y; a[4]=t2.x; a[5]=t2.y; a[6]=t3.x; a[7]=t3.y;
            float2 u0 = bp[0], u1 = bp[1], u2 = bp[2], u3 = bp[3];
            b[0]=u0.x; b[1]=u0.y; b[2]=u1.x; b[3]=u1.y; b[4]=u2.x; b[5]=u2.y; b[6]=u3.x; b[7]=u3.y;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] += a[i] * b[j];
        }
        BARRIER();
    }

    uint r0 = bm + (ty << 3);
    uint c0 = bn + (tx << 3);
    #pragma unroll
    for (uint j = 0; j < 8; j++) {
        uint gn = c0 + j;
        if (gn >= N) continue;
        float bv = bias ? bias[gn] : 0.0f;
        #pragma unroll
        for (uint i = 0; i < 8; i++) {
            uint gm = r0 + i;
            if (gm >= M) continue;
            ulong coff = (ulong)gm * N + gn;
            float v = acc[i][j] + bv;
            if (add_res) v += C[coff];
            C[coff] = v;
        }
    }
}

// 4. Transposed-LDS F32 GEMM (VAE Decoder & small convs): 64x64x32 tile with float2 ds_read_b64!
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void gemm_f32(global const float* A, global const float* W, global const float* bias, global float* C,
              uint M, uint N, uint K, uint add_res) {
    local float As[32 * 68];
    local float Bs[32 * 68];

    uint grid_n = (N + 63u) >> 6;
    uint bm = (GID / grid_n) << 6;
    uint bn = (GID % grid_n) << 6;
    uint tid = LID;
    uint ty = tid >> 4;
    uint tx = tid & 15u;

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    uint lr = tid >> 3;
    uint lc = (tid & 7u) << 2;

    for (uint k0 = 0; k0 < K; k0 += 32u) {
        #pragma unroll
        for (uint p = 0; p < 2; p++) {
            uint r = lr + (p << 5);
            uint gm = bm + r;
            ulong aoff = (ulong)gm * K + k0 + lc;
            #pragma unroll
            for (uint t = 0; t < 4; t++)
                As[(lc + t) * 68 + r] = (gm < M && k0 + lc + t < K) ? A[aoff + t] : 0.0f;
        }
        #pragma unroll
        for (uint p = 0; p < 2; p++) {
            uint r = lr + (p << 5);
            uint gn = bn + r;
            ulong boff = (ulong)gn * K + k0 + lc;
            #pragma unroll
            for (uint t = 0; t < 4; t++)
                Bs[(lc + t) * 68 + r] = (gn < N && k0 + lc + t < K) ? W[boff + t] : 0.0f;
        }
        BARRIER();

        uint r0 = ty << 2;
        uint c0 = tx << 2;
        #pragma unroll 4
        for (uint kk = 0; kk < 32; kk++) {
            local const float2* ap = (local const float2*)(As + kk * 68 + r0);
            local const float2* bp = (local const float2*)(Bs + kk * 68 + c0);
            float2 a01 = ap[0], a23 = ap[1];
            float2 b01 = bp[0], b23 = bp[1];
            acc[0][0] += a01.x * b01.x; acc[0][1] += a01.x * b01.y; acc[0][2] += a01.x * b23.x; acc[0][3] += a01.x * b23.y;
            acc[1][0] += a01.y * b01.x; acc[1][1] += a01.y * b01.y; acc[1][2] += a01.y * b23.x; acc[1][3] += a01.y * b23.y;
            acc[2][0] += a23.x * b01.x; acc[2][1] += a23.x * b01.y; acc[2][2] += a23.x * b23.x; acc[2][3] += a23.x * b23.y;
            acc[3][0] += a23.y * b01.x; acc[3][1] += a23.y * b01.y; acc[3][2] += a23.y * b23.x; acc[3][3] += a23.y * b23.y;
        }
        BARRIER();
    }

    uint r0 = bm + (ty << 2);
    uint c0 = bn + (tx << 2);
    #pragma unroll
    for (uint i = 0; i < 4; i++) {
        uint gm = r0 + i;
        if (gm >= M) continue;
        ulong coff = (ulong)gm * N + c0;
        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            uint gn = c0 + j;
            if (gn >= N) continue;
            float v = acc[i][j];
            if (bias) v += bias[gn];
            if (add_res) v += C[coff + j];
            C[coff + j] = v;
        }
    }
}

// 5. NHWC im2col for 3x3 conv (pad=1, stride=1 or 2), vectorized float4
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void im2col_3x3(global const float* in, global float* out,
                uint H, uint W, uint C, uint W_out, uint stride, uint row0) {
    uint local_row = GID;
    uint out_idx = row0 + local_row;
    uint oy = out_idx / W_out;
    uint ox = out_idx % W_out;
    uint C4 = C >> 2;
    global const float4* in4 = (global const float4*)in;
    global float4* out4 = (global float4*)out;
    ulong dst_base4 = (ulong)local_row * (9u * C4);

    if ((C & 3u) == 0u) {
        for (uint ky = 0; ky < 3; ky++) {
            int iy = (int)(oy * stride) + (int)ky - 1;
            for (uint kx = 0; kx < 3; kx++) {
                int ix = (int)(ox * stride) + (int)kx - 1;
                ulong dst4 = dst_base4 + (ulong)(ky * 3u + kx) * C4;
                if (iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W) {
                    ulong src4 = ((ulong)(uint)iy * W + (uint)ix) * C4;
                    for (uint c = LID; c < C4; c += 256) out4[dst4 + c] = in4[src4 + c];
                } else {
                    for (uint c = LID; c < C4; c += 256) out4[dst4 + c] = (float4)(0.0f);
                }
            }
        }
    } else {
        ulong dst_base = (ulong)local_row * (9u * C);
        for (uint ky = 0; ky < 3; ky++) {
            int iy = (int)(oy * stride) + (int)ky - 1;
            for (uint kx = 0; kx < 3; kx++) {
                int ix = (int)(ox * stride) + (int)kx - 1;
                ulong dst = dst_base + (ulong)(ky * 3u + kx) * C;
                if (iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W) {
                    ulong src = ((ulong)(uint)iy * W + (uint)ix) * C;
                    for (uint c = LID; c < C; c += 256) out[dst + c] = in[src + c];
                } else {
                    for (uint c = LID; c < C; c += 256) out[dst + c] = 0.0f;
                }
            }
        }
    }
}

// 6. Two-pass NHWC GroupNorm (+ optional SiLU)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void groupnorm_silu(global const float* x, global const float* w, global const float* b,
                    global float* out, uint HW, uint C, uint G, float eps, uint do_silu) {
    local float red[256];
    uint g = GID;
    uint cg = C / G;
    uint c0 = g * cg;
    uint total = HW * cg;

    float sum = 0.0f;
    for (uint idx = LID; idx < total; idx += 256) {
        uint hw = idx / cg;
        uint dc = idx - hw * cg;
        sum += x[(ulong)hw * C + c0 + dc];
    }
    float mean = group_sum256(red, sum) / (float)total;

    float vs = 0.0f;
    for (uint idx = LID; idx < total; idx += 256) {
        uint hw = idx / cg;
        uint dc = idx - hw * cg;
        float d = x[(ulong)hw * C + c0 + dc] - mean;
        vs += d * d;
    }
    float inv = rsq(group_sum256(red, vs) / (float)total + eps);

    for (uint idx = LID; idx < total; idx += 256) {
        uint hw = idx / cg;
        uint dc = idx - hw * cg;
        uint c = c0 + dc;
        float v = (x[(ulong)hw * C + c] - mean) * inv * w[c] + b[c];
        if (do_silu) v = silu_(v);
        out[(ulong)hw * C + c] = v;
    }
}

// 7. Two-pass LayerNorm
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void layernorm(global const float* x, global const float* w, global const float* b,
               global float* out, uint C, float eps) {
    local float red[256];
    ulong row = (ulong)GID * C;

    float sum = 0.0f;
    for (uint c = LID; c < C; c += 256) sum += x[row + c];
    float mean = group_sum256(red, sum) / (float)C;

    float vs = 0.0f;
    for (uint c = LID; c < C; c += 256) {
        float d = x[row + c] - mean;
        vs += d * d;
    }
    float inv = rsq(group_sum256(red, vs) / (float)C + eps);

    for (uint c = LID; c < C; c += 256)
        out[row + c] = (x[row + c] - mean) * inv * w[c] + b[c];
}

// 8. Broadcast add time embedding
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void add_temb(global float* h, global const float* temb, uint HW, uint C) {
    uint i = GID * 256 + LID;
    if (i < HW * C) h[i] += temb[i % C];
}

// 9. Channel concat NHWC
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void concat_c(global const float* a, global const float* b, global float* out, uint HW, uint Ca, uint Cb) {
    uint Ct = Ca + Cb;
    uint i = GID * 256 + LID;
    if (i < HW * Ct) {
        uint hw = i / Ct;
        uint c = i - hw * Ct;
        out[i] = (c < Ca) ? a[(ulong)hw * Ca + c] : b[(ulong)hw * Cb + (c - Ca)];
    }
}

// 10. Nearest-neighbor 2x spatial upsample
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void upsample2x(global const float* in, global float* out, uint H, uint W, uint C) {
    uint W2 = W << 1;
    ulong total = (ulong)(H << 1) * W2 * C;
    ulong i = (ulong)GID * 256 + LID;
    if (i < total) {
        uint c = (uint)(i % C);
        uint pix = (uint)(i / C);
        uint oy = pix / W2;
        uint ox = pix - oy * W2;
        out[i] = in[((ulong)(oy >> 1) * W + (ox >> 1)) * C + c];
    }
}

// 11. GEGLU
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void geglu(global const float* in, global float* out, uint M, uint D) {
    ulong i = (ulong)GID * 256 + LID;
    ulong total = (ulong)M * D;
    if (i < total) {
        uint m = (uint)(i / D);
        uint d = (uint)(i - (ulong)m * D);
        ulong base = (ulong)m * (D << 1);
        out[i] = in[base + d] * gelu_exact_(in[base + D + d]);
    }
}

// 12. In-place activation
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void act_inplace(global float* x, uint n, uint mode) {
    uint i = GID * 256 + LID;
    if (i < n) {
        float v = x[i];
        x[i] = (mode == 0u) ? silu_(v) : (mode == 1u ? quick_gelu_(v) : gelu_exact_(v));
    }
}

// 13. CLIP token + position embedding
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void clip_embed(global const half_t* tok_emb, global const half_t* pos_emb,
                global const uint* tokens, global float* out, uint seq_len, uint C) {
    uint i = GID * 256 + LID;
    if (i < seq_len * C) {
        uint s = i / C;
        uint c = i - s * C;
        uint tok = tokens[s];
        out[i] = (float)tok_emb[(ulong)tok * C + c] + (float)pos_emb[(ulong)s * C + c];
    }
}

// 14. FlashAttention-style LDS-tiled multi-head attention (D=64, 16-lane ds_swizzle)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void attn_d64(global const float* Q, global const float* K, global const float* V, global float* Out,
              uint M, uint S, uint H, float scale, uint causal) {
    local float Ks[16 * 64];
    local float Vs[16 * 64];

    uint tid = LID;
    uint q_local = tid >> 4;
    uint lane = tid & 15u;
    uint q_blocks = (M + 15u) >> 4;
    uint h = GID / q_blocks;
    uint m0 = (GID - h * q_blocks) << 4;
    uint m = m0 + q_local;
    uint stride4 = H << 4;
    uint hoff4 = h << 4;

    global const float4* Q4 = (global const float4*)Q;
    global const float4* K4 = (global const float4*)K;
    global const float4* V4 = (global const float4*)V;
    global float4* O4 = (global float4*)Out;

    float4 qv = (m < M) ? Q4[(ulong)m * stride4 + hoff4 + lane] : (float4)(0.0f);
    float4 ov = (float4)(0.0f);
    float mx = -1e30f, ls = 0.0f;

    uint s_lim = causal ? (m0 + 16u < S ? m0 + 16u : S) : S;
    for (uint s0 = 0; s0 < s_lim; s0 += 16u) {
        uint sk = s0 + q_local;
        float4 lk = (sk < S) ? K4[(ulong)sk * stride4 + hoff4 + lane] : (float4)(0.0f);
        float4 lv = (sk < S) ? V4[(ulong)sk * stride4 + hoff4 + lane] : (float4)(0.0f);
        uint lpos = (q_local << 6) + (lane << 2);
        Ks[lpos + 0] = lk.x; Ks[lpos + 1] = lk.y; Ks[lpos + 2] = lk.z; Ks[lpos + 3] = lk.w;
        Vs[lpos + 0] = lv.x; Vs[lpos + 1] = lv.y; Vs[lpos + 2] = lv.z; Vs[lpos + 3] = lv.w;
        BARRIER();

        uint bk = (s_lim - s0 < 16u) ? (s_lim - s0) : 16u;
        uint dpos = lane << 2;
        for (uint k = 0; k < bk; k++) {
            uint kbase = (k << 6) + dpos;
            float dot = qv.x * Ks[kbase + 0] + qv.y * Ks[kbase + 1] +
                        qv.z * Ks[kbase + 2] + qv.w * Ks[kbase + 3];
            float sc = sum16_wave(dot) * scale;
            if (!causal || (s0 + k) <= m) {
                float nm = __builtin_fmaxf(mx, sc);
                float a = exp_(mx - nm);
                float b = exp_(sc - nm);
                ov.x = ov.x * a + Vs[kbase + 0] * b;
                ov.y = ov.y * a + Vs[kbase + 1] * b;
                ov.z = ov.z * a + Vs[kbase + 2] * b;
                ov.w = ov.w * a + Vs[kbase + 3] * b;
                ls = ls * a + b;
                mx = nm;
            }
        }
        BARRIER();
    }

    if (m < M) {
        float inv_l = 1.0f / ls;
        O4[(ulong)m * stride4 + hoff4 + lane] = ov * inv_l;
    }
}

// 15a. Specialized zero-branch FlashAttention for SD 1.5 D=40 (32 queries x 8 lanes, 32x40 LDS tiles = 10 KiB!)
// groups = H * ((M + 31) / 32), threads = 256.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void attn_d40(global const float* Q, global const float* K, global const float* V, global float* Out,
              uint M, uint S, uint H, float scale) {
    local float Ks[32 * 40];
    local float Vs[32 * 40];

    uint tid = LID;
    uint q_local = tid >> 3;  // 0..31
    uint lane = tid & 7u;     // 0..7 (5 floats/lane = 40!)
    uint q_blocks = (M + 31u) >> 5;
    uint h = GID / q_blocks;
    uint m0 = (GID - h * q_blocks) << 5;
    uint m = m0 + q_local;
    uint stride = H * 40u;
    uint hoff = h * 40u;

    ulong qbase = (ulong)m * stride + hoff + lane;
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f, q4 = 0.0f;
    if (m < M) {
        q0 = Q[qbase +  0]; q1 = Q[qbase +  8]; q2 = Q[qbase + 16]; q3 = Q[qbase + 24]; q4 = Q[qbase + 32];
    }
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f, o4 = 0.0f;
    float mx = -1e30f, ls = 0.0f;

    uint ldst = q_local * 40u + lane;
    for (uint s0 = 0; s0 < S; s0 += 32u) {
        uint sk = s0 + q_local;
        if (sk < S) {
            ulong sbase = (ulong)sk * stride + hoff + lane;
            Ks[ldst +  0] = K[sbase +  0]; Ks[ldst +  8] = K[sbase +  8]; Ks[ldst + 16] = K[sbase + 16];
            Ks[ldst + 24] = K[sbase + 24]; Ks[ldst + 32] = K[sbase + 32];
            Vs[ldst +  0] = V[sbase +  0]; Vs[ldst +  8] = V[sbase +  8]; Vs[ldst + 16] = V[sbase + 16];
            Vs[ldst + 24] = V[sbase + 24]; Vs[ldst + 32] = V[sbase + 32];
        } else {
            Ks[ldst + 0] = 0; Ks[ldst + 8] = 0; Ks[ldst + 16] = 0; Ks[ldst + 24] = 0; Ks[ldst + 32] = 0;
            Vs[ldst + 0] = 0; Vs[ldst + 8] = 0; Vs[ldst + 16] = 0; Vs[ldst + 24] = 0; Vs[ldst + 32] = 0;
        }
        BARRIER();

        uint bk = (S - s0 < 32u) ? (S - s0) : 32u;
        #pragma unroll 4
        for (uint k = 0; k < bk; k++) {
            uint kb = k * 40u + lane;
            float dot = q0 * Ks[kb + 0] + q1 * Ks[kb + 8] + q2 * Ks[kb + 16] + q3 * Ks[kb + 24] + q4 * Ks[kb + 32];
            float sc = sum8_wave(dot) * scale;
            float nm = __builtin_fmaxf(mx, sc);
            float a = exp_(mx - nm);
            float b = exp_(sc - nm);
            o0 = o0 * a + Vs[kb +  0] * b;
            o1 = o1 * a + Vs[kb +  8] * b;
            o2 = o2 * a + Vs[kb + 16] * b;
            o3 = o3 * a + Vs[kb + 24] * b;
            o4 = o4 * a + Vs[kb + 32] * b;
            ls = ls * a + b;
            mx = nm;
        }
        BARRIER();
    }

    if (m < M) {
        float inv_l = 1.0f / ls;
        Out[qbase +  0] = o0 * inv_l; Out[qbase +  8] = o1 * inv_l; Out[qbase + 16] = o2 * inv_l;
        Out[qbase + 24] = o3 * inv_l; Out[qbase + 32] = o4 * inv_l;
    }
}

// 15b. Specialized zero-branch FlashAttention for SD 1.5 D=80 (16 queries x 16 lanes, 16x80 LDS = 10 KiB!)
// groups = H * ((M + 15) / 16), threads = 256.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void attn_d80(global const float* Q, global const float* K, global const float* V, global float* Out,
              uint M, uint S, uint H, float scale) {
    local float Ks[16 * 80];
    local float Vs[16 * 80];

    uint tid = LID;
    uint q_local = tid >> 4;
    uint lane = tid & 15u;
    uint q_blocks = (M + 15u) >> 4;
    uint h = GID / q_blocks;
    uint m0 = (GID - h * q_blocks) << 4;
    uint m = m0 + q_local;
    uint stride = H * 80u;
    uint hoff = h * 80u;

    ulong qbase = (ulong)m * stride + hoff + lane;
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f, q4 = 0.0f;
    if (m < M) {
        q0 = Q[qbase +  0]; q1 = Q[qbase + 16]; q2 = Q[qbase + 32]; q3 = Q[qbase + 48]; q4 = Q[qbase + 64];
    }
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f, o4 = 0.0f;
    float mx = -1e30f, ls = 0.0f;

    uint ldst = q_local * 80u + lane;
    for (uint s0 = 0; s0 < S; s0 += 16u) {
        uint sk = s0 + q_local;
        if (sk < S) {
            ulong sbase = (ulong)sk * stride + hoff + lane;
            Ks[ldst +  0] = K[sbase +  0]; Ks[ldst + 16] = K[sbase + 16]; Ks[ldst + 32] = K[sbase + 32];
            Ks[ldst + 48] = K[sbase + 48]; Ks[ldst + 64] = K[sbase + 64];
            Vs[ldst +  0] = V[sbase +  0]; Vs[ldst + 16] = V[sbase + 16]; Vs[ldst + 32] = V[sbase + 32];
            Vs[ldst + 48] = V[sbase + 48]; Vs[ldst + 64] = V[sbase + 64];
        } else {
            Ks[ldst + 0] = 0; Ks[ldst + 16] = 0; Ks[ldst + 32] = 0; Ks[ldst + 48] = 0; Ks[ldst + 64] = 0;
            Vs[ldst + 0] = 0; Vs[ldst + 16] = 0; Vs[ldst + 32] = 0; Vs[ldst + 48] = 0; Vs[ldst + 64] = 0;
        }
        BARRIER();

        uint bk = (S - s0 < 16u) ? (S - s0) : 16u;
        #pragma unroll 4
        for (uint k = 0; k < bk; k++) {
            uint kb = k * 80u + lane;
            float dot = q0 * Ks[kb + 0] + q1 * Ks[kb + 16] + q2 * Ks[kb + 32] + q3 * Ks[kb + 48] + q4 * Ks[kb + 64];
            float sc = sum16_wave(dot) * scale;
            float nm = __builtin_fmaxf(mx, sc);
            float a = exp_(mx - nm);
            float b = exp_(sc - nm);
            o0 = o0 * a + Vs[kb +  0] * b;
            o1 = o1 * a + Vs[kb + 16] * b;
            o2 = o2 * a + Vs[kb + 32] * b;
            o3 = o3 * a + Vs[kb + 48] * b;
            o4 = o4 * a + Vs[kb + 64] * b;
            ls = ls * a + b;
            mx = nm;
        }
        BARRIER();
    }

    if (m < M) {
        float inv_l = 1.0f / ls;
        Out[qbase +  0] = o0 * inv_l; Out[qbase + 16] = o1 * inv_l; Out[qbase + 32] = o2 * inv_l;
        Out[qbase + 48] = o3 * inv_l; Out[qbase + 64] = o4 * inv_l;
    }
}

// 15c. Specialized zero-branch FlashAttention for SD 1.5 D=160 (16 queries x 16 lanes, 16x160 LDS = 20 KiB!)
// groups = H * ((M + 15) / 16), threads = 256.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void attn_d160(global const float* Q, global const float* K, global const float* V, global float* Out,
               uint M, uint S, uint H, float scale) {
    local float Ks[16 * 160];
    local float Vs[16 * 160];

    uint tid = LID;
    uint q_local = tid >> 4;
    uint lane = tid & 15u;
    uint q_blocks = (M + 15u) >> 4;
    uint h = GID / q_blocks;
    uint m0 = (GID - h * q_blocks) << 4;
    uint m = m0 + q_local;
    uint stride = H * 160u;
    uint hoff = h * 160u;

    ulong qbase = (ulong)m * stride + hoff + lane;
    float q0=0,q1=0,q2=0,q3=0,q4=0,q5=0,q6=0,q7=0,q8=0,q9=0;
    if (m < M) {
        q0=Q[qbase+  0]; q1=Q[qbase+ 16]; q2=Q[qbase+ 32]; q3=Q[qbase+ 48]; q4=Q[qbase+ 64];
        q5=Q[qbase+ 80]; q6=Q[qbase+ 96]; q7=Q[qbase+112]; q8=Q[qbase+128]; q9=Q[qbase+144];
    }
    float o0=0,o1=0,o2=0,o3=0,o4=0,o5=0,o6=0,o7=0,o8=0,o9=0;
    float mx = -1e30f, ls = 0.0f;

    uint ldst = q_local * 160u + lane;
    for (uint s0 = 0; s0 < S; s0 += 16u) {
        uint sk = s0 + q_local;
        if (sk < S) {
            ulong sbase = (ulong)sk * stride + hoff + lane;
            #pragma unroll
            for (uint j = 0; j < 10; j++) {
                Ks[ldst + (j << 4)] = K[sbase + (j << 4)];
                Vs[ldst + (j << 4)] = V[sbase + (j << 4)];
            }
        } else {
            #pragma unroll
            for (uint j = 0; j < 10; j++) { Ks[ldst + (j << 4)] = 0.0f; Vs[ldst + (j << 4)] = 0.0f; }
        }
        BARRIER();

        uint bk = (S - s0 < 16u) ? (S - s0) : 16u;
        #pragma unroll 2
        for (uint k = 0; k < bk; k++) {
            uint kb = k * 160u + lane;
            float dot = q0*Ks[kb+  0] + q1*Ks[kb+ 16] + q2*Ks[kb+ 32] + q3*Ks[kb+ 48] + q4*Ks[kb+ 64]
                      + q5*Ks[kb+ 80] + q6*Ks[kb+ 96] + q7*Ks[kb+112] + q8*Ks[kb+128] + q9*Ks[kb+144];
            float sc = sum16_wave(dot) * scale;
            float nm = __builtin_fmaxf(mx, sc);
            float a = exp_(mx - nm);
            float b = exp_(sc - nm);
            o0=o0*a+Vs[kb+  0]*b; o1=o1*a+Vs[kb+ 16]*b; o2=o2*a+Vs[kb+ 32]*b; o3=o3*a+Vs[kb+ 48]*b; o4=o4*a+Vs[kb+ 64]*b;
            o5=o5*a+Vs[kb+ 80]*b; o6=o6*a+Vs[kb+ 96]*b; o7=o7*a+Vs[kb+112]*b; o8=o8*a+Vs[kb+128]*b; o9=o9*a+Vs[kb+144]*b;
            ls = ls * a + b;
            mx = nm;
        }
        BARRIER();
    }

    if (m < M) {
        float inv_l = 1.0f / ls;
        Out[qbase+  0]=o0*inv_l; Out[qbase+ 16]=o1*inv_l; Out[qbase+ 32]=o2*inv_l; Out[qbase+ 48]=o3*inv_l; Out[qbase+ 64]=o4*inv_l;
        Out[qbase+ 80]=o5*inv_l; Out[qbase+ 96]=o6*inv_l; Out[qbase+112]=o7*inv_l; Out[qbase+128]=o8*inv_l; Out[qbase+144]=o9*inv_l;
    }
}

// 16. VAE mid-block attention (D=512, H=1)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void attn_d512(global const float* Q, global const float* K, global const float* V, global float* Out,
               uint M, float scale) {
    local float red[256];
    uint tid = LID;
    ulong qoff = (ulong)GID * 512u;
    float qa = Q[qoff + tid];
    float qb = Q[qoff + tid + 256u];
    float oa = 0.0f, ob = 0.0f, mx = -1e30f, ls = 0.0f;

    for (uint s = 0; s < M; s++) {
        ulong soff = (ulong)s * 512u;
        float dot = qa * K[soff + tid] + qb * K[soff + tid + 256u];
        float sc = group_sum256(red, dot) * scale;
        float nm = __builtin_fmaxf(mx, sc);
        float a = exp_(mx - nm), b = exp_(sc - nm);
        oa = oa * a + V[soff + tid] * b;
        ob = ob * a + V[soff + tid + 256u] * b;
        ls = ls * a + b;
        mx = nm;
    }
    Out[qoff + tid] = oa / ls;
    Out[qoff + tid + 256u] = ob / ls;
}

// 17. Per-channel PReLU in-place (NHWC) for Real-ESRGAN Anime v3 upscaler
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void prelu_inplace(global float* x, global const float* w, uint HW, uint C) {
    ulong i = (ulong)GID * 256 + LID;
    ulong total = (ulong)HW * C;
    if (i < total) {
        float v = x[i];
        if (v < 0.0f) x[i] = v * w[(uint)(i % C)];
    }
}
