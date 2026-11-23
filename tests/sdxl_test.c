// Host CPU verification test for Garlic Diffusion (CLIP tokenizer, PNG writer, optional full SDXL run).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clip_tok.h"
#include "gallery.h"
#include "model_dl.h"
#include "png_write.h"
#include "sdxl.h"

static void on_prog(void* user, float f, const char* msg) {
    (void)user;
    printf("  [%5.1f%%] %s\n", f * 100.0f, msg);
    fflush(stdout);
}

int main(int argc, char** argv) {
    ClipTok tok;
    if (clip_tok_load(&tok, "diffusion/assets/clip_tok.bin")) {
        fprintf(stderr, "FAIL: clip_tok_load\n");
        return 1;
    }
    uint32_t ids_l[CLIP_MAX_LEN], ids_g[CLIP_MAX_LEN];
    const char* prompt = "1girl, silver hair, blue eyes, masterpiece, best quality";
    uint32_t eos_l = clip_tok_encode(&tok, prompt, ids_l, CLIP_EOS);
    uint32_t eos_g = clip_tok_encode(&tok, prompt, ids_g, 0);
    printf("CLIP-L eos=%u ids[0..15]:", eos_l);
    for (int i = 0; i < 16; i++) printf(" %u", ids_l[i]);
    printf("\nCLIP-G eos=%u ids[eos..eos+2]: %u %u %u\n", eos_g, ids_g[eos_g], ids_g[eos_g + 1], ids_g[eos_g + 2]);
    if (ids_l[0] != 49406 || ids_l[eos_l] != 49407 || ids_l[eos_l + 1] != 49407 || ids_g[eos_g + 1] != 0) {
        fprintf(stderr, "FAIL: bad BOS/EOS/PAD\n");
        return 1;
    }
    clip_tok_free(&tok);

    // Test PNG writer with A1111 tEXt parameters chunk (64x64 gradient)
    uint8_t rgb[64 * 64 * 3];
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) {
            rgb[(y * 64 + x) * 3 + 0] = (uint8_t)(x * 4);
            rgb[(y * 64 + x) * 3 + 1] = (uint8_t)(y * 4);
            rgb[(y * 64 + x) * 3 + 2] = 180;
        }
    const char* a1111_meta = "1girl, silver hair\nNegative prompt: blurry\nSteps: 6, Sampler: LCM, CFG scale: 1.0, Seed: 4242, Size: 64x64, Model: WAI_v17";
    size_t png_sz = 0;
    uint8_t* png = png_encode_rgb_meta(rgb, 64, 64, a1111_meta, &png_sz);
    if (!png || png_sz < 100) {
        fprintf(stderr, "FAIL: png_encode_rgb_meta\n");
        return 1;
    }
    char meta_out[512] = {0};
    if (png_extract_parameters(png, png_sz, meta_out, sizeof(meta_out)) == 0 || strcmp(meta_out, a1111_meta) != 0) {
        fprintf(stderr, "FAIL: png_extract_parameters mismatch: got '%s'\n", meta_out);
        return 1;
    }
    printf("A1111 PNG tEXt parameters chunk PASS!\n");
    FILE* f = fopen("build/test_grad.png", "wb");
    fwrite(png, 1, png_sz, f);
    fclose(f);

    // Verify stored-DEFLATE PNG roundtrip + rgb_resize + gallery/preset persistence
    uint32_t dw = 0, dh = 0;
    uint8_t* dec = png_decode_rgb(png, png_sz, &dw, &dh);
    if (!dec || dw != 64 || dh != 64 || memcmp(dec, rgb, sizeof(rgb)) != 0) {
        fprintf(stderr, "FAIL: png_decode_rgb roundtrip\n");
        return 1;
    }
    uint8_t* rsz = rgb_resize(dec, dw, dh, 128, 96);
    if (!rsz) { fprintf(stderr, "FAIL: rgb_resize\n"); return 1; }
    free(rsz); free(dec);

    gallery_init("build/test_gal_out", "build/test_gal_presets");
    GalleryItem gm = {0};
    snprintf(gm.model, sizeof(gm.model), "MeinaMix_V11");
    snprintf(gm.prompt, sizeof(gm.prompt), "%s", prompt);
    snprintf(gm.sampler, sizeof(gm.sampler), "Euler");
    gm.width = 64; gm.height = 64; gm.steps = 8; gm.cfg = 3.5f; gm.seed = 4242;
    char gid[64] = {0};
    if (gallery_save(png, png_sz, &gm, gid, sizeof(gid)) != 0) {
        fprintf(stderr, "FAIL: gallery_save\n"); return 1;
    }
    // Verify legacy img_* sorts AFTER new g_NNNNN_* in Gallery!
    {
        FILE* lf = fopen("build/test_gal_out/img_999999.png", "wb");
        fwrite(png, 1, png_sz, lf);
        fclose(lf);
    }
    GalleryItem list[8];
    int ng = gallery_list(list, 8);
    if (ng < 2 || strcmp(list[0].id, gid) != 0 || list[0].width != 64) {
        fprintf(stderr, "FAIL: gallery_list sort order (ng=%d first=%s expected=%s)\n", ng, ng ? list[0].id : "", gid);
        return 1;
    }
    gallery_delete(gid);
    gallery_delete("img_999999");
    UserPreset up[16];
    int np = presets_list(up, 16);
    if (np < 5) { fprintf(stderr, "FAIL: default presets seed (np=%d)\n", np); return 1; }

    styles_init("build/test_gal_styles");
    StyleItem st[16];
    int ns = styles_list(st, 16);
    if (ns < 6) { fprintf(stderr, "FAIL: default styles seed (ns=%d)\n", ns); return 1; }
    model_dl_init("diffusion/models", "build/test_repo.cfg");
    model_dl_set_repo("https://huggingface.co/testuser/garlic-diffusion-models/");
    char rchk[128] = {0};
    model_dl_get_repo(rchk, sizeof(rchk));
    if (strcmp(rchk, "testuser/garlic-diffusion-models") != 0) {
        fprintf(stderr, "FAIL: model_dl_set_repo normalize (got '%s')\n", rchk);
        return 1;
    }
    if (model_dl_refresh_catalog() != 1) {
        fprintf(stderr, "FAIL: model_dl manifest parse\n");
        return 1;
    }
    // Regression test: sdxl_init with NULL model_dir (GPU-only startup phase before model fallback)
    {
        SdxlEngine e0;
        if (sdxl_init(&e0, "diffusion/assets", NULL, NULL, NULL) != 0) {
            fprintf(stderr, "FAIL: sdxl_init(NULL model_dir): %s\n", e0.error);
            return 1;
        }
        sdxl_free(&e0);
    }
    printf("Styles (%d seeded) + Gallery newest-first sort + Model Manager + NULL-model init PASS!\n", ns);

    // Verify fast PSD1 completeness validator catches truncated/incomplete models cleanly
    {
        FILE* tf = fopen("build/test_trunc.psd1", "wb");
        uint8_t hdr[24] = {'P','S','D','1'};
        uint32_t nt = 1;
        uint64_t dsz = 1048576ull; // claims 1 MiB payload, write only 512 bytes
        memcpy(hdr + 4, &nt, 4);
        memcpy(hdr + 8, &dsz, 8);
        fwrite(hdr, 1, 24, tf);
        uint8_t pad[512] = {0};
        fwrite(pad, 1, sizeof(pad), tf);
        fclose(tf);
        uint64_t act = 0, exp = 0;
        int ck = sdxl_check_pack("build/test_trunc.psd1", &act, &exp);
        if (ck != -3 || exp != dsz || act >= dsz) {
            fprintf(stderr, "FAIL: sdxl_check_pack truncated (rc=%d act=%llu exp=%llu)\n",
                    ck, (unsigned long long)act, (unsigned long long)exp);
            return 1;
        }
        int ck_ok = sdxl_check_pack("diffusion/assets/upscaler.psd1", &act, &exp);
        if (ck_ok != 0 || act < exp) {
            fprintf(stderr, "FAIL: sdxl_check_pack valid upscaler (rc=%d)\n", ck_ok);
            return 1;
        }
        printf("PSD1 completeness validator PASS (truncated=-3, upscaler=0)!\n");
    }
    printf("Gallery + PNG decode + rgb_resize + %d seeded presets PASS!\n", np);

    free(png);
    printf("Wrote build/test_grad.png (%zu bytes)\n", png_sz);

    // Optional full SDXL CPU smoke test (e.g. --run 256 256 1)
    if (argc >= 2 && strcmp(argv[1], "--run") == 0) {
        uint32_t w = argc >= 3 ? (uint32_t)atoi(argv[2]) : 256;
        uint32_t h = argc >= 4 ? (uint32_t)atoi(argv[3]) : 256;
        uint32_t steps = argc >= 5 ? (uint32_t)atoi(argv[4]) : 1;
        SdxlEngine eng;
        if (sdxl_init(&eng, "diffusion/assets", "diffusion/models/waiNSFWIllustrious_v140", on_prog, NULL)) {
            fprintf(stderr, "FAIL sdxl_init: %s\n", eng.error);
            return 1;
        }
        uint8_t* img = (uint8_t*)malloc((size_t)w * h * 3);
        SdxlParams p = { prompt, "lowres, bad anatomy", w, h, steps, 7.0f, 12345ull };
        if (sdxl_generate(&eng, &p, img, on_prog, NULL, NULL)) {
            fprintf(stderr, "FAIL sdxl_generate: %s\n", eng.error);
            return 1;
        }
        uint8_t* opng = png_encode_rgb(img, w, h, &png_sz);
        FILE* of = fopen("build/test_sdxl.png", "wb");
        fwrite(opng, 1, png_sz, of);
        fclose(of);
        free(opng);
        free(img);
        sdxl_free(&eng);
        printf("PASS: generated build/test_sdxl.png (%zu bytes)\n", png_sz);
    }
    printf("ALL HOST CHECKS PASSED\n");
    return 0;
}
