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
    UserPreset* up = (UserPreset*)malloc(sizeof(UserPreset) * PRESETS_MAX_ITEMS);
    int np = presets_list(up, PRESETS_MAX_ITEMS);
    if (np < 70) { fprintf(stderr, "FAIL: default presets seed (np=%d)\n", np); return 1; }
    free(up);

    styles_init("build/test_gal_styles");
    StyleItem st[STYLES_MAX_ITEMS];
    int ns = styles_list(st, STYLES_MAX_ITEMS);
    if (ns < 20) { fprintf(stderr, "FAIL: default styles seed (ns=%d)\n", ns); return 1; }

    characters_init("build/test_gal_chars");
    CharacterItem* ch = (CharacterItem*)malloc(sizeof(CharacterItem) * CHARS_MAX_ITEMS);
    int nc = characters_list(ch, CHARS_MAX_ITEMS);
    free(ch);
    if (nc < 240) { fprintf(stderr, "FAIL: default characters seed (nc=%d, expected >=240)\n", nc); return 1; }
    CharacterItem custom_c = { "test_custom_char", "Test Hero", "Custom", "1boy, test hero", "blurry" };
    if (characters_save(&custom_c, NULL, 0) != 0 || characters_delete("test_custom_char") != 0) {
        fprintf(stderr, "FAIL: characters_save/delete roundtrip\n");
        return 1;
    }
    printf("Seeded Presets (%d) + Styles (%d) + Characters (%d) PASS!\n", np, ns, nc);

    // Verify 1024x1344 Illustrious preset auto-detection + Speed preset + Dupe cleaner + Wildcards + Folders + ZIP
    {
        ModelPreset mp, sp;
        sdxl_load_preset("build/nonexistent_dir", "Hassaku_XL_v31_dmd2", 0, &mp);
        if (mp.width != 1024 || mp.height != 1344 || mp.steps != 4 || fabsf(mp.cfg - 1.0f) > 0.05f || mp.sampler != 2) {
            fprintf(stderr, "FAIL: sdxl_load_preset Illustrious 1024x1344 (got %ux%u %u steps cfg=%.2f smp=%d)\n",
                    mp.width, mp.height, mp.steps, (double)mp.cfg, mp.sampler);
            return 1;
        }
        sdxl_load_preset("build/nonexistent_dir", "Animagine_XL_4_dmd2", 0, &mp);
        if (mp.width != 1024 || mp.height != 1344 || mp.steps != 6 || fabsf(mp.cfg - 1.4f) > 0.05f || mp.sampler != 2 || strncmp(mp.prompt, "1girl,", 6) != 0) {
            fprintf(stderr, "FAIL: sdxl_load_preset Animagine_XL_4_dmd2 (got %ux%u %u steps cfg=%.2f smp=%d prompt='%s')\n",
                    mp.width, mp.height, mp.steps, (double)mp.cfg, mp.sampler, mp.prompt);
            return 1;
        }
        sdxl_load_preset("build/nonexistent_dir", "NoobAI_XL_1.1_dmd2", 0, &mp);
        if (mp.width != 1024 || mp.height != 1344 || mp.steps != 4 || fabsf(mp.cfg - 1.0f) > 0.05f || mp.sampler != 2 || strstr(mp.prompt, "newest") == NULL) {
            fprintf(stderr, "FAIL: sdxl_load_preset NoobAI_XL_1.1_dmd2 (got %ux%u %u steps cfg=%.2f smp=%d)\n",
                    mp.width, mp.height, mp.steps, (double)mp.cfg, mp.sampler);
            return 1;
        }
        sdxl_load_preset("build/nonexistent_dir", "Counterfeit_V25", 1, &mp);
        if (mp.width != 512 || mp.height != 512 || mp.steps != 20 || mp.sampler != 3 || mp.upscale != 1) {
            fprintf(stderr, "FAIL: sdxl_load_preset Counterfeit_V25\n");
            return 1;
        }
        sdxl_load_speed_preset("WAI_Illustrious_v17_dmd2", 0, &sp);
        if (sp.steps != 5 || sp.sampler != 2) {
            fprintf(stderr, "FAIL: sdxl_load_speed_preset DMD2\n");
            return 1;
        }

        // Verify prompt dupe cleaner
        char dedup[512] = {0};
        sdxl_clean_prompt_dupes("masterpiece, 1girl, Hatsune Miku, hatsune_miku, (blue eyes:1.2), blue eyes, <lora:miku:0.75>, <lora:miku:0.75>, solo",
                                dedup, sizeof(dedup));
        if (strstr(dedup, "(blue eyes:1.2)") == NULL || strstr(dedup, "hatsune_miku") != NULL) {
            fprintf(stderr, "FAIL: sdxl_clean_prompt_dupes got '%s'\n", dedup);
            return 1;
        }

        // Verify inline {a|b|c} + 500-word SFW & NSFW randomizer + SFW guard
        char exp1[1024] = {0}, exp2[1024] = {0}, ng_sfw[512] = "blurry", ng_nsfw[512] = "blurry";
        sdxl_expand_wildcards("1girl, {red dress|blue dress}, __scene__", 11111ull, 0, 1u | 2u, exp1, sizeof(exp1));
        sdxl_expand_wildcards("1girl, adult woman, __ecchi__", 22222ull, 1, 128u | 256u, exp2, sizeof(exp2));
        if (strchr(exp1, '{') != NULL || strstr(exp1, "__scene__") != NULL || strlen(exp2) < 25) {
            fprintf(stderr, "FAIL: sdxl_expand_wildcards exp1='%s' exp2='%s'\n", exp1, exp2);
            return 1;
        }
        sdxl_apply_quality_shield("WAI_Illustrious_v17_dmd2", 0, 1, 0, exp1, sizeof(exp1), ng_sfw, sizeof(ng_sfw));
        sdxl_apply_quality_shield("WAI_Illustrious_v17_dmd2", 0, 1, 1, exp2, sizeof(exp2), ng_nsfw, sizeof(ng_nsfw));
        if (strstr(ng_sfw, "nsfw") == NULL || strstr(ng_nsfw, "nsfw") != NULL || strstr(exp1, "masterpiece") == NULL) {
            fprintf(stderr, "FAIL: sdxl_apply_quality_shield sfw='%s' nsfw='%s'\n", ng_sfw, ng_nsfw);
            return 1;
        }
        // Verify Animagine XL 4.0 appends quality tags at the END (character/framing first)
        char anim_p[512] = "1girl, hatsune miku, full body", anim_n[512] = "blurry";
        sdxl_apply_quality_shield("Animagine_XL_4_dmd2", 0, 1, 0, anim_p, sizeof(anim_p), anim_n, sizeof(anim_n));
        if (strncmp(anim_p, "1girl, hatsune miku", 19) != 0 || strstr(anim_p, "high score") == NULL) {
            fprintf(stderr, "FAIL: sdxl_apply_quality_shield Animagine tag order got '%s'\n", anim_p);
            return 1;
        }
        // Verify SD 1.5 Quality Shield injects built-in EasyNegative TI trigger even on presets that already have bad hands/extra digits
        char sd15_p[512] = "1girl, silver hair", sd15_n[512] = "lowres, bad hands, extra digits, blurry";
        sdxl_apply_quality_shield("MeinaMix_V11", 1, 1, 0, sd15_p, sizeof(sd15_p), sd15_n, sizeof(sd15_n));
        if (strstr(sd15_n, "EasyNegative") == NULL || strstr(anim_n, "EasyNegative") != NULL) {
            fprintf(stderr, "FAIL: EasyNegative TI shield check (sd15='%s', sdxl='%s')\n", sd15_n, anim_n);
            return 1;
        }

        // Verify Gallery Folders, Starred Favorites, and Stored-PKZIP archive builder
        gallery_init("build/test_gal_folders", "build/test_gal_presets");
        char cf[64] = {0}, id1[64] = {0}, id2[64] = {0};
        if (gallery_create_folder("Miku Batch 01!", cf, sizeof(cf)) != 0 || strcmp(cf, "Miku Batch 01") != 0) {
            fprintf(stderr, "FAIL: gallery_create_folder got '%s'\n", cf);
            return 1;
        }
        GalleryItem fm = {0};
        snprintf(fm.model, sizeof(fm.model), "WAI_v17");
        snprintf(fm.prompt, sizeof(fm.prompt), "1girl, hatsune miku");
        snprintf(fm.folder, sizeof(fm.folder), "%s", cf);
        fm.width = 64; fm.height = 64; fm.steps = 6; fm.cfg = 1.4f; fm.seed = 777;
        gallery_save(png, png_sz, &fm, id1, sizeof(id1));
        fm.folder[0] = 0; fm.seed = 888;
        gallery_save(png, png_sz, &fm, id2, sizeof(id2));
        gallery_set_starred(id2, 1);

        GalleryFolder fls[8];
        int uns = 0, star = 0, all = 0;
        int nfl = gallery_list_folders(fls, 8, &uns, &star, &all);
        if (nfl < 1 || star < 1 || all < 2) {
            fprintf(stderr, "FAIL: gallery_list_folders nfl=%d star=%d all=%d\n", nfl, star, all);
            return 1;
        }
        char mv_ids[1][64];
        snprintf(mv_ids[0], 64, "%s", id2);
        if (gallery_set_folder_batch(mv_ids, 1, cf) != 1) {
            fprintf(stderr, "FAIL: gallery_set_folder_batch\n");
            return 1;
        }
        size_t zsz = 0;
        uint8_t* zbuf = gallery_build_zip(NULL, 0, cf, &zsz);
        if (!zbuf || zsz <= png_sz * 2 || memcmp(zbuf, "PK\x03\x04", 4) != 0) {
            fprintf(stderr, "FAIL: gallery_build_zip zsz=%zu\n", zsz);
            return 1;
        }
        free(zbuf);
        gallery_delete_folder(cf, 1);
        printf("v1.14 Presets + Dupe Cleaner + 500-Word Wildcards + Shield + Folders + PKZIP PASS!\n");
    }
    model_dl_init("diffusion/models", "build/test_repo.cfg");
    model_dl_set_repo("https://huggingface.co/testuser/garlic-diffusion-models/");
    char rchk[128] = {0};
    model_dl_get_repo(rchk, sizeof(rchk));
    if (strcmp(rchk, "testuser/garlic-diffusion-models") != 0) {
        fprintf(stderr, "FAIL: model_dl_set_repo normalize (got '%s')\n", rchk);
        return 1;
    }
    if (model_dl_refresh_catalog() != 1 || lora_refresh_catalog() != 1) {
        fprintf(stderr, "FAIL: model_dl / lora_dl manifest parse\n");
        return 1;
    }
    if (!app_version_is_newer("1.15", "1.16") || !app_version_is_newer("01.14", "1.15") ||
        app_version_is_newer("1.15", "1.15") || app_version_is_newer("1.15", "1.14") ||
        !app_version_is_newer("1.15", "v2.0.0")) {
        fprintf(stderr, "FAIL: app_version_is_newer\n");
        return 1;
    }
    if (app_update_check_now() != 0) {
        fprintf(stderr, "FAIL: app_update_check_now host stub\n");
        return 1;
    }
    {
        AppUpdateStatus us;
        app_update_get_status(&us);
        if (!us.checked || !us.update_available || strcmp(us.current_ver, GD_APP_VERSION) != 0) {
            fprintf(stderr, "FAIL: app_update_get_status (checked=%d avail=%d cur=%s rem=%s)\n",
                    us.checked, us.update_available, us.current_ver, us.remote_ver);
            return 1;
        }
        printf("v1.15 Obfuscated Seeds/Pools + OTA Version Checker (%s -> %s) PASS!\n", us.current_ver, us.remote_ver);
    }

    // Verify >128 gallery items sort newest-first without truncation + thumbnail generation!
    {
        gallery_init("build/test_gal_big", "build/test_gal_presets");
        char last_saved[64] = {0};
        for (int i = 0; i < 150; i++) {
            GalleryItem bm = {0};
            snprintf(bm.model, sizeof(bm.model), "TestModel");
            snprintf(bm.prompt, sizeof(bm.prompt), "item_%d", i);
            snprintf(bm.sampler, sizeof(bm.sampler), "LCM");
            bm.width = 64; bm.height = 64; bm.steps = 6; bm.cfg = 1.4f; bm.seed = (uint64_t)(1000 + i);
            if (gallery_save_ex(png, png_sz, (i == 149) ? rgb : NULL, 64, 64, &bm, last_saved, sizeof(last_saved)) != 0) {
                fprintf(stderr, "FAIL: gallery_save_ex %d\n", i);
                return 1;
            }
        }
        GalleryItem* big = (GalleryItem*)malloc(sizeof(GalleryItem) * GALLERY_MAX_ITEMS);
        int nbig = gallery_list(big, GALLERY_MAX_ITEMS);
        if (nbig < 150 || strcmp(big[0].id, last_saved) != 0) {
            fprintf(stderr, "FAIL: >128 gallery_list (nbig=%d, first=%s, expected=%s)\n",
                    nbig, nbig > 0 ? big[0].id : "", last_saved);
            return 1;
        }
        // Verify server-side gallery_list_page pagination + search query!
        GalleryItem pg[24];
        int pg_tot = 0;
        int np1 = gallery_list_page(pg, 0, 24, NULL, &pg_tot);
        int np2 = gallery_list_page(pg, 24, 24, NULL, &pg_tot);
        if (np1 != 24 || np2 != 24 || pg_tot != nbig || strcmp(pg[0].id, big[24].id) != 0) {
            fprintf(stderr, "FAIL: gallery_list_page pagination (np1=%d np2=%d tot=%d)\n", np1, np2, pg_tot);
            return 1;
        }
        int q_tot = 0;
        int nq = gallery_list_page(pg, 0, 24, "item_149", &q_tot);
        if (nq != 1 || q_tot != 1 || strcmp(pg[0].id, last_saved) != 0) {
            fprintf(stderr, "FAIL: gallery_list_page search query (nq=%d q_tot=%d)\n", nq, q_tot);
            return 1;
        }
        size_t tsz = 0;
        uint8_t* thumb = gallery_load_thumb_png(last_saved, &tsz);
        uint32_t tw = 0, th = 0;
        if (!thumb || png_probe_dims(thumb, tsz, &tw, &th) != 0 || tw != 64 || th != 64) {
            fprintf(stderr, "FAIL: gallery_load_thumb_png\n");
            return 1;
        }
        free(thumb);
        for (int i = 0; i < nbig; i++) gallery_delete(big[i].id);
        free(big);
        printf(">128 Gallery sort-first (%d items) + Thumbnail cache PASS!\n", nbig);
    }

    // Verify <lora:name:weight> prompt extraction & stripping
    {
        char cpr[256] = {0}, lspec[256] = {0};
        sdxl_extract_loras("1girl, <lora:hifumi-newgame-01:0.85>, smile, <lora:add_detail:0.5>",
                           "takimoto_hifumi", 0.70f, cpr, sizeof(cpr), lspec, sizeof(lspec));
        if (strstr(cpr, "<lora:") != NULL || strcmp(lspec, "takimoto_hifumi:0.70,hifumi-newgame-01:0.85,add_detail:0.50") != 0) {
            fprintf(stderr, "FAIL: sdxl_extract_loras cpr='%s' spec='%s'\n", cpr, lspec);
            return 1;
        }
        printf("LoRA prompt tag extraction & stacking ('%s') PASS!\n", lspec);
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
    // Verify each individual Batch Word Randomizer bitflag produces distinct category tags!
    {
        char r_pose[512], r_expr[512], r_scene[512], r_light[512], r_outf[512], r_det[512], r_cam[512], r_combo[512];
        sdxl_expand_wildcards("1girl", 12345ull, 0, 1u << 0, r_pose, sizeof(r_pose));
        sdxl_expand_wildcards("1girl", 12345ull, 0, 1u << 1, r_expr, sizeof(r_expr));
        sdxl_expand_wildcards("1girl", 12345ull, 0, 1u << 2, r_scene, sizeof(r_scene));
        sdxl_expand_wildcards("1girl", 12345ull, 0, 1u << 3, r_light, sizeof(r_light));
        sdxl_expand_wildcards("1girl", 12345ull, 0, 1u << 4, r_outf, sizeof(r_outf));
        sdxl_expand_wildcards("1girl", 12345ull, 0, 1u << 5, r_det, sizeof(r_det));
        sdxl_expand_wildcards("1girl", 12345ull, 0, 1u << 6, r_cam, sizeof(r_cam));
        sdxl_expand_wildcards("1girl", 12345ull, 1, (1u << 6) | (1u << 7), r_combo, sizeof(r_combo));
        const char* outs[7] = {r_pose, r_expr, r_scene, r_light, r_outf, r_det, r_cam};
        for (int i = 0; i < 7; i++) {
            if (strlen(outs[i]) <= 6) {
                fprintf(stderr, "FAIL: randomizer flag %d produced empty expansion: %s\n", i, outs[i]);
                return 1;
            }
            for (int j = i + 1; j < 7; j++) {
                if (strcmp(outs[i], outs[j]) == 0) {
                    fprintf(stderr, "FAIL: randomizer flags %d and %d produced identical output: %s\n", i, j, outs[i]);
                    return 1;
                }
            }
        }
        // Verify RAND_ECCHI no longer masks out RAND_CAMERA when both are checked!
        const char* cam_part = r_cam + 7; // skip "1girl, "
        if (!strstr(r_combo, cam_part)) {
            fprintf(stderr, "FAIL: RAND_ECCHI masked out RAND_CAMERA (%s not in %s)\n", cam_part, r_combo);
            return 1;
        }
        printf("Per-category independent Batch Word Randomizer PASS (pose='%s' | scene='%s')!\n", r_pose, r_scene);
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
