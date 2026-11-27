# Changelog

All notable changes to **GarlicDiffusion** (Native SD 1.5 & SDXL on PS4 / PS4 Pro) are documented here.

---

## [v1.17] — 2026-11-27

### Smart NSFW / SFW Prompt Conflict Sanitizer (Backend + Web UI)
- **Automatic Negative Prompt NSFW Stripping:** When **NSFW Mode** is enabled, core NSFW/adult/ecchi blocking keywords (including weighted `((nsfw:1.4))` and `rating: explicit` / `rating: questionable` / `rating: sensitive` forms) are automatically removed from the negative prompt across Web UI, PS4 TV Studio, Batch Randomizer, and A1111 API — while preserving non-conflicting negatives like `censored`, `mosaic censoring`, and `bad hands`.
- **Positive-vs-Negative Conflict Resolution:** If an NSFW or ecchi keyword (or part of a tag, e.g. `black lace lingerie`, `bikini`, `cleavage`) appears in the positive prompt, any matching tag is automatically deleted from the negative prompt.
- **Automatic SFW Tag Stripping:** When NSFW/ecchi tags are present in the positive prompt while NSFW Mode is ON, conflicting SFW rating tags (`sfw`, `rating: safe`, `rating: general`, `rating: sfw`, `safe for work`) are stripped from the positive prompt without touching unrelated phrases (e.g. `safety pin`).
- **Live Web UI Cleanup:** Toggling NSFW Mode ON, clicking Clean Dupes, applying presets, or loading SFW gallery images to Studio immediately cleans stale SFW guard tags from the prompt fields.

---

## [v1.16] — 2026-11-26

### Reworked LoRA Manager, Download Queue & Compatibility Guard
- **PS4 TV LoRA Manager Rework:** Split into clean **Installed** and **Download Catalog** sub-views (`L2/R2`) with **pagination** (`Left/Right`, 8 two-line cards per page), clear compatibility badges (`COMPATIBLE`, `ACTIVE IN STUDIO`, `INCOMPATIBLE`), and **`Square`** to **Download All Compatible** LoRAs in one press.
- **Web UI LoRA Manager Rework:** Added `Compatible Only` / `Show All` filter pills, pagination (`Prev / Next`), **`⬇ Download All Compatible`** & **`⬇ Download All`** buttons, **`+ Queue Download`** buttons while downloading, and live `+N queued` progress telemetry.
- **Sequential LoRA Download Queue:** Clicking additional LoRAs while one is downloading now queues them seamlessly instead of failing; `Cancel` stops the active transfer and clears the queue.
- **Expanded Compatible LoRA Catalog (22 LoRAs):** Synced all 18 built-in LoRAs + added 4 verified SDXL LoRAs (`Anime Detailer XL`, `Studio Ghibli Style Slider`, `Watercolor Anime Slider`, `Cinematic Lighting Slider`) into both `catalog/loras.json` and the built-in catalog (**19 SDXL/Illustrious + 3 SD 1.5**).
- **Strict Cross-Architecture LoRA Protection:** Added safetensors header architecture inspection in `src/sdxl.c` before layer fusion so SD 1.5 LoRAs can never partially fuse into SDXL CLIP-L/ResNet weights (and vice versa), plus UI/HTTP blocking and auto-clearing on model switch.
- **Unicode JSON Unescaping:** Fixed `\u00b7` (`·`) and `\uXXXX` escapes in the C manifest parser so model/LoRA descriptions render cleanly on TV.

### Model Quality, Presets & Face Fidelity
- **Per-Model Tuned DMD2 Presets & Tag Ordering:**
  - `Animagine_XL_4_dmd2`: Tuned to **`5 steps · CFG 1.1 · LCM`** (Speed Mode: `4 steps · CFG 1.0`) with **Animagine-native tag order** (character & framing first, quality tags appended at the end) to eliminate DMD2 overshoot and blown highlights.
  - `NoobAI_XL_1.1_dmd2`: Tuned to **`6 steps · CFG 1.2 · LCM`** with NoobAI-native `newest` quality/negative tags.
  - Illustrious DMD2 checkpoints (`WAI v17`, `WAI v14`, `Nova v18`, `Hassaku v3.1`, `Pencil v3.2`, `Plant Milk`): Standardized at **`6 steps · CFG 1.4 · LCM`** across `sdxl.c`, `preset.cfg`, `catalog/models.json`, `models/manifest.json`, `model_dl.c`, and `import_civitai.py`.
- **LCM Late-Step Eye/Detail Stabilization:** Enabled variance-preserving late-step `gamma` tapering starting at **`5+ steps`** (previously `>= 7`) so 5-step and 6-step DMD2 presets benefit from cleaner pupils and facial details.


### Gallery Recreation, Fix Face & Full Settings Restore
- **"Load to Studio" & "Re-Render Variation (New Seed)" (TV + Web UI):** Now automatically switch to the original image's checkpoint (when installed) and restore **every** stored setting (`model`, `prompt`, `<lora:...>` tags, `negative`, native base `width × height` even for upscaled images, `steps`, `cfg`, `sampler`, `upscale`, `eco`, and `seed`) without being overwritten by model switch defaults.
- **Fix Face (TV + Web UI):** Added **`Fix Face (AI Face & Eye Detailer)`** to the PS4 TV Gallery Artwork Actions modal, fixed the error-masking bug (`Gallery operation failed`), added automatic download & Recommended Add-on entries for universal `vae_enc_sdxl.psd1` / `vae_enc_sd15.psd1`, plus built-in **Real-ESRGAN Anime v3 Face-Crop Restoration** fallback when offline.
- **Active Hardware Thermal Guard (Turbo Safety):** Added real-time `CPU`/`SoC` temperature monitoring inside `block_breather()` and the diffusion loop (`src/sdxl.c`). Automatically steps down CU masks at `>=80 °C` / `>=84 °C` and pauses to actively cool down at `>=86–88 °C` (`<=81–82 °C`) so even Turbo mode can never trigger a PS4 overheating shutdown.

### Live Progressive Previews & Discord / A1111 API Upgrades
- **Live Per-Step & Per-Stage Previews (PS4 TV + Web UI):** Renders an instant CPU latent-to-RGB preview after every UNet step (`< 0.5 ms`, zero GPU overhead), followed by crisp intermediate previews after VAE decode and Real-ESRGAN Upscale while FaceFix runs (`/preview.png`).
- **Silent API / Bot Power Mode:** Added a persisted `API / Bot Power` setting (defaults to **Silent** for whisper-quiet unattended Discord/A1111 calls) configurable in both the Web UI and PS4 TV Power Mode menu, without overwriting Studio power mode.
- **Exact-Request API Cache:** Added a 64-entry LRU gallery-backed cache for `/sdapi/v1/txt2img` and `/v1/images/generations` that returns identical requests immediately with zero GPU work or extra PNG RAM (`POST /cache/clear` or `"no_cache": true` to bypass).
- **Optional SDXL / Illustrious API FaceFix:** Auto FaceFix now defaults to **OFF** on SDXL/Illustrious API calls unless explicitly requested (`ADetailer` / `auto_face_fix: true`) or enabled via the new **🤖 API FaceFix** toggle in the Web UI.
- **Red-DiscordBot (`bz-cogs/aimage`) Compatibility:** Added `/sdapi/v1/sd-modules` (and `hypernetworks`, `face-restorers`, `realesrgan-models`) returning `200 OK` (`[]`) to eliminate autocomplete cache 404 warnings.

---

## [v1.15] — Previous Release
- 60 FPS non-blocking PS4 TV UI with Easy Mode & Advanced Studio 2-choice welcome screen.
- GitHub OTA updater (`/data/pkg/Garlic-Diffusion.pkg`) with one-click download and GoldHEN install prep.
- Gallery Folders (`Create / Move / Delete / Download .zip`), Starred favorites, and >128 newest-first Gallery sorting with fast thumbnails.
- 500-word SFW + 500-word NSFW-gated batch randomizer, inline `{a|b|c}` wildcards and prompt dupe cleaner.

## [v1.10] — Initial Public Release
- Native PS4 / PS4 Pro GCN compute inference for SD 1.5 & SDXL / Illustrious checkpoints, 0-VRAM Kohya LoRA fusion, and Real-ESRGAN Anime v3 1080p upscaling.
