# Changelog

All notable changes to **GarlicDiffusion** (Native SD 1.5 & SDXL on PS4 / PS4 Pro) are documented here.

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
- **Honest Out-of-the-Box Benchmark (`pc/run_miku_benchmark.py`):** Removed hidden step/CFG overrides and benchmark-only LoRA stacks; now defaults to a fresh 2-seed run using each checkpoint's exact app defaults (`--resume` optional).

### Gallery Recreation & Full Settings Restore
- **"Load to Studio" & "Re-Render Variation (New Seed)" (TV + Web UI):** Now automatically switch to the original image's checkpoint (when installed) and restore **every** stored setting (`model`, `prompt`, `<lora:...>` tags, `negative`, native base `width × height` even for upscaled images, `steps`, `cfg`, `sampler`, `upscale`, `eco`, and `seed`) without being overwritten by model switch defaults.

---

## [v1.15] — Previous Release
- 60 FPS non-blocking PS4 TV UI with Easy Mode & Advanced Studio 2-choice welcome screen.
- GitHub OTA updater (`/data/pkg/Garlic-Diffusion.pkg`) with one-click download and GoldHEN install prep.
- Gallery Folders (`Create / Move / Delete / Download .zip`), Starred favorites, and >128 newest-first Gallery sorting with fast thumbnails.
- 500-word SFW + 500-word NSFW-gated batch randomizer, inline `{a|b|c}` wildcards, prompt dupe cleaner, and obfuscated seed headers.

## [v1.10] — Initial Public Release
- Native PS4 / PS4 Pro GCN compute inference for SD 1.5 & SDXL / Illustrious checkpoints, 0-VRAM Kohya LoRA fusion, and Real-ESRGAN Anime v3 1080p upscaling.
