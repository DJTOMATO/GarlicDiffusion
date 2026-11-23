<p align="center"><img src="docs/logo.svg" width="108" alt="GarlicDiffusion logo"></p>

# GarlicDiffusion

**Native Stable Diffusion 1.5 & SDXL image generation running directly on PlayStation 4 & PS4 Pro GCN GPU. 100% local, no PC required.**

[Website](https://djtomato.github.io/GarlicDiffusion/) · [Download PKG](https://github.com/DJTOMATO/GarlicDiffusion/releases/latest) · [Pre-Converted Models (Hugging Face)](https://huggingface.co/DJTomato/ps4-diffusion-models) · [API & Discord Bot](#api--discord-integration) · [Support on Ko-fi](https://ko-fi.com/djtomato)

---

## Real PS4 Pro Outputs

Generated natively on a PS4 Pro (`1920×1080` TV / LAN Web UI) in **Quiet Mode (20 CU)** with zero PC offloading:

| WAI-Illustrious v1.4 DMD2 · `768×1024` · 6 steps · **51.5s** | SD 1.5 + RealESRGAN 1080p · `[Placeholder]` | WAI-Illustrious v17 DMD2 · `[Placeholder]` |
| :---: | :---: | :---: |
| ![PS4 Pro SDXL output](docs/samples/sample_wai_v14_dmd2.jpg) | *Capture SFW 1080p showcase image* | *Capture SFW WAI v17 showcase image* |

---

## Features

| Feature | Details |
|---|---|
| **Dual Architecture Engine** | Runs both **Stable Diffusion 1.5** (`~1.5 GiB` VRAM) and **SDXL** (`~4.0 GiB` VRAM) with custom zero-scratch GCN (`gfx701`) OpenCL compute kernels and hand-written PM4 command buffers. |
| **PS4 Base, Slim & PS4 Pro Neo** | Full **PS4 Pro Neo** (36 CU, `4.67 GiB` GPU pool, up to `1024×1024`) and **Base/Slim PS4** (18 CU, compact `325 MiB` scratch so both SD 1.5 and SDXL fit!). |
| **Hardware CU Thermal Scaling** | 4 live power profiles (`Silent`, `Quiet`, `Balanced`, `Turbo`) using symmetric checkerboard `SH_STATIC_THREAD_MGMT` compute-unit masks + inter-block breathers to keep PS4 temps & fans calm. |
| **Built-in 1080p AI Upscaler** | Bundled **Real-ESRGAN Anime v3** (`SRVGGNetCompact`, 2.4 MiB) running natively on PS4 GPU — upscales `512×768 → 1080×1620` in **8.2s**! |
| **Fast 6–8 Step Samplers** | `Euler a`, `Euler`, and `LCM` (DMD2 6-step `CFG 1.0` single-pass generation in **~50s** at `768×1024` in Quiet mode). |
| **Full LAN Web Studio** | Dark-mode Studio on `http://<PS4_IP>:8081` with **Live Preview**, **Save Prompt**, **Reusable Art Styles**, **Persistent Gallery + Lightbox** (1:1 zoom, copy prompt, recreate, upscale, resize, delete), **Preset Manager** (export/import JSON), and **Model Manager**. |
| **In-App Hugging Face Downloader** | Browse and 1-click download pre-converted `PSD1` models straight onto PS4 over HTTPS (`libSceHttp` + `libSceSsl`) from Web UI → **Models** or PS4 TV (`[Up]`). |
| **Standard PNG Info Metadata** | Every generated & upscaled PNG embeds a standard Automatic1111/Civitai `tEXt` `"parameters"` chunk (prompt, negative, steps, sampler, CFG, seed, size, model). |
| **A1111, Discord & OpenAI API** | Drop-in `/sdapi/v1/txt2img` (compatible with **Automatic1111** clients and **`zhaobenny/bz-cogs/aimage`** Discord bot) + OpenAI `/v1/images/generations`. |

---

## Benchmarks (Measured on Real Hardware)

### PS4 Pro (Neo Mode · 36 CUs · `2.76 TFLOPS` peak Q8_0 GEMM)

| Model | Arch | Resolution | Steps / CFG / Sampler | Power Mode | UNet | VAE | Upscale (1080p) | Total Time | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---|
| **WAI-Illustrious v14 DMD2** | SDXL | `768×1024` | 6 · CFG 1.0 · `LCM`/`Euler` | Quiet (20 CU) | 31.3s | 19.2s | — | **50.8s** | Razor-sharp anime |
| **WAI-Illustrious v17 DMD2** | SDXL | `768×1024` | 6 · CFG 1.0 · `LCM` | Quiet (20 CU) | *TBD* | *TBD* | — | *~50s (TBD)* | Latest WAI v17 |
| **WAI-Illustrious v14 DMD2** | SDXL | `768×1024` | 6 · CFG 1.0 · `LCM` | Silent (14 CU) | *TBD* | *TBD* | — | *TBD* | Ultra-quiet profile |
| **WAI-Illustrious v14 8-Step** | SDXL | `768×1024` | 8 · CFG 3.5 · `Euler` | Quiet (20 CU) | 79.0s | 18.5s | — | **98.0s** | Negative prompt active |
| **Anything V5** | SD 1.5 | `512×768 → 1080×1620` | 20 · CFG 7.0 · `Euler a` | Quiet (20 CU) | 68.2s | 11.0s | **8.2s** | **87.5s** | `3.4s/step` (`attn_d40/80/160`) |
| **MeinaMix V11 8-Step** | SD 1.5 | `512×768 → 1080×1620` | 8 · CFG 3.5 · `Euler` | Quiet (20 CU) | *~27s* | *~11s* | **8.2s** | *~46s (TBD)* | Fastest SD 1.5 + 1080p |
| **Base PS4 / Slim (18 CU)** | SD 1.5 / SDXL | `512×768` / `768×1024` | — | Balanced (14 CU) | *TBD* | *TBD* | *TBD* | *TBD* | Community tests welcome! |

**Boot Hardware CU Power Micro-Benchmark (PS4 Pro Neo, `gemm_q8 2304×1280×1280`):**
- `Turbo (36 CU)`: **2.76 TFLOPS**
- `Balanced (25 CU)`: **1.92 TFLOPS**
- `Quiet (20 CU, default)`: **1.61 TFLOPS**
- `Silent (14 CU)`: **0.99 TFLOPS**

---

## Requirements

- A **PS4, PS4 Slim, or PS4 Pro** running [GoldHEN](https://github.com/GoldHEN/GoldHEN) (tested on FW 11.00; compatible with 9.00–12.50+ homebrew setups).
- **2–4 GB free HDD space** per installed checkpoint (SD 1.5: ~1.2 GiB · SDXL: ~3.7 GiB).
- Optional: any phone, tablet, or PC on the same LAN to open the full Web Studio at `http://<PS4_IP>:8081`.

---

## Install

1. Download **`Garlic-Diffusion.pkg`** (~6.6 MiB — includes the GCN kernels, CLIP tokenizer, Web UI, and Real-ESRGAN Anime v3 upscaler) from [**GitHub Releases**](https://github.com/DJTOMATO/GarlicDiffusion/releases/latest).
2. Install the PKG via *GoldHEN → Package Installer*.
3. Launch **Garlic Diffusion** from your PS4 home screen.
4. **Download a model right inside the app** (no FTP needed!):
   - Open `http://<PS4_IP>:8081` in any browser → click the **Models** tab → click **Download to PS4** on any model from [`DJTomato/ps4-diffusion-models`](https://huggingface.co/DJTomato/ps4-diffusion-models) (or press **`[Up]`** on the PS4 controller to open the native TV Model Manager!).
   - *Manual / Offline FTP option:* Copy any converted model folder (`unet.psd1`, `clip_l.psd1`, `clip_g.psd1`, `vae.psd1`, `preset.cfg`) to `/data/garlic-diffusion/models/<ModelName>/`.

---

## PS4 Controller Controls

### Studio Mode
| Button | Action |
|---|---|
| **✕** | Edit prompt (PS4 On-Screen Keyboard or USB Keyboard) |
| **△** | Generate image |
| **○** | Cancel active generation |
| **□** | Open **TV Image Gallery** |
| **↑** | Open **Model Manager & Downloader** |
| **← / →** | Cycle saved prompt presets |
| **L1 / R1** | Change resolution (`512×512` to `1024×1024`) |
| **L2 / R2** | Adjust steps (`4..30`) |
| **L3 / R3** | Cycle AI Upscaler (`Off / 1080p / 2x`) / GPU Power Mode |
| **OPTIONS** | Switch checkpoint model |

### TV Gallery (`□`) & Model Manager (`↑`)
- **Gallery:** `← / →` browse · `R1` 100% edge-to-edge full-screen · `✕` load prompt to Studio · `△` recreate (new seed) · `L3` AI upscale to 1080p · `OPTIONS ×2` delete · `○` back.
- **Model Manager:** `↑ / ↓` select · `✕` load installed model or download from Hugging Face · `△` cancel download · `OPTIONS ×2` delete · `○` back.

---

## API & Discord Integration

Garlic Diffusion runs a multi-threaded HTTP server on port **`8081`** with CORS enabled.

### Automatic1111 & Discord Bot (`zhaobenny/bz-cogs/aimage`)
Point any Automatic1111 client or Red-DiscordBot [`bz-cogs/aimage`](https://github.com/zhaobenny/bz-cogs/tree/main/aimage) to:
```text
http://<PS4_IP>:8081
```
Supported endpoints:
- `POST /sdapi/v1/txt2img` (returns base64 PNG + full A1111 `info` & `infotexts`, supports `override_settings.sd_model_checkpoint` & `enable_hr: true`)
- `GET /sdapi/v1/sd-models`, `GET /sdapi/v1/samplers`, `GET /sdapi/v1/upscalers`, `GET /sdapi/v1/prompt-styles`, `GET /sdapi/v1/scripts`, `GET /sdapi/v1/progress`, `GET/POST /sdapi/v1/options`, `POST /sdapi/v1/interrupt`

### OpenAI-Compatible Image API
```bash
curl http://<PS4_IP>:8081/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"masterpiece, best quality, 1girl, silver hair, neon night","size":"768x1024"}'
```

---

## Convert Your Own Models (PC)

Convert any Hugging Face Diffusers **SDXL** or **SD 1.5** checkpoint (with optional Kohya LoRA fusion like DMD2 or Hyper-SD) into Q8_0 `PSD1`:

```bash
# Fast 6-step SDXL with DMD2 LoRA:
python pc/convert_sdxl.py frankjoshua/waiIllustriousSDXL_v170 WAI_Illustrious_v17_dmd2 \
  --arch sdxl --lora tianweiy/DMD2/dmd2_sdxl_4step_lora_fp16.safetensors --lora-scale 0.85

# Fast 8-step SD 1.5 with Hyper-SD LoRA:
python pc/convert_sdxl.py Meina/MeinaMix_V11 MeinaMix_V11_8step \
  --arch sd15 --lora ByteDance/Hyper-SD/Hyper-SD15-8steps-CFG-lora.safetensors --lora-scale 0.8

# Publish your own catalog + manifest.json to Hugging Face:
python pc/upload_to_hf.py <your-hf-username>/ps4-diffusion-models
```

---

## Build from Source

Requires Python 3 (`numpy`, `Pillow`), [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain), [Zig](https://ziglang.org) (for `zig clang` GCN OpenCL & x86_64 FreeBSD compilation), and .NET runtime for `PkgTool.Core`:

```bash
python build_diffusion.py           # builds PKG
python build_diffusion.py --upload  # builds & uploads to PS4_IP via GoldHEN FTP
```

---

## Support

GarlicDiffusion is free and open source (GPLv3). If it brought your PS4 back to life and you'd like to support further GCN optimizations, new model architectures, and Base/Slim/Pro testing:

- **Ko-fi:** https://ko-fi.com/djtomato

Starring the repo, sharing PS4-generated artwork, and posting Base PS4 / Slim benchmark reports in [Issues](https://github.com/DJTOMATO/GarlicDiffusion/issues) helps just as much!

---

## Credits & License

Licensed under **GNU GPLv3** (see [`LICENSE`](LICENSE) and [`CREDITS.md`](CREDITS.md)). Inspired by [Garlic LLM](https://github.com/leonardo-alegria/garlic-llm)'s bare-metal PS4 GCN PM4 compute architecture.

*Not affiliated with or endorsed by Sony Interactive Entertainment. PlayStation and PS4 are trademarks of their respective owners.*
