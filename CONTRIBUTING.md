# Contributing to GarlicDiffusion

Thank you for your interest in contributing to **GarlicDiffusion**! Whether you are submitting hardware benchmarks from a Base PS4 / Slim, optimizing GCN OpenCL compute kernels, adding presets, or converting new models, your help is welcome.

---

## Ways to Contribute

### 1. Submit Console Benchmarks (Especially Base PS4 & PS4 Slim!)
We actively collect verified real-world timings across PS4 models and firmwares.
- Open an issue using the **[Hardware Benchmark Report](.github/ISSUE_TEMPLATE/benchmark_report.md)** template.
- Attach the `[gdiff]` timing lines from `/data/garlic_diffusion_log.txt` (via FTP or Web UI) — please never estimate or fabricate numbers.

### 2. Code Contributions (C Engine, GCN Kernels, Web UI, Python Converters)
1. **Fork** `DJTOMATO/GarlicDiffusion` and create a feature branch (`git checkout -b feat/my-improvement`).
2. **Prerequisites** (Windows/Linux build environment):
   - Python 3.10+ with `numpy` and `Pillow`
   - [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain)
   - [Zig](https://ziglang.org) (used for both x86-64 PS4 C compilation and AMDGPU GCN kernel compilation)
   - .NET runtime (for OpenOrbis `PkgTool.Core`)
   - Node.js (for Web UI syntax verification)
   - Set `GARLIC_TOOLCHAIN` to your toolchain folder (or place it in `./toolchain/`).
3. **Run Verification Before Opening a PR:**
   ```bash
   # 1. Host CPU unit tests (CLIP tokenizer, A1111 PNG metadata, Gallery/Presets/Styles/Characters, PSD1 validator)
   zig cc -O2 -Wall -Idiffusion/src -DSDXL_CPU \
     tests/sdxl_test.c src/clip_tok.c src/gallery.c src/model_dl.c src/sdxl.c \
     -o build/sdxl_test.exe -lm
   ./build/sdxl_test.exe

   # 2. Validate Web UI JavaScript syntax
   node --check <extracted-script>

   # 3. Build PS4 PKG cleanly with zero errors
   python build_diffusion.py
   ```
4. Keep C code clean C99 with zero new `-Wall` warnings.
5. Respect thermal safety: never disable CU power scaling or make `Turbo` the default mode.

### 3. Model Conversions & Presets
- Use the converters in `pc/` (`convert_sd15.py`, `convert_sdxl_q8.py`, `upload_to_hf.py`) to package checkpoints into `.psd1` format.
- **Never commit model weights (`.psd1`, `.safetensors`, `.ckpt`) to Git.** Host models on Hugging Face and share your `owner/repo` manifest.
- **Strict SFW Repository Policy:** All sample images, default presets, character presets, and documentation assets submitted to this repository or website MUST be **100% SFW**.

---

## Licensing

GarlicDiffusion is licensed under the **[GNU General Public License v3.0 (GPLv3)](LICENSE)**. By submitting a pull request, you agree that your contributions will be licensed under the GPLv3 as well.
