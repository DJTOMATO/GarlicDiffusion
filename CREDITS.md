# Credits and Licences

This project's own source code and OpenCL GCN kernels are licensed under the **GNU General Public License v3.0** (`LICENSE`). It builds upon the open-source tools, research, and community models listed below. Each external project and model retains its own licence.

## PS4 Bare-Metal GPU & Toolchain Foundation
- **[Garlic LLM](https://github.com/leonardo-alegria/garlic-llm)** by **Leonardo Alegría** (GPLv3) — Pioneered the bare-metal PS4 GCN compute technique (compiling OpenCL C to `gfx701` via Zig/LLVM and dispatching hand-written PM4 command buffers in Garlic memory without Sony's proprietary SDK).
- **[OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain)** (GPLv3) — Open-source PS4 SDK headers, stub libraries, `create-fself`, `create-gp4`, and `PkgTool.Core`.
- **[Zig](https://ziglang.org)** (MIT) — Bundled `clang` and `lld` used to compile both the PS4 x86_64 ELF and the AMD GCN (`amdgcn-amd-amdhsa -mcpu=bonaire`) compute kernels.
- **[GoldHEN](https://github.com/GoldHEN/GoldHEN)** — Homebrew enabler and FTP/package installer environment on PS4.

## Upscaler & Distillation Research
- **[Real-ESRGAN (`realesr-animevideov3` / `SRVGGNetCompact`)](https://github.com/xinntao/Real-ESRGAN)** by Xintao Wang et al. (BSD-3-Clause) — Bundled 4x anime upscaler weights (`assets/upscaler.psd1`, 2.4 MiB) running natively on PS4 GPU.
- **[DMD2 (Improved Distribution Matching Distillation)](https://huggingface.co/tianweiy/DMD2)** by Tianwei Yin et al. (CC-BY-NC-4.0) — 4–6 step CFG-1.0 distillation LoRA for SDXL.
- **[Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD)** by ByteDance — 8-step CFG distillation LoRAs for SD 1.5 and SDXL.

## Community Diffusion Checkpoints (Hosted Separately on Hugging Face)
Model weights are NOT bundled inside the app PKG and remain the work of their respective authors:
- **WAI-Illustrious SDXL (v14 / v17)** — WAI / Illustrious-XL community checkpoint (Fair AI Public License 1.0-SD)
- **MeinaMix V11** — Meina (CreativeML Open RAIL-M)
- **DreamShaper 8** — Lykon (CreativeML Open RAIL-M)
- **Anything V5** — Community SD 1.5 anime checkpoint (CreativeML Open RAIL-M)
- **CLIP BPE Tokenizer** — OpenAI CLIP / OpenCLIP vocabulary (`assets/clip_tok.bin`)

---

Not affiliated with or endorsed by Sony Interactive Entertainment. PlayStation and PS4 are trademarks of their respective owners.
