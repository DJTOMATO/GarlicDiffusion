# Credits, Lineage & Licenses

**GarlicDiffusion**'s source code is free and open-source software licensed under the **[GNU General Public License v3.0 (GPLv3)](LICENSE)**.

This project stands directly on the shoulders of the PlayStation 4 homebrew community, open-source AI researchers, and especially **Garlic LLM**. Every upstream project and third-party model retains its own license and copyright.

---

## Upstream Foundation & Inspiration

| Project | Author | License | Relationship |
|---|---|---|---|
| **[Garlic LLM](https://github.com/leonardo-alegria/garlic-llm)** ([garlicllm.dev](https://garlicllm.dev)) | **[Leonardo Alegría](https://www.linkedin.com/in/leonardoalegria/)** | **GPLv3** | Original inspiration and foundational PS4 bare-metal GCN compute architecture ("Garlic" GPU direct-memory pipeline, PM4 command buffer dispatch, OpenOrbis + Zig AMDGPU workflow). GarlicDiffusion adapts this compute foundation to run Stable Diffusion 1.5, SDXL, and Real-ESRGAN natively on PS4/PS4 Pro. |

> If you appreciate bare-metal AI on PS4, please consider supporting Leonardo Alegría's pioneering work on **[Garlic LLM](https://garlicllm.dev)** and his technical book [*Nine Billion Parameters on a PS4*](https://ko-fi.com/s/5d8050b149).

---

## AI Models & Upscalers

Model weights are **not** embedded in this Git repository or the 6.6 MiB PS4 PKG. Users download converted `.psd1` packs directly inside the app (or convert their own checkpoints with the scripts in `pc/`). Each model remains governed by its original creator's license:

| Component / Model | Author / Organization | License | Role in GarlicDiffusion |
|---|---|---|---|
| **[Stable Diffusion XL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)** | Stability AI | CreativeML Open RAIL++-M | Base SDXL architecture, VAE & CLIP-L / OpenCLIP-bigG encoders |
| **[Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)** | RunwayML / Stability AI | CreativeML Open RAIL-M | Base SD 1.5 architecture (fast & low-VRAM mode for Base PS4 / Slim) |
| **[WAI-Illustrious-SDXL (v1.7 / v1.4)](https://civitai.com/models/827184)** | WAI / OnomaAI (Illustrious-XL) | Fair AI Public License 1.0-SD | Primary curated SDXL anime illustration checkpoints |
| **[DMD2 (Distribution Matching Distillation)](https://github.com/tianweiy/DMD2)** | Tianwei Yin et al. (MIT / Adobe) | MIT / Apache-2.0 | 4–6 step CFG 1.0 distillation fused into SDXL checkpoints |
| **[Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD)** | ByteDance | CreativeML Open RAIL++-M | Fast 8-step distillation fused into SD 1.5 checkpoints |
| **[MeinaMix V11](https://civitai.com/models/7240/meinamix)** | Meina | CreativeML Open RAIL-M | Fast glossy anime SD 1.5 checkpoint |
| **[DreamShaper 8](https://civitai.com/models/4384/dreamshaper)** | Lykon | CreativeML Open RAIL-M | Painterly fantasy / 2.5D & stylized 2D SD 1.5 checkpoint |
| **[Anything V5](https://huggingface.co/stablediffusionapi/anything-v5)** | Community / Linaqruf | CreativeML Open RAIL-M | Classic cel-shaded anime SD 1.5 checkpoint |
| **[Real-ESRGAN AnimeVideo v3](https://github.com/xinntao/Real-ESRGAN)** | Xintao Wang et al. | BSD-3-Clause | Bundled 2.4 MiB `SRVGGNetCompact` 4× GPU AI upscaler (`upscaler.psd1`) |

---

## Toolchain, Ecosystem & Standards

| Project | Author | License | Use in GarlicDiffusion |
|---|---|---|---|
| **[OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain)** | OpenOrbis Team | GPLv3 | PS4 C runtime, headers, stubs, and PKG packaging |
| **[Zig](https://ziglang.org) & LLVM AMDGPU** | Zig Software Foundation / LLVM | MIT / Apache-2.0 w/ LLVM Exception | Compiles both x86-64 PS4 C code and bare-metal GCN OpenCL kernels |
| **[GoldHEN](https://github.com/GoldHEN/GoldHEN)** | SiSTRo & Contributors | Third-party | Homebrew execution environment on jailbroken PS4 consoles |
| **[Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)** & **`bz-cogs/aimage`** | AUTOMATIC1111 / Community | AGPL-3.0 / MIT | API route compatibility (`/sdapi/v1/txt2img`) & PNG `tEXt` metadata convention |

---

## Trademarks & Showcase Art Notice

- **Not affiliated with or endorsed by Sony Interactive Entertainment.** "PlayStation", "PS4", "PS4 Pro", and "DualShock" are registered trademarks of Sony Interactive Entertainment Inc.
- **Hatsune Miku** is a trademark and character of Crypton Future Media, Inc. Showcase images featuring Hatsune Miku are non-commercial technical benchmarks demonstrating local PS4 GPU inference fidelity.
- All third-party trademarks and character names mentioned in presets belong to their respective owners.
