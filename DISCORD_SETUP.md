# GarlicDiffusion — Official Discord Server & PS4 Bot Setup Guide

**Official Support Discord Invite:** [https://discord.gg/Gbyq3kpGjc](https://discord.gg/Gbyq3kpGjc)

This guide contains everything you need to structure the **GarlicDiffusion Support Discord** and connect your PS4 (`http://<PS4-IP>:8081`) to a Discord bot (`bz-cogs/aimage` or any Automatic1111 / OpenAI-compatible bot).

---

## 1. Recommended Discord Channel Layout

Create these categories and channels so users can easily find downloads, request new models, share PS4 renders, and use your PS4 bot:

### 📢 `INFO & RELEASES` (Read-Only for Members)
- `#welcome-and-rules` — Server overview, links to GitHub/HuggingFace/Ko-fi, and thermal safety disclaimer.
- `#announcements` — New `.pkg` releases, newly uploaded Hugging Face models, and benchmark updates.
- `#download-and-install` — Quick 3-step PS4 PKG install guide + Hugging Face repo ID (`DJTomato/ps4-diffusion-models`).

### 💬 `COMMUNITY & MODELS`
- `#model-requests` — **(Forum or Text Channel)** Where users suggest and vote on new SD 1.5 / SDXL checkpoints to convert to `PSD1` for the Hugging Face catalog.
- `#ps4-showcase` — Share images generated on Base PS4, PS4 Slim, and PS4 Pro (with seed/model info!).
- `#prompts-and-presets` — Share exported `.json` presets, art styles, and character tags.
- `#general-chat` — General PS4 homebrew & local AI discussion.

### 🤖 `PS4 BOT & SUPPORT`
- `#ps4-bot-generate` — Dedicated channel where members can run `/generate` on your live PS4 bot!
- `#support-and-help` — Troubleshooting PS4 installation, Web UI, network/HTTPS downloads, and bot setup.

---

## 2. Ready-to-Paste Pinned Messages

### 📌 Pin in `#welcome-and-rules`
```markdown
# 🧄 Welcome to the Official GarlicDiffusion Discord!

**GarlicDiffusion** runs native **Stable Diffusion 1.5**, **SDXL**, and **Real-ESRGAN 1080p/4K AI Upscaling** 100% offline on a **PlayStation 4 / PS4 Slim / PS4 Pro** GPU — no PC or cloud required!

### 🔗 Official Links
• **GitHub Repo & PS4 PKG:** https://github.com/DJTOMATO/GarlicDiffusion
• **Website & Benchmarks:** https://djtomato.github.io/GarlicDiffusion/
• **PS4 Model Catalog (Hugging Face):** https://huggingface.co/DJTomato/ps4-diffusion-models
• **Support Development on Ko-fi:** https://ko-fi.com/djtomato
• **Inspired by Garlic LLM:** https://garlicllm.dev (by Leonardo Alegría)

### 📜 Server Rules & Safety
1. **Keep `#ps4-showcase` and `#ps4-bot-generate` SFW** so everyone can browse safely.
2. **Have a checkpoint idea?** Post it in `#model-requests` with a link to the Hugging Face / Civitai model and tell us if it's SD 1.5 or SDXL!
3. **Thermal Safety Reminder:** Use **Quiet (20 CU)** or **Silent (14 CU)** mode on your PS4 and keep your console's vents clear.
```

### 📌 Pin in `#model-requests`
```markdown
# 📦 Request New PS4 Models (`DJTomato/ps4-diffusion-models`)

Want us to convert and upload a new checkpoint to the in-app Hugging Face catalog? Reply with:
1. **Model Name & Link** (Civitai or Hugging Face URL)
2. **Architecture:** `SD 1.5` (1.2 GiB · best for Base PS4 & fast 25s renders) or `SDXL` (3.7 GiB · we can fuse the 6-step DMD2 or 8-step Hyper-SD LoRA so it renders in ~50s on PS4!)
3. **Style / Specialty:** (e.g., 2D Anime, Pixel Art, Mecha, Dark Fantasy, Realistic, Scenic)
```

---

## 3. Connecting Your Discord Bot to Your PS4 (`http://<PS4-IP>:8081`)

GarlicDiffusion's built-in HTTP server on port `8081` implements the **Automatic1111 WebUI REST API** out of the box:

| Endpoint | Method | Purpose |
| :--- | :---: | :--- |
| `/sdapi/v1/txt2img` | `POST` | Generates an image and returns base64 PNG + A1111 `parameters` & `info` JSON |
| `/sdapi/v1/progress` | `GET` | Live generation progress (`0.0–1.0`), ETA, and current step |
| `/sdapi/v1/sd-models` | `GET` | Lists all installed models on the PS4 |
| `/sdapi/v1/samplers` | `GET` | Lists supported samplers (`Euler a`, `Euler`, `LCM`) |
| `/sdapi/v1/upscalers` | `GET` | Lists `None`, `RealESRGAN_AnimeVideo-v3` (1080p / 2x / 4K) |
| `/sdapi/v1/options` | `GET` / `POST` | Reads or switches the active PS4 checkpoint (`sd_model_checkpoint`) |
| `/sdapi/v1/interrupt` | `POST` | Cancels the active generation |
| `/v1/images/generations` | `POST` | OpenAI DALL·E compatible image generation endpoint |

### Option A: Using `bz-cogs/aimage` (Red-DiscordBot Cog)
1. Install Red-DiscordBot on any PC, Raspberry Pi, or home server on the same LAN as your PS4:
   ```bash
   pip install Red-DiscordBot
   redbot-setup
   redbot <your-bot-instance>
   ```
2. In Discord, load the `downloader` cog and install `bz-cogs`:
   ```text
   [p]load downloader
   [p]repo add bz-cogs https://github.com/zhaobenny/bz-cogs
   [p]cog install bz-cogs aimage
   [p]load aimage
   ```
3. Point `aimage` to your PS4's LAN IP address and port `8081`:
   ```text
   [p]anelset endpoint http://192.168.1.29:8081
   ```
   *(or `/aimage_config endpoint http://192.168.1.29:8081` depending on cog version)*
4. Test generating directly from Discord onto your PS4 GPU:
   ```text
   /generate prompt: 1girl, hatsune miku, turquoise twintails, cherry blossoms, masterpiece
   ```
   > **Tip:** Enable **Hires. fix** in `aimage` (or set `enable_hr: true` in `/sdapi/v1/txt2img`) and GarlicDiffusion will automatically run the on-console **Real-ESRGAN AnimeVideo v3 1080p HD upscaler** before returning the image to Discord!

### Option B: Standalone 30-Line Python Discord Bot (`discord.py`)
If you want a lightweight single-file bot that talks directly to your PS4 (`http://192.168.1.29:8081/sdapi/v1/txt2img`) without Red-DiscordBot, run:

```python
# ps4_discord_bot.py — Lightweight Discord Bot for GarlicDiffusion PS4
# pip install discord.py aiohttp
import base64, io, os, aiohttp, discord
from discord import app_commands

PS4_URL = os.environ.get("PS4_URL", "http://192.168.1.29:8081")
TOKEN   = os.environ["DISCORD_BOT_TOKEN"]

class GarlicBot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)
    async def setup_hook(self):
        await self.tree.sync()

client = GarlicBot()

@client.tree.command(name="ps4gen", description="Generate an image on the PS4 GPU with GarlicDiffusion!")
@app_commands.describe(prompt="Positive prompt", steps="Steps (6 for SDXL DMD2, 8 for MeinaMix 8-step)", hd1080="Upscale to 1080p HD with Real-ESRGAN")
async def ps4gen(interaction: discord.Interaction, prompt: str, steps: int = 6, hd1080: bool = True):
    await interaction.response.defer(thinking=True)
    payload = {
        "prompt": prompt,
        "negative_prompt": "lowres, bad anatomy, bad hands, extra fingers, blurry, worst quality",
        "width": 768,
        "height": 1024,
        "steps": steps,
        "cfg_scale": 1.0 if steps <= 6 else 3.5,
        "sampler_name": "LCM" if steps <= 6 else "Euler",
        "enable_hr": hd1080,
    }
    async with aiohttp.ClientSession() as sess:
        async with sess.post(f"{PS4_URL}/sdapi/v1/txt2img", json=payload, timeout=300) as resp:
            data = await resp.json()
            img_bytes = base64.b64decode(data["images"][0])
            info = data.get("info", "")
            await interaction.followup.send(
                content=f"🧄 **Rendered on PlayStation 4 GPU** (`{prompt[:120]}`)",
                file=discord.File(io.BytesIO(img_bytes), filename="ps4_garlic_diffusion.png"),
            )

client.run(TOKEN)
```
