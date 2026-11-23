"""Fast converter: downloads Hugging Face Diffusers SDXL safetensors via curl (30+ MB/s), converts to PSD1
binary files for Garlic Diffusion on PS4 / PS4 Pro, and removes each temporary safetensors file right away.

Output in diffusion/models/<out_name>/:
  vae.psd1      (~190 MiB: post_quant_conv + decoder.*, all F32)
  clip_l.psd1   (~160 MiB: embeddings F16, weights Q8_0, norms/biases F32)
  clip_g.psd1   (~755 MiB: embeddings F16, weights Q8_0, norms/biases F32)
  unet.psd1     (~2605 MiB: large weights Q8_0, conv_in/conv_out/norms/biases F32)

Usage:
  python diffusion/pc/convert_sdxl.py [owner/repo] [out_name] [--keep-cache] [--force]
  python diffusion/pc/convert_sdxl.py Ine007/waiNSFWIllustrious_v140 waiNSFWIllustrious_v140_8step ^
      --lora ByteDance/Hyper-SD/Hyper-SDXL-8steps-CFG-lora.safetensors --lora-scale 0.8 --unet-only
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPO = "Ine007/waiNSFWIllustrious_v140"
DEFAULT_NAME = "waiNSFWIllustrious_v140"


def quantize_q8_0(w: np.ndarray):
    N, K = w.shape
    assert K % 32 == 0, f"K={K} not divisible by 32"
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    g = w.reshape(N, K // 32, 32)
    sc = (np.max(np.abs(g), axis=2) / 127.0).astype(np.float16)
    sc_f32 = sc.astype(np.float32)
    inv = np.divide(1.0, sc_f32, out=np.zeros_like(sc_f32), where=sc_f32 > 0.0)
    q = np.clip(np.round(g * inv[:, :, None]), -127, 127).astype(np.int8)
    return np.ascontiguousarray(q.reshape(N, K)), np.ascontiguousarray(sc)


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl.exe", "-f", "-L", "-A", "Mozilla/5.0", "-C", "-", "--retry", "5", "-o", str(dest), url]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_st_tensor(sf, base: int, entry: dict) -> np.ndarray:
    a, b = entry["data_offsets"]
    sf.seek(base + a)
    raw = sf.read(b - a)
    dt = entry["dtype"]
    shape = entry["shape"]
    if dt == "F16":
        return np.frombuffer(raw, np.float16).astype(np.float32).reshape(shape)
    if dt == "BF16":
        return ((np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)).reshape(shape)
    if dt == "F32":
        return np.frombuffer(raw, np.float32).copy().reshape(shape)
    raise ValueError(f"unsupported dtype {dt}")


def load_kohya_unet_lora(lora_path: Path):
    """Load Kohya-format SDXL UNet LoRA safetensors into {prefix: (down, up, alpha)}."""
    layers = {}
    with open(lora_path, "rb") as sf:
        hlen = struct.unpack("<Q", sf.read(8))[0]
        header = json.loads(sf.read(hlen).decode("utf-8"))
        base = 8 + hlen
        prefixes = sorted({k.rsplit(".lora_down.weight", 1)[0]
                           for k in header if k.endswith(".lora_down.weight") and k.startswith("lora_unet_")})
        for pfx in prefixes:
            down = load_st_tensor(sf, base, header[f"{pfx}.lora_down.weight"])
            up = load_st_tensor(sf, base, header[f"{pfx}.lora_up.weight"])
            ak = f"{pfx}.alpha"
            alpha = float(load_st_tensor(sf, base, header[ak]).item()) if ak in header else float(down.shape[0])
            layers[pfx] = (down, up, alpha)
    print(f"Loaded {len(layers)} UNet LoRA layers from {lora_path.name}", flush=True)
    return layers


def apply_lora_to_weight(w: np.ndarray, key: str, lora_layers: dict, lora_scale: float) -> bool:
    if not lora_layers or not key.endswith(".weight"):
        return False
    pfx = "lora_unet_" + key[:-7].replace(".", "_")
    item = lora_layers.get(pfx)
    if item is None:
        return False
    down, up, alpha = item
    rank = down.shape[0]
    mult = lora_scale * (alpha / float(rank))
    if w.ndim == 2:
        delta = up.reshape(w.shape[0], rank) @ down.reshape(rank, w.shape[1])
    elif w.ndim == 4:
        # Handles both 1x1 and 3x3 Conv2d LoRA (down: [r, cin, kh, kw], up: [cout, r, 1, 1] or vice versa)
        if down.shape[2:] == w.shape[2:] and up.shape[2:] == (1, 1):
            delta = (up.reshape(w.shape[0], rank) @ down.reshape(rank, -1)).reshape(w.shape)
        elif up.shape[2:] == w.shape[2:] and down.shape[2:] == (1, 1):
            # up: [cout, r, kh, kw], down: [r, cin, 1, 1] -> [cout, cin, kh, kw]
            delta = np.einsum("orkl,ri->oikl", up, down.reshape(rank, w.shape[1]))
        else:
            raise ValueError(f"Unsupported conv LoRA shapes for {key}: w={w.shape} down={down.shape} up={up.shape}")
    else:
        raise ValueError(f"Unexpected ndim {w.ndim} for LoRA target {key}")
    w += (mult * delta).astype(np.float32)
    return True


def normalize_tensor_name(k: str) -> str:
    """Normalize legacy Diffusers SD 1.5 VAE attention names to match modern to_q/to_k/to_v/to_out.0."""
    for old, new in [
        (".attentions.0.query.", ".attentions.0.to_q."),
        (".attentions.0.key.", ".attentions.0.to_k."),
        (".attentions.0.value.", ".attentions.0.to_v."),
        (".attentions.0.proj_attn.", ".attentions.0.to_out.0."),
    ]:
        if old in k:
            return k.replace(old, new)
    return k


def convert_local_safetensors(st_path: Path, out_path: Path, key_filter, quant_mode: str,
                              vae_scale: float = 0.13025, lora_layers: dict = None, lora_scale: float = 1.0):
    t0 = time.time()
    merged = 0
    with open(st_path, "rb") as sf:
        hlen = struct.unpack("<Q", sf.read(8))[0]
        header = json.loads(sf.read(hlen).decode("utf-8"))
        base = 8 + hlen

        key_pairs = sorted((normalize_tensor_name(k), k) for k in header if k != "__metadata__" and key_filter(k))
        n = len(key_pairs)
        rec_size = struct.calcsize("<96sQQQIIII")
        assert rec_size == 136
        header_end = 24 + n * rec_size
        pad0 = (-header_end) % 256
        data_start = header_end + pad0

        tmp = out_path.with_suffix(".psd1.part")
        recs = []
        offset = 0

        with open(tmp, "wb") as f:
            f.write(bytes(data_start))
            for idx, (out_k, k) in enumerate(key_pairs):
                e = header[k]
                a, b = e["data_offsets"]
                sf.seek(base + a)
                raw = sf.read(b - a)
                dt = e["dtype"]
                shape = e["shape"]
                is_emb = "embedding.weight" in out_k

                if is_emb and quant_mode != "f32":
                    if dt == "F16":
                        arr = np.frombuffer(raw, np.float16).copy().reshape(shape)
                    elif dt == "BF16":
                        arr = ((np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)).astype(np.float16).reshape(shape)
                    else:
                        arr = np.frombuffer(raw, np.float32).astype(np.float16).reshape(shape)
                    rows, cols = arr.shape
                    blob = np.ascontiguousarray(arr).tobytes()
                    fmt, scales_rel = 1, 0
                else:
                    if dt == "F16":
                        w = np.frombuffer(raw, np.float16).astype(np.float32).reshape(shape)
                    elif dt == "BF16":
                        w = ((np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)).reshape(shape)
                    elif dt == "F32":
                        w = np.frombuffer(raw, np.float32).copy().reshape(shape)
                    else:
                        raise ValueError(f"unsupported dtype {dt} for {k}")

                    if lora_layers and apply_lora_to_weight(w, k, lora_layers, lora_scale):
                        merged += 1

                    if w.ndim == 4:
                        w = np.ascontiguousarray(w.transpose(0, 2, 3, 1)).reshape(w.shape[0], -1)
                    if w.ndim == 2:
                        rows, cols = w.shape
                    elif w.ndim == 1:
                        rows, cols = w.shape[0], 0
                    else:
                        raise ValueError(f"unexpected ndim {w.ndim} for {k}")

                    use_q8 = (quant_mode == "q8" and w.ndim == 2 and cols % 32 == 0 and rows >= 32 and cols >= 32)
                    if use_q8:
                        q, sc = quantize_q8_0(w)
                        qbytes = q.tobytes()
                        spad = (-len(qbytes)) % 256
                        scales_rel = len(qbytes) + spad
                        blob = qbytes + bytes(spad) + sc.tobytes()
                        fmt = 2
                    else:
                        blob = np.ascontiguousarray(w, dtype=np.float32).tobytes()
                        fmt, scales_rel = 0, 0

                pad = (-offset) % 256
                if pad:
                    f.write(bytes(pad))
                    offset += pad
                f.write(blob)
                nb = out_k.encode("utf-8")
                assert len(nb) < 96, f"name too long: {out_k}"
                recs.append(struct.pack("<96sQQQIIII", nb, offset, len(blob), scales_rel, rows, cols, fmt, 0))
                offset += len(blob)

            f.seek(0)
            f.write(struct.pack("<4sIQfI", b"PSD1", n, offset, vae_scale, 0))
            for rec_bytes in recs:
                f.write(rec_bytes)

    tmp.replace(out_path)
    extra = f", merged {merged}/{len(lora_layers)} LoRA layers (scale={lora_scale})" if lora_layers else ""
    print(f"DONE {out_path.name}: {n} tensors{extra}, {out_path.stat().st_size / 2**20:.1f} MiB in {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?", default=DEFAULT_REPO)
    ap.add_argument("name", nargs="?", default=DEFAULT_NAME)
    ap.add_argument("--arch", choices=["sdxl", "sd15"], default="sdxl", help="Model architecture (sdxl or sd15)")
    ap.add_argument("--fp16-variant", action="store_true", help="Download .fp16.safetensors variants on HF")
    ap.add_argument("--no-vae", action="store_true", help="Skip vae.psd1 (shares VAE from sibling model of same arch)")
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--unet-only", action="store_true", help="Only build unet.psd1 (shares CLIP/VAE on PS4)")
    ap.add_argument("--lora", default="", help="Local .safetensors path OR Hugging Face owner/repo/file.safetensors")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--preset-desc", default="", help="Optional description for preset.cfg")
    args = ap.parse_args()

    out_dir = ROOT / "diffusion" / "models" / args.name
    cache_dir = ROOT / "diffusion" / "cache" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_layers = None
    lora_tmp = None
    if args.lora:
        lp = Path(args.lora)
        if lp.is_file():
            lora_layers = load_kohya_unet_lora(lp)
        else:
            parts = args.lora.split("/")
            if len(parts) < 3:
                raise SystemExit("--lora must be an existing file or owner/repo/path/to/file.safetensors")
            lrepo = "/".join(parts[:2])
            lfile = "/".join(parts[2:])
            lora_tmp = cache_dir / Path(lfile).name
            download_file(f"https://huggingface.co/{lrepo}/resolve/main/{lfile}", lora_tmp)
            lora_layers = load_kohya_unet_lora(lora_tmp)

    sfx = ".fp16.safetensors" if args.fp16_variant else ".safetensors"
    vae_scale = 0.18215 if args.arch == "sd15" else 0.13025
    jobs = []
    if not args.no_vae:
        jobs.append(("vae.psd1", f"vae/diffusion_pytorch_model{sfx}",
                     lambda k: k.startswith("decoder.") or k.startswith("post_quant_conv."), "f32"))
    jobs.append(("clip_l.psd1", f"text_encoder/model{sfx}",
                 lambda k: k != "text_model.embeddings.position_ids", "q8"))
    if args.arch == "sdxl":
        jobs.append(("clip_g.psd1", f"text_encoder_2/model{sfx}",
                     lambda k: k != "text_model.embeddings.position_ids", "q8"))
    jobs.append(("unet.psd1", f"unet/diffusion_pytorch_model{sfx}", lambda k: True, "q8"))
    if args.unet_only:
        jobs = [j for j in jobs if j[0] == "unet.psd1"]

    for fname, remote_path, filt, qmode in jobs:
        out_p = out_dir / fname
        if out_p.is_file() and not args.force:
            print(f"skip existing {out_p} ({out_p.stat().st_size / 2**20:.1f} MiB)")
            continue
        local_st = cache_dir / remote_path.replace("/", "_")
        url = f"https://huggingface.co/{args.repo}/resolve/main/{remote_path}"
        download_file(url, local_st)
        ll = lora_layers if fname == "unet.psd1" else None
        convert_local_safetensors(local_st, out_p, filt, qmode, vae_scale=vae_scale, lora_layers=ll, lora_scale=args.lora_scale)
        if not args.keep_cache and local_st.exists():
            os.remove(local_st)

    if lora_tmp and not args.keep_cache and lora_tmp.exists():
        os.remove(lora_tmp)

    # Write automatic preset.cfg for this model
    nl = args.name.lower()
    if args.arch == "sd15":
        w, h, ups = 512, 768, 1
        prompt = "masterpiece, best quality, 1girl, silver hair, blue eyes, glowing sakura petals, night sky, anime style"
        neg = "(worst quality, low quality:1.4), lowres, bad anatomy, bad hands, blurry, text, watermark"
        if "8step" in nl or "hyper" in nl:
            steps, cfg, smp, desc = 8, 3.5, "Euler", "SD 1.5 Fast 8-step (Hyper-SD15 · ~10s + 1080p)"
        elif "dreamshaper" in nl:
            steps, cfg, smp, desc = 20, 7.0, "Euler a", "SD 1.5 DreamShaper 8 · 2.5D fantasy/anime (~20s + 1080p)"
        elif "anything" in nl:
            steps, cfg, smp, desc = 20, 7.0, "Euler a", "SD 1.5 Anything V5 · Crisp cel-shaded anime (~20s + 1080p)"
        else:
            steps, cfg, smp, desc = 20, 7.0, "Euler a", "SD 1.5 MeinaMix V11 · Glossy classic anime (~20s + 1080p)"
    else:
        w, h, ups = 768, 1024, 0
        prompt = "masterpiece, best quality, amazing quality, 1girl, silver hair, blue eyes, glowing sakura petals, night sky, anime style"
        neg = "lowres, bad anatomy, bad hands, text, error, blurry, worst quality, low quality"
        if "dmd" in nl:
            steps, cfg, smp, desc = 6, 1.0, "LCM", f"SDXL {args.name} · DMD2 Ultra-Fast (6 steps, CFG 1.0, ~45s)"
        elif "8step" in nl or "hyper" in nl:
            steps, cfg, smp, desc = 8, 3.5, "Euler", f"SDXL {args.name} · Hyper-SD 8-step (Euler, CFG 3.5)"
        elif "animagine" in nl:
            steps, cfg, smp, desc = 20, 5.0, "Euler a", "SDXL Animagine XL 4.0 · Clean modern cel-shaded anime"
        elif "noob" in nl:
            steps, cfg, smp, desc = 20, 5.5, "Euler a", "SDXL NoobAI-XL 1.1 · Crisp Danbooru anime"
        else:
            steps, cfg, smp, desc = 20, 7.0, "Euler a", f"SDXL {args.name} · Full 20-step reference"
    if args.preset_desc:
        desc = args.preset_desc
    (out_dir / "preset.cfg").write_text(
        f"arch={args.arch}\nwidth={w}\nheight={h}\nsteps={steps}\ncfg={cfg}\n"
        f"sampler={smp}\nupscale={ups}\ndescription={desc}\nprompt={prompt}\nnegative={neg}\n",
        encoding="utf-8",
    )
    print(f"WROTE {out_dir / 'preset.cfg'}", flush=True)


if __name__ == "__main__":
    main()
