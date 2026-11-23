"""Generate manifest.json + Hugging Face README and upload converted Garlic Diffusion models to Hugging Face.

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login   (paste a Write token from https://huggingface.co/settings/tokens)

Usage:
  # Generate manifest.json only (dry run):
  python diffusion/pc/upload_to_hf.py --dry-run

  # Upload selected (or all) converted models to your Hugging Face model repo:
  python diffusion/pc/upload_to_hf.py <your-hf-username>/garlic-diffusion-models
  python diffusion/pc/upload_to_hf.py <your-hf-username>/garlic-diffusion-models --models WAI_Illustrious_v17_dmd2 MeinaMix_V11_8step
"""
import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "diffusion" / "models"


def inspect_model(mdir: Path):
    unet = mdir / "unet.psd1"
    if not unet.is_file():
        return None
    with open(unet, "rb") as f:
        magic, n, dsz, _, _ = struct.unpack("<4sIQfI", f.read(24))
    if magic != b"PSD1":
        return None
    exp = ((24 + n * 136 + 255) & ~255) + dsz
    if unet.stat().st_size < exp:
        print(f"SKIP incomplete {mdir.name}")
        return None
    arch = "sd15" if dsz < (1500 << 20) else "sdxl"
    files = sorted(p.name for p in mdir.iterdir() if p.is_file() and p.name.endswith((".psd1", ".cfg")))
    total_mb = sum((mdir / fn).stat().st_size for fn in files) >> 20
    desc = f"{arch.upper()} checkpoint for Garlic Diffusion PS4"
    pcfg = mdir / "preset.cfg"
    if pcfg.is_file():
        for line in pcfg.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("description="):
                desc = line.split("=", 1)[1].strip()
                break
    return {
        "id": mdir.name,
        "title": mdir.name.replace("_", " "),
        "arch": arch,
        "size_mb": int(total_mb),
        "desc": desc,
        "files": ",".join(files),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", nargs="?", default="", help="Hugging Face repo ID, e.g. yourname/garlic-diffusion-models")
    ap.add_argument("--models", nargs="*", default=None, help="Specific model folder names to include (default: all complete)")
    ap.add_argument("--dry-run", action="store_true", help="Only write diffusion/models/manifest.json without uploading")
    ap.add_argument("--private", action="store_true", help="Create repo as private on Hugging Face")
    args = ap.parse_args()

    entries = []
    chosen = set(args.models) if args.models else None
    for mdir in sorted(MODELS_DIR.iterdir()):
        if not mdir.is_dir():
            continue
        if chosen and mdir.name not in chosen:
            continue
        info = inspect_model(mdir)
        if info:
            entries.append(info)

    manifest_path = MODELS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path} with {len(entries)} model(s):")
    for e in entries:
        print(f"  - {e['id']} ({e['arch']}, {e['size_mb']} MiB): {e['files']}")

    if args.dry_run or not args.repo_id:
        if not args.repo_id and not args.dry_run:
            print("\nPass a Hugging Face repo ID to upload, e.g.:")
            print("  python diffusion/pc/upload_to_hf.py <username>/garlic-diffusion-models")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Install huggingface_hub first: pip install huggingface_hub")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    readme = [
        "---",
        "tags:",
        "- ps4",
        "- garlic-diffusion",
        "- stable-diffusion",
        "- sdxl",
        "---",
        "# Garlic Diffusion PS4 Models",
        "",
        "Pre-converted Q8_0 `PSD1` checkpoints ready for **Garlic Diffusion** on PS4 and PS4 Pro.",
        "",
        "## Quick Install on PS4",
        f"Open **Garlic Diffusion Web UI → Models tab**, enter `{args.repo_id}` as the Hugging Face Repo, click **Save & Fetch Catalog**, and click **Download** on any model.",
        "",
        "## Included Models",
        "| Model | Arch | Size | Notes |",
        "|---|---|---:|---|",
    ]
    for e in entries:
        readme.append(f"| `{e['id']}` | {e['arch'].upper()} | {e['size_mb']} MiB | {e['desc']} |")
    readme_path = MODELS_DIR / "HF_README.md"
    readme_path.write_text("\n".join(readme) + "\n", encoding="utf-8")

    api.upload_file(path_or_fileobj=str(readme_path), path_in_repo="README.md", repo_id=args.repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=str(manifest_path), path_in_repo="manifest.json", repo_id=args.repo_id, repo_type="model")
    for e in entries:
        mdir = MODELS_DIR / e["id"]
        print(f"Uploading {e['id']} ({e['size_mb']} MiB)...", flush=True)
        api.upload_folder(folder_path=str(mdir), path_in_repo=e["id"], repo_id=args.repo_id, repo_type="model")
    print(f"\nDONE! Your PS4 catalog is live at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
