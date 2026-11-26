#!/usr/bin/env python3
"""One-click Build + Test + Upload to PS4 script for GarlicDiffusion.

Usage:
  python pc/build_and_push.py                  # Uses PS4_IP env or 192.168.1.29
  python pc/build_and_push.py --ip 192.168.1.50
  python pc/build_and_push.py --no-upload      # Build PKG only
"""
from __future__ import annotations

import argparse
import ftplib
import os
import shutil
import subprocess
import sys
from pathlib import Path

DIFF = Path(__file__).resolve().parent.parent
PARENT = DIFF.parent


def find_zig() -> Path:
    env_tc = os.environ.get("GARLIC_TOOLCHAIN")
    if env_tc:
        return Path(env_tc) / "zig" / ("zig.exe" if os.name == "nt" else "zig")
    if (DIFF / "toolchain" / "zig").is_dir():
        return DIFF / "toolchain" / "zig" / ("zig.exe" if os.name == "nt" else "zig")
    return PARENT / "toolchain" / "zig" / ("zig.exe" if os.name == "nt" else "zig")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile GarlicDiffusion PKG and upload to PS4")
    ap.add_argument("--ip", default=os.environ.get("PS4_IP", "192.168.1.29"), help="PS4 IP address")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PS4_FTP_PORT") or 2121), help="GoldHEN FTP port")
    ap.add_argument("--skip-tests", action="store_true", help="Skip host C/JS verification")
    ap.add_argument("--no-upload", action="store_true", help="Compile PKG without FTP upload")
    args = ap.parse_args()

    if not args.skip_tests:
        node = shutil.which("node")
        if node:
            html_txt = (DIFF / "assets" / "index.html").read_text(encoding="utf-8")
            js = html_txt.split("<script>", 1)[1].rsplit("</script>", 1)[0]
            tmp_js = DIFF / "build_check_ui.tmp.js"
            try:
                tmp_js.write_text(js, encoding="utf-8")
                subprocess.run([node, "--check", str(tmp_js)], check=True)
                print("[OK] Web UI JavaScript syntax verified.")
            finally:
                tmp_js.unlink(missing_ok=True)

        zig = find_zig()
        if zig.is_file() and (PARENT / "diffusion").is_dir():
            out_exe = PARENT / "build" / ("sdxl_test.exe" if os.name == "nt" else "sdxl_test")
            out_exe.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    str(zig), "cc", "-O2", "-Wall", "-Idiffusion/src", "-DSDXL_CPU",
                    "diffusion/tests/sdxl_test.c", "diffusion/src/clip_tok.c",
                    "diffusion/src/gallery.c", "diffusion/src/model_dl.c", "diffusion/src/sdxl.c",
                    "-o", str(out_exe),
                ],
                cwd=PARENT,
                check=True,
            )
            subprocess.run([str(out_exe)], cwd=PARENT, check=True)

    print("\n==> Building PS4 PKG...")
    subprocess.run([sys.executable, str(DIFF / "build_diffusion.py")], cwd=DIFF, check=True)

    pkg_candidates = [
        PARENT / "build" / "diffusion" / "IV0000-PSAI00003_00-PS4LOCALAIDIFF03.pkg",
        DIFF / "build" / "IV0000-PSAI00003_00-PS4LOCALAIDIFF03.pkg",
    ]
    pkg = next((p for p in pkg_candidates if p.is_file()), None)
    if not pkg:
        sys.exit("ERROR: Built PKG file not found!")
    print(f"[OK] Built PKG: {pkg} ({pkg.stat().st_size / 1048576:.2f} MiB)")

    if args.no_upload:
        return

    print(f"\n==> Uploading PKG to PS4 ({args.ip}:{args.port} -> /data/pkg/Garlic-Diffusion.pkg)...")
    ftp = ftplib.FTP()
    ftp.connect(args.ip, args.port, timeout=20)
    ftp.login()
    try:
        ftp.mkd("/data/pkg")
    except Exception:
        pass
    with pkg.open("rb") as f:
        ftp.storbinary("STOR /data/pkg/Garlic-Diffusion.pkg", f, blocksize=1 << 20)

    if (DIFF / "models").is_dir():
        for cfg in (DIFF / "models").glob("*/preset.cfg"):
            try:
                with cfg.open("rb") as cf:
                    ftp.storbinary(f"STOR /data/garlic-diffusion/models/{cfg.parent.name}/preset.cfg", cf)
            except Exception:
                pass
    ftp.quit()
    print(f"[DONE] Uploaded /data/pkg/Garlic-Diffusion.pkg to {args.ip}! Install via GoldHEN -> Package Installer.")


if __name__ == "__main__":
    main()
