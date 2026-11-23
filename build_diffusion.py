"""Build GarlicDiffusion (PSAI00003) as a standalone PS4 / PS4 Pro Neo fake PKG.

Works both standalone (after cloning DJTOMATO/GarlicDiffusion) and inside a parent garlic workspace.
Set GARLIC_TOOLCHAIN or place OpenOrbis/PS4Toolchain, zig, and dotnet under ./toolchain (or ../toolchain).

Usage:
  python build_diffusion.py             -> build/IV0000-PSAI00003_00-PS4LOCALAIDIFF03.pkg
  python build_diffusion.py --upload    -> also uploads to /data/pkg/Garlic-Diffusion.pkg on PS4_IP
"""
import ftplib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw

DIFF = Path(__file__).resolve().parent
PARENT = DIFF.parent

# Resolve toolchain from GARLIC_TOOLCHAIN, ./toolchain, or ../toolchain
def find_toolchain() -> Path:
    env_tc = os.environ.get("GARLIC_TOOLCHAIN")
    if env_tc:
        return Path(env_tc)
    if (DIFF / "toolchain").is_dir():
        return DIFF / "toolchain"
    return PARENT / "toolchain"

TOOLCHAIN = find_toolchain()
OO = TOOLCHAIN / "OpenOrbis" / "PS4Toolchain"
WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
ZIG = TOOLCHAIN / "zig" / f"zig{EXE}"
DOTNET = TOOLCHAIN / "dotnet" / "dotnet.exe"
BIN = OO / ("bin/windows" if WIN else "bin/linux")
OUT = (PARENT / "build" / "diffusion") if (PARENT / "toolchain").is_dir() and not (DIFF / "toolchain").is_dir() else (DIFF / "build")
PKGDIR = OUT / "pkg"

PS4_IP = os.environ.get("PS4_IP", "")
FTP_PORT = int(os.environ.get("PS4_FTP_PORT") or 2121)

CFLAGS = [
    "-target", "x86_64-freebsd-none", "-march=btver2", "-O2", "-fPIC", "-funwind-tables",
    "-fno-stack-protector", "-fshort-wchar", "-nostdinc", "-isystem", str(OO / "include"), "-Wall", "-c",
]
ENV = dict(os.environ, DOTNET_ROLL_FORWARD="Major", OO_PS4_TOOLCHAIN=str(OO),
           DOTNET_SYSTEM_GLOBALIZATION_INVARIANT="1")

TITLE, VERSION, TITLE_ID = "GarlicDiffusion", "1.00", "PSAI00003"
CONTENT_ID = f"IV0000-{TITLE_ID}_00-PS4LOCALAIDIFF03"
REMOTE = "/data/pkg/Garlic-Diffusion.pkg"

SOURCES = [
    DIFF / "src" / "clip_tok.c",
    DIFF / "src" / "gallery.c",
    DIFF / "src" / "model_dl.c",
    DIFF / "src" / "sdxl.c",
    DIFF / "src" / "http_srv.c",
    DIFF / "src" / "main.c",
]
LIBS = [
    "-lc", "-lm", "-lkernel", "-lSceGnmDriver", "-lSceVideoOut", "-lScePad",
    "-lSceUserService", "-lSceSysmodule", "-lSceImeDialog", "-lSceCommonDialog",
    "-lSceSystemService", "-lSceNet", "-lSceSsl", "-lSceHttp",
]


def run(cmd, **kw):
    print(">", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, env=ENV, **kw)


def pkgtool(*args):
    if WIN:
        run([DOTNET, BIN / "PkgTool.Core.dll", *args], cwd=PKGDIR)
    else:
        run([BIN / "PkgTool.Core", *args], cwd=PKGDIR)


def ps4_ftp(timeout=30):
    if not PS4_IP:
        sys.exit("Set the PS4_IP environment variable to your console's address (and start GoldHEN's FTP server).")
    ftp = ftplib.FTP()
    ftp.connect(PS4_IP, FTP_PORT, timeout=timeout)
    ftp.login()
    return ftp


def make_icons(sce_sys: Path):
    ic = Image.new("RGB", (512, 512), (16, 24, 48))
    d = ImageDraw.Draw(ic)
    for y in range(512):
        r = int(16 + 48 * y / 512)
        g = int(28 + 30 * y / 512)
        bl = int(68 + 110 * y / 512)
        d.line([(0, y), (511, y)], fill=(r, g, bl))
    d.ellipse([96, 96, 416, 416], outline=(100, 195, 255), width=14)
    d.ellipse([156, 156, 356, 356], fill=(95, 178, 255), outline=(235, 245, 255), width=8)
    ic.save(sce_sys / "icon0.png", optimize=True)

    bg = Image.new("RGB", (1920, 1080), (10, 16, 29))
    db = ImageDraw.Draw(bg)
    for y in range(1080):
        r = int(10 + 24 * y / 1080)
        g = int(16 + 18 * y / 1080)
        bl = int(32 + 52 * y / 1080)
        db.line([(0, y), (1919, y)], fill=(r, g, bl))
    bg.save(sce_sys / "pic1.png", optimize=True)


def main():
    if (DIFF / "pc" / "export_sdxl_kernels.py").is_file() and ZIG.is_file():
        subprocess.run([sys.executable, str(DIFF / "pc" / "export_sdxl_kernels.py")], check=True)
    if not (DIFF / "assets" / "clip_tok.bin").is_file():
        subprocess.run([sys.executable, str(DIFF / "pc" / "export_clip_tok.py")], check=True)

    shutil.rmtree(PKGDIR, ignore_errors=True)
    (PKGDIR / "sce_sys" / "about").mkdir(parents=True)
    (PKGDIR / "sce_module").mkdir()
    shutil.copytree(DIFF / "assets", PKGDIR / "assets")

    objdir = OUT / "obj"
    objdir.mkdir(parents=True, exist_ok=True)
    objs = []
    for src in SOURCES:
        obj = objdir / (src.stem + ".o")
        run([ZIG, "cc", *CFLAGS, f"-I{DIFF / 'src'}", str(src), "-o", str(obj)])
        objs.append(obj)

    elf = OUT / "gdiff.elf"
    run([ZIG, "ld.lld", "-m", "elf_x86_64", "-pie", "--script", OO / "link.x", "--eh-frame-hdr",
         *objs, "-L", OO / "lib", *LIBS, OO / "lib" / "crt1.o", "-o", elf])
    run([BIN / f"create-fself{EXE}", f"-in={elf}", f"-out={OUT / 'gdiff.oelf'}",
         f"--eboot={PKGDIR / 'eboot.bin'}", "--paid", "0x3800000000000011"])

    sample = OO / "samples" / "hello_world"
    shutil.copy(sample / "sce_sys" / "about" / "right.sprx", PKGDIR / "sce_sys" / "about")
    make_icons(PKGDIR / "sce_sys")
    for prx in (sample / "sce_module").glob("*.prx"):
        shutil.copy(prx, PKGDIR / "sce_module")

    sfo = "sce_sys/param.sfo"
    pkgtool("sfo_new", sfo)
    for key, typ, size, value in [
        ("APP_TYPE", "Integer", 4, "1"),
        ("APP_VER", "Utf8", 8, VERSION),
        ("ATTRIBUTE", "Integer", 4, "8388608"),  # PS4 Base + PS4 Pro Neo mode enabled!
        ("CATEGORY", "Utf8", 4, "gd"),
        ("CONTENT_ID", "Utf8", 48, CONTENT_ID),
        ("DOWNLOAD_DATA_SIZE", "Integer", 4, "0"),
        ("SYSTEM_VER", "Integer", 4, "0"),
        ("TITLE", "Utf8", 128, TITLE),
        ("TITLE_ID", "Utf8", 12, TITLE_ID),
        ("VERSION", "Utf8", 8, VERSION),
    ]:
        pkgtool("sfo_setentry", sfo, key, "--type", typ, "--maxsize", str(size), "--value", value)

    files = ["eboot.bin", sfo, "sce_sys/about/right.sprx", "sce_sys/icon0.png", "sce_sys/pic1.png"]
    files += [f"sce_module/{p.name}" for p in (PKGDIR / "sce_module").glob("*.prx")]
    files += [p.relative_to(PKGDIR).as_posix() for p in sorted((PKGDIR / "assets").rglob("*")) if p.is_file()]
    run([BIN / f"create-gp4{EXE}", "-out", "pkg.gp4", f"--content-id={CONTENT_ID}", "--files", " ".join(files)],
        cwd=PKGDIR)
    pkgtool("pkg_build", "pkg.gp4", str(OUT))
    pkg = OUT / f"{CONTENT_ID}.pkg"
    print(f"BUILT {pkg} ({pkg.stat().st_size} bytes)")

    if "--upload" in sys.argv:
        ftp = ps4_ftp()
        with open(pkg, "rb") as f:
            ftp.storbinary(f"STOR {REMOTE}", f, blocksize=1 << 20)
        print(f"UPLOADED to {REMOTE}")
        ftp.quit()


if __name__ == "__main__":
    main()
