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

TITLE, VERSION, TITLE_ID = "GarlicDiffusion", "01.17", "PSAI00003"
CONTENT_ID = f"IV0000-{TITLE_ID}_00-PS4LOCALAIDIFF03"
REMOTE = "/data/pkg/Garlic-Diffusion.pkg"

SOURCES = [
    DIFF / "src" / "clip_tok.c",
    DIFF / "src" / "face_detect.c",
    DIFF / "src" / "gallery.c",
    DIFF / "src" / "model_dl.c",
    DIFF / "src" / "sdxl.c",
    DIFF / "src" / "http_srv.c",
    DIFF / "src" / "main.c",
]
LIBS = [
    "-lc", "-lm", "-lkernel", "-lSceGnmDriver", "-lSceVideoOut", "-lScePad",
    "-lSceUserService", "-lSceSysmodule", "-lSceImeDialog", "-lSceCommonDialog",
    "-lSceSystemService", "-lSceAppInstUtil", "-lSceNet", "-lSceSsl", "-lSceHttp",
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


def _bpath(start, *segments):
    pts = [start]
    for c1, c2, end in segments:
        p0 = pts[-1]
        for i in range(1, 25):
            t = i / 24
            a, b, c, e = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
            pts.append((a * p0[0] + b * c1[0] + c * c2[0] + e * end[0], a * p0[1] + b * c1[1] + c * c2[1] + e * end[1]))
    return pts


BULB = _bpath((50, 15), ((51, 23), (56, 29), (66, 35)), ((80, 43), (88, 55), (88, 67)), ((88, 81), (74, 90), (50, 90)),
              ((26, 90), (12, 81), (12, 67)), ((12, 55), (20, 43), (34, 35)), ((44, 29), (49, 23), (50, 15)))
CLOVE = _bpath((50, 21), ((40, 38), (35, 70), (43, 89.6)), ((47.67, 90.13), (52.33, 90.13), (57, 89.6)),
               ((65, 70), (60, 38), (50, 21)))
SEAMS = [_bpath((44, 31), ((27, 44), (22, 68), (31, 87))), _bpath((56, 31), ((73, 44), (78, 68), (69, 87)))]


def _garlic_mark(size):
    from PIL import ImageFilter
    S = size * 4
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    k = S / 100
    d.polygon([(x * k, y * k) for x, y in BULB], fill=(230, 218, 196, 255))
    d.polygon([(x * k, y * k) for x, y in CLOVE], fill=(246, 240, 228, 255))
    for seam in SEAMS:
        d.line([(x * k, y * k) for x, y in seam], fill=(201, 184, 156, 255), width=max(1, int(0.014 * S)), joint="curve")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse([0.43 * S, 0.40 * S, 0.49 * S, 0.66 * S], fill=150)
    mask = mask.filter(ImageFilter.GaussianBlur(0.025 * S))
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white.putalpha(Image.composite(mask, Image.new("L", img.size, 0), img.getchannel("A")))
    img.alpha_composite(white)
    shade = Image.new("L", img.size, 0)
    ImageDraw.Draw(shade).ellipse([0.14 * S, 0.72 * S, 0.86 * S, 0.96 * S], fill=70)
    shade = shade.filter(ImageFilter.GaussianBlur(0.05 * S))
    dark = Image.new("RGBA", img.size, (120, 98, 70, 255))
    dark.putalpha(Image.composite(shade, Image.new("L", img.size, 0), img.getchannel("A")))
    img.alpha_composite(dark)
    return img.resize((size, size), Image.LANCZOS)


def _blue_bg(w, h, glow_at=(0.5, 0.42), glow_r=0.75):
    import math
    img = Image.new("RGB", (w, h))
    px = img.load()
    gx, gy = glow_at[0] * w, glow_at[1] * h
    R = glow_r * max(w, h)
    for y in range(h):
        t = y / max(1, h - 1)
        top, bot = (11, 30, 64), (4, 10, 26)
        base = [top[i] + (bot[i] - top[i]) * t for i in range(3)]
        for x in range(w):
            dist = math.hypot(x - gx, y - gy) / R
            g = max(0.0, 1 - dist) ** 2
            px[x, y] = tuple(int(base[i] + g * (0, 75, 170)[i] * 0.58) for i in range(3))
    return img


def make_icons(sce_sys: Path):
    from PIL import ImageFilter, ImageFont
    W = 512
    ic = _blue_bg(W, W, (0.5, 0.40), 0.75).convert("RGBA")
    glow = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([116, 48, 396, 328], fill=(63, 169, 245, 95))
    ic.alpha_composite(glow.filter(ImageFilter.GaussianBlur(38)))
    g = _garlic_mark(280)
    ic.alpha_composite(g, (116, 34))
    font_path = PARENT / "pc" / "fonts_libres" / "Selawik" / "selawksb.ttf"
    if not font_path.is_file():
        for cand in [Path("C:/Windows/Fonts/seguisb.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")]:
            if cand.is_file():
                font_path = cand
                break
    d = ImageDraw.Draw(ic)
    if font_path.is_file():
        f1 = ImageFont.truetype(str(font_path), 54)
        f2 = ImageFont.truetype(str(font_path), 38)
        t1, t2 = "Garlic", "DIFFUSION"
        d.text(((W - d.textlength(t1, font=f1)) / 2, 332), t1, font=f1, fill=(255, 255, 255))
        d.text(((W - d.textlength(t2, font=f2)) / 2, 398), t2, font=f2, fill=(63, 169, 245))
    d.rectangle([W / 2 - 44, 464, W / 2 + 44, 468], fill=(61, 220, 151))
    ic.convert("RGB").save(sce_sys / "icon0.png", optimize=True)

    bg = _blue_bg(480, 270, (0.72, 0.55), 0.6).resize((1920, 1080), Image.BICUBIC).convert("RGBA")
    gb = _garlic_mark(620)
    gb.putalpha(gb.getchannel("A").point(lambda a: int(a * 0.16)))
    bg.alpha_composite(gb, (1150, 260))
    bg.convert("RGB").save(sce_sys / "pic1.png", optimize=True)


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
        ("APP_TYPE", "Integer", 4, "1"),  # Paid Standalone Full App (prevents CE-39929-2 Freemium SKU error)
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
