"""Compile shaders/sdxl.cl into assets/kernels.bin (PKRN format, zero register spills)."""
import os
import struct
import subprocess
from pathlib import Path

DIFF = Path(__file__).resolve().parent.parent
PARENT = DIFF.parent


def find_zig() -> Path:
    exe = ".exe" if os.name == "nt" else ""
    env_tc = os.environ.get("GARLIC_TOOLCHAIN")
    if env_tc:
        return Path(env_tc) / "zig" / f"zig{exe}"
    if (DIFF / "toolchain").is_dir():
        return DIFF / "toolchain" / "zig" / f"zig{exe}"
    return PARENT / "toolchain" / "zig" / f"zig{exe}"


ZIG = find_zig()
CLANG = [
    str(ZIG), "clang", "-x", "cl", "-cl-std=CL1.2", "-cl-no-stdinc",
    "-target", "amdgcn-amd-amdhsa", "-mcpu=bonaire", "-nogpulib",
    "-O3", "-ffast-math",
    "-Xclang", "-target-feature", "-Xclang", "-flat-for-global",
    "-Xclang", "-target-feature", "-Xclang", "-enable-ds128",
]

SRC = DIFF / "shaders" / "sdxl.cl"
OBJ = DIFF / "shaders" / "sdxl.o"
OUT = DIFF / "assets" / "kernels.bin"


def elf_symbols(obj: bytes):
    shoff, = struct.unpack_from("<Q", obj, 0x28)
    entsize, num, strndx = struct.unpack_from("<HHH", obj, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", obj, shoff + i * entsize) for i in range(num)]
    names = secs[strndx][4]
    secname = [obj[names + s[0]: obj.index(b"\0", names + s[0])].decode() for s in secs]
    syms = {}
    for s in secs:
        if s[1] != 2:  # SHT_SYMTAB
            continue
        strtab = secs[s[6]][4]
        for off in range(s[4], s[4] + s[5], 24):
            name_off, _info, _other, shndx, value, size = struct.unpack_from("<IBBHQQ", obj, off)
            if name_off and shndx < len(secs):
                name = obj[strtab + name_off: obj.index(b"\0", strtab + name_off)].decode()
                syms[name] = (secname[shndx], value, size)
    sections = {secname[i]: obj[s[4]: s[4] + s[5]] for i, s in enumerate(secs)}
    return syms, sections


class Kernel:
    def __init__(self, name: str, entry: int, kd: bytes):
        self.name, self.addr = name, entry
        self.lds, self.scratch, self.kernarg_size = struct.unpack_from("<III", kd, 0)
        rsrc1, rsrc2, props = struct.unpack_from("<IIH", kd, 48)
        if self.scratch:
            raise ValueError(f"{name} needs {self.scratch} bytes of scratch (spills); not supported")
        self.rsrc1 = rsrc1
        self.rsrc2 = rsrc2 | (((self.lds + 511) // 512) << 15)
        sizes = [(0, 4), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 1)]
        self.kernarg_sgpr, n = 0, 0
        for bit, size in sizes:
            if props & (1 << bit):
                if bit == 3:
                    self.kernarg_sgpr = n
                elif bit not in (0, 5):
                    raise ValueError(f"{name} wants user SGPR feature {bit}; not supported")
                n += size
        self.user_sgprs = n


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = CLANG + ["-c", str(SRC), "-o", str(OBJ)]
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)
    syms, sections = elf_symbols(OBJ.read_bytes())
    text = sections[".text"]
    recs = []
    for sym, (sec, value, _size) in sorted(syms.items()):
        if sym.endswith(".kd"):
            name = sym[:-3]
            entry = syms[name][1]
            assert entry % 256 == 0, f"{name} not 256-byte aligned"
            k = Kernel(name, entry, sections[sec][value: value + 64])
            print(f"  {name:18s} entry={entry:6d} lds={k.lds:5d} scratch={k.scratch} sgprs={k.user_sgprs}")
            recs.append(struct.pack("<32sIIIII", name.encode(), entry, k.rsrc1, k.rsrc2, k.user_sgprs, k.kernarg_sgpr))
    data = struct.pack("<4sII", b"PKRN", len(recs), len(text)) + b"".join(recs) + text
    OUT.write_bytes(data)
    print(f"wrote {OUT}: {len(recs)} kernels, {len(text)} bytes of GCN code")


if __name__ == "__main__":
    main()
