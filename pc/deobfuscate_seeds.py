#!/usr/bin/env python3
"""Deobfuscate `src/preset_seeds.h` and `src/randomizer_pools.h` back to clean, readable C code.

Use this locally whenever you want to inspect or edit presets, characters, or randomizer words.
Before committing to Git, run `python pc/obfuscate_seeds.py` (or use `sync_to_git.bat`, which
runs `obfuscate_seeds.py` automatically!).
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRESET_H = ROOT / "src" / "preset_seeds.h"
POOLS_H = ROOT / "src" / "randomizer_pools.h"

POOL_NAMES = [
    "RND_POSE", "RND_CAMERA", "RND_EXPR", "RND_SCENE",
    "RND_LIGHT", "RND_OUTFIT", "RND_DETAIL",
    "RND_ECCHI_POSE", "RND_ECCHI_EXPR", "RND_ECCHI_OUTFIT",
    "RND_ECCHI_SCENE", "RND_ECCHI_ACCENT",
]


def mask_byte(idx: int) -> int:
    v = ((idx * 2654435761) & 0xFFFFFFFF)
    return ((v >> 16) ^ (v & 0xFF) ^ ((idx * 73) & 0xFF) ^ 0xA7) & 0xFF


def decode_blob_from_text(txt: str, arr_name: str) -> bytes:
    m = re.search(rf"{arr_name}\[\d+\]\s*=\s*\{{(.*?)\}};", txt, re.S)
    if not m:
        sys.exit(f"ERROR: Could not find {arr_name} in header")
    enc = [int(h, 16) for h in re.findall(r"0x[0-9a-fA-F]{2}", m.group(1))]
    return bytes((b ^ mask_byte(i)) for i, b in enumerate(enc))


def read_cstr(blob: bytes, off: int) -> str:
    end = blob.index(0, off)
    return blob[off:end].decode("utf-8")


def c_quote(s: str) -> str:
    escaped = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def deobfuscate_presets() -> None:
    txt = PRESET_H.read_text(encoding="utf-8")
    if "#define GD_PRESETS_OBFUSCATED 1" not in txt:
        if "gd_init_preset_seeds" not in txt:
            txt = txt.replace("#endif", "static inline void gd_init_preset_seeds(void) {}\n\n#endif")
            PRESET_H.write_text(txt, encoding="utf-8")
        print("preset_seeds.h: already readable (OK)")
        return

    blob = decode_blob_from_text(txt, "GD_SEEDS_BLOB")
    mp = re.search(r"GD_PRESETS_PK\[\d+\]\s*=\s*\{(.*?)\};", txt, re.S)
    mc = re.search(r"GD_CHARS_PK\[\d+\]\s*=\s*\{(.*?)\};", txt, re.S)
    if not mp or not mc:
        sys.exit("ERROR: Missing GD_PRESETS_PK or GD_CHARS_PK")

    p_lines = []
    for row in re.findall(r"\{([^{}]+)\}", mp.group(1)):
        vals = [v.strip() for v in row.split(",") if v.strip()]
        o_id, o_nm, o_md, o_pr, o_ng, o_sm = (int(vals[k].rstrip("u")) for k in range(6))
        w, h, st = (int(vals[k].rstrip("u")) for k in range(6, 9))
        cfg = float(vals[9].rstrip("f"))
        ups, eco, nsfw, spd = (int(vals[k]) for k in range(10, 14))
        p_lines.append(
            f"    {{ {c_quote(read_cstr(blob, o_id))}, {c_quote(read_cstr(blob, o_nm))}, {c_quote(read_cstr(blob, o_md))},\n"
            f"      {c_quote(read_cstr(blob, o_pr))},\n"
            f"      {c_quote(read_cstr(blob, o_ng))},\n"
            f"      {c_quote(read_cstr(blob, o_sm))}, {w}, {h}, {st}, {cfg:.1f}f, {ups}, {eco}, {nsfw}, {spd} }},"
        )

    c_lines = []
    for row in re.findall(r"\{([^{}]+)\}", mc.group(1)):
        vals = [int(v.strip().rstrip("u")) for v in row.split(",") if v.strip()]
        o_id, o_nm, o_sr, o_pr, o_ng = vals[:5]
        c_lines.append(
            f"    {{ {c_quote(read_cstr(blob, o_id))}, {c_quote(read_cstr(blob, o_nm))}, {c_quote(read_cstr(blob, o_sr))},\n"
            f"      {c_quote(read_cstr(blob, o_pr))},\n"
            f"      {c_quote(read_cstr(blob, o_ng))} }},"
        )

    out = [
        "// Curated Default Prompt Presets & Characters (Readable Mode — run `python pc/obfuscate_seeds.py` before Git commit)",
        "#ifndef PRESET_SEEDS_H",
        "#define PRESET_SEEDS_H",
        "",
        "static const UserPreset SEED_PRESETS[] = {",
        *p_lines,
        "};",
        "",
        "static const CharacterItem SEED_CHARS[] = {",
        *c_lines,
        "};",
        "",
        "static inline void gd_init_preset_seeds(void) {}",
        "",
        "#endif",
        "",
    ]
    PRESET_H.write_text("\n".join(out), encoding="utf-8")
    print(f"preset_seeds.h: restored {len(p_lines)} presets + {len(c_lines)} characters to readable C!")


def deobfuscate_pools() -> None:
    txt = POOLS_H.read_text(encoding="utf-8")
    if "#define GD_POOLS_OBFUSCATED 1" not in txt:
        if "gd_init_randomizer_pools" not in txt:
            txt = txt.replace("#endif", "static inline void gd_init_randomizer_pools(void) {}\n\n#endif")
            POOLS_H.write_text(txt, encoding="utf-8")
        print("randomizer_pools.h: already readable (OK)")
        return

    blob = decode_blob_from_text(txt, "GD_POOLS_BLOB")
    sections = []
    total = 0
    for name in POOL_NAMES:
        m = re.search(rf"GD_OFF_{name}\[\d+\]\s*=\s*\{{(.*?)\}};", txt, re.S)
        if not m:
            sys.exit(f"ERROR: Missing GD_OFF_{name}")
        offs = [int(v.strip().rstrip("u")) for v in m.group(1).split(",") if v.strip()]
        strs = [c_quote(read_cstr(blob, o)) for o in offs]
        total += len(strs)
        rows = []
        for i in range(0, len(strs), 3):
            rows.append("    " + ", ".join(strs[i : i + 3]) + ",")
        sections.append(f"static const char* const {name}[] = {{\n" + "\n".join(rows) + "\n};")
        sections.append(f"#define {name}_N ((int)(sizeof({name}) / sizeof({name}[0])))\n")

    out = [
        "// Curated 525-item SFW & 525-item Ecchi/NSFW Batch Randomizer Pools (Readable Mode)",
        "#ifndef RANDOMIZER_POOLS_H",
        "#define RANDOMIZER_POOLS_H",
        "",
        "// Batch Randomizer bitflags (DiffJob.batch_rand_mask)",
        "#define RAND_POSE     (1u << 0)",
        "#define RAND_EXPR     (1u << 1)",
        "#define RAND_SCENE    (1u << 2)",
        "#define RAND_LIGHT    (1u << 3)",
        "#define RAND_OUTFIT   (1u << 4)",
        "#define RAND_DETAIL   (1u << 5)",
        "#define RAND_CAMERA   (1u << 6)",
        "#define RAND_ECCHI    (1u << 7)",
        "#define RAND_EXPLICIT (1u << 8)",
        "",
        *sections,
        "static inline void gd_init_randomizer_pools(void) {}",
        "",
        "#endif",
        "",
    ]
    POOLS_H.write_text("\n".join(out), encoding="utf-8")
    print(f"randomizer_pools.h: restored {total} phrases to readable C!")


if __name__ == "__main__":
    deobfuscate_presets()
    deobfuscate_pools()
