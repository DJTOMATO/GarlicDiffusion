#!/usr/bin/env python3
"""Obfuscate `src/preset_seeds.h` and `src/randomizer_pools.h` in-place for Git commits.

- Encodes all string literals (including adult/NSFW Danbooru test tags, presets, characters,
  and 1,050 randomizer phrases) into rolling-XOR byte blobs (`GD_SEEDS_BLOB` / `GD_POOLS_BLOB`)
  so zero plaintext words appear in Git diffs, GitHub search, or raw source files.
- Remains 100% directly compilable by `zig cc` / `build_diffusion.py` out of the box!
- Pair with `python pc/deobfuscate_seeds.py` anytime you want readable headers locally.
"""
from __future__ import annotations

import ast
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


def encode_blob(strings: list[str]) -> tuple[bytes, list[int]]:
    raw = bytearray()
    offsets: list[int] = []
    memo: dict[str, int] = {}
    for s in strings:
        if s in memo:
            offsets.append(memo[s])
            continue
        off = len(raw)
        memo[s] = off
        offsets.append(off)
        raw.extend(s.encode("utf-8"))
        raw.append(0)
    enc = bytes((b ^ mask_byte(i)) for i, b in enumerate(raw))
    return enc, offsets


def fmt_byte_array(name: str, data: bytes) -> str:
    lines = [f"static uint8_t {name}[{len(data)}] = {{"]
    for i in range(0, len(data), 20):
        chunk = ", ".join(f"0x{b:02x}" for b in data[i : i + 20])
        lines.append(f"    {chunk},")
    lines.append("};")
    return "\n".join(lines)


def c_unescape_str(token: str) -> str:
    return ast.literal_eval(token)


def parse_c_brace_tuples(block: str) -> list[list[str]]:
    items: list[list[str]] = []
    i, n = 0, len(block)
    while i < n:
        if block[i] == "/" and i + 1 < n and block[i + 1] == "/":
            while i < n and block[i] != "\n":
                i += 1
            continue
        if block[i] == "{":
            i += 1
            fields: list[str] = []
            cur: list[str] = []
            in_str = False
            esc = False
            while i < n:
                ch = block[i]
                if in_str:
                    cur.append(ch)
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == "/" and i + 1 < n and block[i + 1] == "/":
                        while i < n and block[i] != "\n":
                            i += 1
                        continue
                    if ch == '"':
                        in_str = True
                        cur.append(ch)
                    elif ch == ",":
                        fields.append("".join(cur).strip())
                        cur = []
                    elif ch == "}":
                        tail = "".join(cur).strip()
                        if tail:
                            fields.append(tail)
                        break
                    else:
                        cur.append(ch)
                i += 1
            if fields:
                items.append(fields)
        i += 1
    return items


def eval_c_str_concat(expr: str) -> str:
    parts = re.findall(r'"(?:\\.|[^"\\])*"', expr)
    return "".join(c_unescape_str(p) for p in parts)


def obfuscate_presets() -> None:
    txt = PRESET_H.read_text(encoding="utf-8")
    if "#define GD_PRESETS_OBFUSCATED 1" in txt:
        print("preset_seeds.h: already obfuscated (OK)")
        return

    m_p = re.search(r"SEED_PRESETS\[\]\s*=\s*\{(.*?)\n\};", txt, re.S)
    m_c = re.search(r"SEED_CHARS\[\]\s*=\s*\{(.*?)\n\};", txt, re.S)
    if not m_p or not m_c:
        sys.exit("ERROR: Could not locate SEED_PRESETS[] or SEED_CHARS[] in preset_seeds.h")

    raw_presets = parse_c_brace_tuples(m_p.group(1))
    raw_chars = parse_c_brace_tuples(m_c.group(1))

    seen_ids: set[str] = set()
    presets = []
    for f in raw_presets:
        if len(f) < 12:
            continue
        pid = eval_c_str_concat(f[0])
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        presets.append({
            "id": pid,
            "name": eval_c_str_concat(f[1]),
            "model": eval_c_str_concat(f[2]),
            "prompt": eval_c_str_concat(f[3]),
            "neg": eval_c_str_concat(f[4]),
            "smp": eval_c_str_concat(f[5]),
            "w": int(f[6].rstrip("uU")),
            "h": int(f[7].rstrip("uU")),
            "st": int(f[8].rstrip("uU")),
            "cfg": float(f[9].rstrip("fF")),
            "ups": int(f[10]),
            "eco": int(f[11]),
            "nsfw": int(f[12]) if len(f) > 12 else 0,
            "spd": int(f[13]) if len(f) > 13 else 0,
        })

    seen_cids: set[str] = set()
    chars = []
    for f in raw_chars:
        if len(f) < 5:
            continue
        cid = eval_c_str_concat(f[0])
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        chars.append({
            "id": cid,
            "name": eval_c_str_concat(f[1]),
            "ser": eval_c_str_concat(f[2]),
            "prompt": eval_c_str_concat(f[3]),
            "neg": eval_c_str_concat(f[4]),
        })

    all_strs: list[str] = []
    for p in presets:
        all_strs.extend([p["id"], p["name"], p["model"], p["prompt"], p["neg"], p["smp"]])
    for c in chars:
        all_strs.extend([c["id"], c["name"], c["ser"], c["prompt"], c["neg"]])

    blob, offs = encode_blob(all_strs)
    it = iter(offs)

    p_rows = []
    for p in presets:
        o_id, o_nm, o_md, o_pr, o_ng, o_sm = next(it), next(it), next(it), next(it), next(it), next(it)
        p_rows.append(
            f"    {{ {o_id}u, {o_nm}u, {o_md}u, {o_pr}u, {o_ng}u, {o_sm}u, "
            f"{p['w']}u, {p['h']}u, {p['st']}u, {p['cfg']:.2f}f, {p['ups']}, {p['eco']}, {p['nsfw']}, {p['spd']} }},"
        )

    c_rows = []
    for c in chars:
        o_id, o_nm, o_sr, o_pr, o_ng = next(it), next(it), next(it), next(it), next(it)
        c_rows.append(f"    {{ {o_id}u, {o_nm}u, {o_sr}u, {o_pr}u, {o_ng}u }},")

    out = [
        "// Obfuscated Preset & Character Seed Table (compiles directly; run `python pc/deobfuscate_seeds.py` to edit)",
        "#ifndef PRESET_SEEDS_H",
        "#define PRESET_SEEDS_H",
        "#define GD_PRESETS_OBFUSCATED 1",
        "",
        "#include <stdint.h>",
        "#include <stdio.h>",
        "",
        fmt_byte_array("GD_SEEDS_BLOB", blob),
        "",
        "typedef struct {",
        "    uint32_t id, name, model, prompt, neg, smp, w, h, st;",
        "    float cfg;",
        "    int ups, eco, nsfw, spd;",
        "} GdPresetPacked;",
        "",
        "typedef struct {",
        "    uint32_t id, name, ser, prompt, neg;",
        "} GdCharPacked;",
        "",
        f"static const GdPresetPacked GD_PRESETS_PK[{len(presets)}] = {{",
        *p_rows,
        "};",
        "",
        f"static const GdCharPacked GD_CHARS_PK[{len(chars)}] = {{",
        *c_rows,
        "};",
        "",
        f"static UserPreset SEED_PRESETS[{len(presets)}];",
        f"static CharacterItem SEED_CHARS[{len(chars)}];",
        "",
        "static inline void gd_init_preset_seeds(void) {",
        "    static int inited = 0;",
        "    if (inited) return;",
        "    inited = 1;",
        "    for (size_t i = 0; i < sizeof(GD_SEEDS_BLOB); i++) {",
        "        uint32_t v = (uint32_t)(i * 2654435761u);",
        "        uint8_t m = (uint8_t)(((v >> 16) ^ (v & 0xFFu) ^ ((i * 73u) & 0xFFu) ^ 0xA7u) & 0xFFu);",
        "        GD_SEEDS_BLOB[i] ^= m;",
        "    }",
        "    const char* b = (const char*)GD_SEEDS_BLOB;",
        f"    for (size_t i = 0; i < {len(presets)}u; i++) {{",
        "        const GdPresetPacked* s = &GD_PRESETS_PK[i];",
        "        UserPreset* d = &SEED_PRESETS[i];",
        "        snprintf(d->id, sizeof(d->id), \"%s\", b + s->id);",
        "        snprintf(d->name, sizeof(d->name), \"%s\", b + s->name);",
        "        snprintf(d->model, sizeof(d->model), \"%s\", b + s->model);",
        "        snprintf(d->prompt, sizeof(d->prompt), \"%s\", b + s->prompt);",
        "        snprintf(d->negative, sizeof(d->negative), \"%s\", b + s->neg);",
        "        snprintf(d->sampler, sizeof(d->sampler), \"%s\", b + s->smp);",
        "        d->width = s->w; d->height = s->h; d->steps = s->st; d->cfg = s->cfg;",
        "        d->upscale = s->ups; d->eco = s->eco; d->nsfw = s->nsfw; d->is_speed = s->spd;",
        "    }",
        f"    for (size_t i = 0; i < {len(chars)}u; i++) {{",
        "        const GdCharPacked* s = &GD_CHARS_PK[i];",
        "        CharacterItem* d = &SEED_CHARS[i];",
        "        snprintf(d->id, sizeof(d->id), \"%s\", b + s->id);",
        "        snprintf(d->name, sizeof(d->name), \"%s\", b + s->name);",
        "        snprintf(d->series, sizeof(d->series), \"%s\", b + s->ser);",
        "        snprintf(d->prompt, sizeof(d->prompt), \"%s\", b + s->prompt);",
        "        snprintf(d->negative, sizeof(d->negative), \"%s\", b + s->neg);",
        "    }",
        "}",
        "",
        "#endif",
        "",
    ]
    PRESET_H.write_text("\n".join(out), encoding="utf-8")
    print(f"preset_seeds.h: obfuscated {len(presets)} presets + {len(chars)} characters ({len(blob)} bytes blob)")


def obfuscate_pools() -> None:
    txt = POOLS_H.read_text(encoding="utf-8")
    if "#define GD_POOLS_OBFUSCATED 1" in txt:
        print("randomizer_pools.h: already obfuscated (OK)")
        return

    pools: dict[str, list[str]] = {}
    all_strs: list[str] = []
    for name in POOL_NAMES:
        m = re.search(rf"{name}\[\]\s*=\s*\{{(.*?)\n\}};", txt, re.S)
        if not m:
            sys.exit(f"ERROR: Could not locate {name}[] in randomizer_pools.h")
        lits = [c_unescape_str(tok) for tok in re.findall(r'"(?:\\.|[^"\\])*"', m.group(1))]
        if not lits:
            sys.exit(f"ERROR: Empty pool {name}")
        pools[name] = lits
        all_strs.extend(lits)

    blob, offs = encode_blob(all_strs)
    it = iter(offs)

    decls = []
    init_lines = []
    for name in POOL_NAMES:
        cnt = len(pools[name])
        poffs = [next(it) for _ in range(cnt)]
        off_str = ", ".join(f"{o}u" for o in poffs)
        decls.append(f"#define {name}_N {cnt}")
        decls.append(f"static const char* {name}[{cnt}];")
        decls.append(f"static const uint32_t GD_OFF_{name}[{cnt}] = {{ {off_str} }};")
        init_lines.append(f"    for (int i = 0; i < {cnt}; i++) {name}[i] = b + GD_OFF_{name}[i];")

    out = [
        "// Obfuscated Batch Prompt Randomizer Pools (compiles directly; run `python pc/deobfuscate_seeds.py` to edit)",
        "#ifndef RANDOMIZER_POOLS_H",
        "#define RANDOMIZER_POOLS_H",
        "#define GD_POOLS_OBFUSCATED 1",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
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
        fmt_byte_array("GD_POOLS_BLOB", blob),
        "",
        *decls,
        "",
        "static inline void gd_init_randomizer_pools(void) {",
        "    static int inited = 0;",
        "    if (inited) return;",
        "    inited = 1;",
        "    for (size_t i = 0; i < sizeof(GD_POOLS_BLOB); i++) {",
        "        uint32_t v = (uint32_t)(i * 2654435761u);",
        "        uint8_t m = (uint8_t)(((v >> 16) ^ (v & 0xFFu) ^ ((i * 73u) & 0xFFu) ^ 0xA7u) & 0xFFu);",
        "        GD_POOLS_BLOB[i] ^= m;",
        "    }",
        "    const char* b = (const char*)GD_POOLS_BLOB;",
        *init_lines,
        "}",
        "",
        "#endif",
        "",
    ]
    POOLS_H.write_text("\n".join(out), encoding="utf-8")
    print(f"randomizer_pools.h: obfuscated {len(all_strs)} phrases across {len(POOL_NAMES)} pools ({len(blob)} bytes blob)")


if __name__ == "__main__":
    obfuscate_presets()
    obfuscate_pools()
