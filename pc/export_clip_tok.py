"""Export OpenAI CLIP BPE tokenizer (vocab.json + merges.txt) into diffusion/assets/clip_tok.bin.

Format (little-endian):
  magic:   "CTOK" (4 bytes)
  n_vocab: uint32 (49408)
  n_merges: uint32
  vocab strings: n_vocab * (uint16 len + UTF-8 bytes)
  merges: n_merges * (uint32 id_a, uint32 id_b, uint32 id_merged)
"""
import json
import struct
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "diffusion" / "assets" / "clip_tok.bin"
REPO = "Ine007/waiNSFWIllustrious_v140"


def fetch(path):
    url = f"https://huggingface.co/{REPO}/resolve/main/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req).read()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(fetch("tokenizer/vocab.json").decode("utf-8"))
    merges_txt = fetch("tokenizer/merges.txt").decode("utf-8").splitlines()
    # CLIP merges.txt has header line "#version: 0.2" and 48894 merge lines
    merge_lines = [l.strip() for l in merges_txt[1:] if l.strip()]

    n_vocab = max(vocab.values()) + 1
    id_to_str = [""] * n_vocab
    for s, idx in vocab.items():
        id_to_str[idx] = s

    merges = []
    for line in merge_lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        a, b = parts
        ab = a + b
        if a in vocab and b in vocab and ab in vocab:
            merges.append((vocab[a], vocab[b], vocab[ab]))

    buf = bytearray()
    buf += struct.pack("<4sII", b"CTOK", n_vocab, len(merges))
    for s in id_to_str:
        b = s.encode("utf-8")
        buf += struct.pack("<H", len(b)) + b
    for a, b, ab in merges:
        buf += struct.pack("<III", a, b, ab)

    OUT.write_bytes(buf)
    print(f"wrote {OUT}: {n_vocab} vocab, {len(merges)} merges ({len(buf) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
