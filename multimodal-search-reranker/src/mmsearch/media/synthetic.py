"""Deterministic synthetic image-retrieval dataset (BEIR-style layout).

Why synthetic: a retrieval benchmark that anyone can regenerate byte-for-byte,
offline, with no licensing questions and with unambiguous relevance labels.
It is a controlled test of cross-modal retrieval, not a substitute for real photos.
Any real dataset in the same layout can be evaluated with the same commands.

    <out>/images/<color>_<shape>_<k>.png
    <out>/captions.jsonl     partial captions, as real users write them
    <out>/queries.jsonl
    <out>/qrels/test.tsv     grade 2 = exact (color, shape); grade 1 = matches one attribute

Captions are deliberately incomplete: a third mention only the colour, a third only
the shape, a third have none. Caption fusion therefore has partial information to
add, never the full answer, which keeps the A/B comparison honest.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {
    "red": (215, 35, 35), "green": (35, 155, 65), "blue": (35, 75, 205),
    "yellow": (238, 198, 20), "purple": (128, 52, 168), "orange": (242, 128, 22),
}
SHAPES = ("circle", "square", "triangle", "star")
BACKGROUNDS = ((255, 255, 255), (236, 236, 236), (247, 242, 230))


def _polygon(shape: str, cx: float, cy: float, r: float, angle: float) -> list[tuple]:
    if shape == "triangle":
        pts = [(math.cos(angle + k * 2 * math.pi / 3), math.sin(angle + k * 2 * math.pi / 3))
               for k in range(3)]
    elif shape == "star":
        pts = [((1.0 if k % 2 == 0 else 0.45) * math.cos(angle + k * math.pi / 5),
                (1.0 if k % 2 == 0 else 0.45) * math.sin(angle + k * math.pi / 5))
               for k in range(10)]
    else:
        raise ValueError(shape)
    return [(cx + r * x, cy + r * y) for x, y in pts]


def draw(color: str, shape: str, rng: random.Random, size: int = 224) -> Image.Image:
    img = Image.new("RGB", (size, size), rng.choice(BACKGROUNDS))
    d = ImageDraw.Draw(img)
    r = rng.uniform(0.22, 0.36) * size
    cx = rng.uniform(r + 4, size - r - 4)
    cy = rng.uniform(r + 4, size - r - 4)
    fill = COLORS[color]
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "square":
        d.rectangle([cx - r * 0.85, cy - r * 0.85, cx + r * 0.85, cy + r * 0.85], fill=fill)
    else:
        d.polygon(_polygon(shape, cx, cy, r, rng.uniform(0, 2 * math.pi)), fill=fill)
    return img


def caption_for(color: str, shape: str, index: int) -> str | None:
    return (f"{color} drawing", f"a simple {shape} shape", None)[index % 3]


def generate(out_dir: str | Path, seed: int = 7, instances: int = 4, size: int = 224) -> Path:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "qrels").mkdir(exist_ok=True)
    rng = random.Random(seed)
    items: list[tuple[str, str, str]] = []
    captions = []
    n = 0
    for color in COLORS:
        for shape in SHAPES:
            for k in range(instances):
                stem = f"{color}_{shape}_{k}"
                draw(color, shape, rng, size).save(out / "images" / f"{stem}.png",
                                                   optimize=False)
                cap = caption_for(color, shape, n)
                captions.append({"_id": stem, "caption": cap})
                items.append((stem, color, shape))
                n += 1

    queries, qrels = [], []
    for color in COLORS:
        for shape in SHAPES:
            qid = f"{color}-{shape}"
            queries.append({"_id": qid, "text": f"a {color} {shape}"})
            for stem, c, s in items:
                if c == color and s == shape:
                    qrels.append((qid, stem, 2))
    for color in COLORS:
        qid = f"{color}-any"
        queries.append({"_id": qid, "text": f"something {color}"})
        qrels += [(qid, stem, 1) for stem, c, _ in items if c == color]
    for shape in SHAPES:
        qid = f"any-{shape}"
        queries.append({"_id": qid, "text": f"a {shape}"})
        qrels += [(qid, stem, 1) for stem, _, s in items if s == shape]

    with (out / "captions.jsonl").open("w") as fh:
        fh.writelines(json.dumps(c) + "\n" for c in captions)
    with (out / "queries.jsonl").open("w") as fh:
        fh.writelines(json.dumps(q) + "\n" for q in queries)
    with (out / "qrels" / "test.tsv").open("w") as fh:
        fh.write("query-id\tcorpus-id\tscore\n")
        fh.writelines(f"{q}\t{d}\t{g}\n" for q, d, g in qrels)
    (out / "README.md").write_text(
        f"Synthetic shapes dataset: seed={seed}, instances={instances}, size={size}.\n"
        "Regenerate with `python -m mmsearch.cli make-image-dataset`.\n")
    return out
