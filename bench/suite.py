"""Benchmark query suite: fixed query images made with mangashot.

Every kind in mangashot.KINDS (one change at a time: a crop, an edit, a
scene around the page), made with the held-out "test" profile, spread evenly
over the set's volumes. `make` writes the images, a manifest and a contact
sheet once; bench.run evaluates systems on exactly those images.

    .venv/bin/python -m bench.suite --set sbr --name sbr-v2 --per-type 25

(.local/suite/v1 was made by an earlier version of this file, before
mangashot, from the v1 page set.)
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
from PIL import Image, ImageDraw

import mangashot
from bench import catalog
from bench.catalog import LOCAL, PAGES
from mangashot.core import fit, font

SUITES = LOCAL / "suite"
BLANK_MEAN = 245  # pages brighter than this on average are near-empty (title pages)


def load(page: dict) -> Image.Image:
    with Image.open(PAGES / page["path"]) as image:
        return image.convert("RGB")


def make(set_name: str, name: str, per_type: int, seed: int) -> None:
    pages = catalog.load(set_name)
    volumes = list(dict.fromkeys((p["series"], p["volume"]) for p in pages))
    by_volume = {v: [p for p in pages if (p["series"], p["volume"]) == v] for v in volumes}
    usable = {}  # not near-empty, so a crop has something in it
    for v, members in by_volume.items():
        usable[v] = []
        for p in members:
            with Image.open(PAGES / p["path"]) as image:
                if np.asarray(image.convert("L").resize((64, 96))).mean() < BLANK_MEAN:
                    usable[v].append(p)
    suite = SUITES / name
    suite.mkdir(parents=True, exist_ok=True)
    manifest = []
    for kind in mangashot.KINDS:
        rng = random.Random(f"{seed}-{kind}")  # per kind: adding a kind leaves the others unchanged
        (suite / kind).mkdir(exist_ok=True)
        used: set[int] = set()
        n = attempts = 0
        while n < per_type:
            attempts += 1
            if attempts > per_type * 200:
                raise RuntimeError(f"not enough pages for {kind}")
            volume = volumes[(n + attempts) % len(volumes)]  # spread over the volumes
            choices = usable[volume]
            if kind == "greyscale":  # turning a grey page grey changes nothing
                choices = [p for p in choices if p["colour"]]
            if not choices:
                continue
            page = rng.choice(choices)
            if page["id"] in used:
                continue
            members = by_volume[volume]
            i = members.index(page)
            neighbour = members[i + 1] if i + 1 < len(members) else None
            shot = mangashot.make(kind, load(page), rng, "test", neighbour=load(neighbour) if neighbour else None)
            if shot is None:
                continue
            ids = [page["id"], neighbour["id"] if neighbour else None]
            used.add(page["id"])
            path = suite / kind / f"{n:02d}.jpg"
            shot.image.save(path, quality=95)
            manifest.append({"type": kind, "image": str(path.relative_to(suite)),
                             "targets": [ids[k] for k, _ in shot.boxes],
                             "boxes": [list(b) for _, b in shot.boxes], "size": list(shot.image.size)})
            n += 1
        print(f"{kind}: {n}", flush=True)
    (suite / "manifest.json").write_text(json.dumps({"set": set_name, "seed": seed, "per_type": per_type,
                                                     "profile": "test", "queries": manifest}, indent=1))
    contact_sheet(suite, manifest)
    print(f"wrote {len(manifest)} queries ({per_type} x {len(mangashot.KINDS)} kinds) to {suite}")


def contact_sheet(suite, manifest: list[dict], per_row: int = 8, height: int = 220) -> None:
    # One row per kind, its first few queries, to eyeball the suite.
    kinds = list(dict.fromkeys(q["type"] for q in manifest))
    rows = []
    for kind in kinds:
        thumbs = [Image.open(suite / q["image"]) for q in manifest if q["type"] == kind][:per_row]
        thumbs = [t.resize((max(1, round(t.width * height / t.height)), height)) for t in thumbs]
        thumbs = [t if t.width <= 2 * height else t.resize((2 * height, round(t.height * 2 * height / t.width)))
                  for t in thumbs]
        row = Image.new("RGB", (240 + sum(t.width + 8 for t in thumbs), height + 8), (250, 250, 250))
        ImageDraw.Draw(row).text((10, height / 2), kind, fill="black", font=font(22), anchor="lm")
        x = 240
        for t in thumbs:
            row.paste(t, (x, 4))
            x += t.width + 8
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), (250, 250, 250))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    fit(sheet, 4000).save(suite / "contact-sheet.jpg", quality=85)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="sbr", choices=list(catalog.SETS))
    parser.add_argument("--name", default="sbr-v2")
    parser.add_argument("--per-type", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    make(args.set, args.name, args.per_type, args.seed)


if __name__ == "__main__":
    main()
