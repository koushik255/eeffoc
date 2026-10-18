"""Extract manga volumes into a page catalog.

Pages are copied byte for byte from the .cbz archives into
.local/pages/<series>/<volume>/NNNN.<ext>, and .local/pages/catalog-<set>.json
lists a set's pages in order: id, series, volume, archive member, size,
colour or grey. Pages that aren't the manga are left out of the catalog:
files named as credits, and the repeated credit, blank and colophon pages
listed in bench/<set>-skip.json (a crop of one would match all its copies).

  v1    two volumes each of Steel Ball Run and Berserk (the first benchmark)
  sbr   all of Steel Ball Run: the test index. Never train on it.

    .venv/bin/python -m bench.catalog --set sbr
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
PAGES = LOCAL / "pages"
MANGA = Path.home() / "MANGA"
SBR = "JoJo Part 7 - Steel Ball Run/Jojo Part 7-) Steel Ball Run"
BERSERK = "Berserk/Berserk (Digital) (danke-Empire)"
SETS = {
    "v1": {
        "steel-ball-run": [f"{SBR}/Volume 01 - September 5th, 1890 San Diego Beach.cbz",
                           f"{SBR}/Volume 12 - Qualifications for the Corpse, Qualifications for Friendship.cbz"],
        "berserk": [f"{BERSERK}/Berserk v10 (2006) (Digital) (danke-Empire).cbz",
                    f"{BERSERK}/Berserk v26 (2008) (Digital) (danke-Empire).cbz"],
    },
    "sbr": {"steel-ball-run": sorted(str(p.relative_to(MANGA)) for p in (MANGA / SBR).glob("*.cbz"))},
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SKIP = re.compile(r"credit", re.IGNORECASE)
COLOUR_CHROMA = 8.0  # mean max-min channel spread above this: a colour page


def chroma(image: Image.Image) -> float:
    # Mean channel spread of a small copy: ~0 for grey pages stored as RGB.
    small = np.asarray(image.convert("RGB").resize((128, 192)), dtype=np.int16)
    return float((small.max(axis=2) - small.min(axis=2)).mean())


def volume_key(archive: str) -> str:
    match = re.search(r"(?:Volume |v)(\d+)", Path(archive).name)
    return f"v{int(match.group(1)):02d}"


def catalog_path(name: str) -> Path:
    return PAGES / f"catalog-{name}.json"


def load(name: str) -> list[dict]:
    return json.loads(catalog_path(name).read_text())["pages"]


def skipped(name: str) -> set[str]:
    path = ROOT / "bench" / f"{name}-skip.json"
    return set(json.loads(path.read_text())["members"]) if path.exists() else set()


def build(name: str) -> None:
    skip = skipped(name)
    pages = []
    for series, archives in SETS[name].items():
        for archive in archives:
            volume = volume_key(archive)
            out = PAGES / series / volume
            out.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(MANGA / archive) as cbz:
                members = sorted(m for m in cbz.namelist()
                                 if Path(m).suffix.lower() in IMAGE_SUFFIXES and not SKIP.search(m))
                for number, member in enumerate(members):  # numbered before skipping: file names stay put
                    if member in skip:
                        continue
                    data = cbz.read(member)
                    path = out / f"{number:04d}{Path(member).suffix.lower()}"
                    if not path.exists():
                        path.write_bytes(data)
                    with Image.open(io.BytesIO(data)) as image:
                        size, colour = image.size, chroma(image) > COLOUR_CHROMA
                    pages.append({"id": len(pages), "series": series, "volume": volume,
                                  "path": str(path.relative_to(PAGES)), "member": member,
                                  "size": list(size), "colour": colour})
            print(f"{series} {volume}: {sum(m not in skip for m in members)} pages", flush=True)
    catalog_path(name).write_text(json.dumps({"set": name, "volumes": SETS[name], "pages": pages}, indent=1))
    colour = sum(p["colour"] for p in pages)
    print(f"{len(pages)} pages ({colour} colour) -> {catalog_path(name)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="sbr", choices=list(SETS))
    build(parser.parse_args().set)


if __name__ == "__main__":
    main()
