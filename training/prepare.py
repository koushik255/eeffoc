"""Shrink training pages and pack them for Modal.

Every page of the training series becomes a grey JPEG at most LONG_SIDE
pixels on its long side, in .local/train-pages/<series>/<volume>/NNNN.jpg,
with pages.json listing them in reading order and which are held out for
validation. Then it's packed into one tar for the Modal volume.

Never trained on: all of Steel Ball Run (the test index) and Berserk v10 and
v26 (the v1 benchmark).

    .venv/bin/python -m training.prepare
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANGA = Path.home() / "MANGA"
OUT = ROOT / ".local/train-pages"
TAR = ROOT / ".local/train-pages.tar"
LONG_SIDE = 1024
QUALITY = 90
SERIES = {
    "berserk": "Berserk/Berserk (Digital) (danke-Empire)",
    "one-piece": "One Piece/One Piece (Digital) (1r0n)",
    "vagabond": "Vagabond/[Pajeet] Vagabond",
}
EXCLUDE = {("berserk", "v10"), ("berserk", "v26")}
VALIDATION = {("vagabond", "v36"), ("vagabond", "v37")}  # held out to watch training
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SKIP = re.compile(r"credit", re.IGNORECASE)


def volume_key(archive: Path) -> str:
    # "v026", "v26", "Volume 26" -> v26; loose chapters ("One Piece 1140") -> c1140.
    name = archive.name
    if match := re.search(r"(?:Volume |\bv)(\d+)", name):
        return f"v{int(match.group(1)):02d}"
    if match := re.search(r"(\d{3,4})", name):
        return f"c{int(match.group(1))}"
    raise ValueError(f"no volume number in {name}")


def shrink(job: tuple[str, str]) -> list[dict]:
    series, archive = job
    volume = volume_key(Path(archive))
    out = OUT / series / volume
    out.mkdir(parents=True, exist_ok=True)
    pages = []
    with zipfile.ZipFile(archive) as cbz:
        members = sorted(m for m in cbz.namelist()
                         if Path(m).suffix.lower() in IMAGE_SUFFIXES and not SKIP.search(m))
        for number, member in enumerate(members):
            path = out / f"{number:04d}.jpg"
            if not path.exists():
                with Image.open(io.BytesIO(cbz.read(member))) as image:
                    image = image.convert("L")
                    scale = LONG_SIDE / max(image.size)
                    if scale < 1:
                        image = image.resize((round(image.width * scale), round(image.height * scale)),
                                             Image.LANCZOS)
                    image.save(path, quality=QUALITY)
            pages.append({"series": series, "volume": volume, "path": str(path.relative_to(OUT))})
    return pages


def main() -> None:
    jobs = []
    for series, folder in SERIES.items():
        for archive in sorted((MANGA / folder).glob("*.cbz")):
            if (series, volume_key(archive)) not in EXCLUDE:
                jobs.append((series, str(archive)))
    print(f"{len(jobs)} archives", flush=True)
    pages = []
    with ProcessPoolExecutor(8) as pool:
        for i, result in enumerate(pool.map(shrink, jobs)):
            pages += result
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} archives, {len(pages)} pages", flush=True)
    for page in pages:
        page["split"] = "validation" if (page["series"], page["volume"]) in VALIDATION else "train"
    (OUT / "pages.json").write_text(json.dumps({"long_side": LONG_SIDE, "pages": pages}, indent=1))
    counts = {s: sum(p["series"] == s for p in pages) for s in SERIES}
    print(f"{len(pages)} pages {counts}, {sum(p['split'] == 'validation' for p in pages)} held out", flush=True)
    with tarfile.open(TAR, "w") as tar:
        tar.add(OUT, arcname="train-pages")
    print(f"packed {TAR} ({TAR.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
