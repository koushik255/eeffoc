"""Compress the manga library to AVIF and embed it, in one pass.

For every series the plan marks "compress" (~/MANGA/.logs/compression-plan.csv),
each volume (.cbz, .cbr or .zip, read with libarchive, so RAR works too)
becomes a .cbz of AVIF pages (q55, speed 10) under ~/MANGA-avif, mirroring
~/MANGA. Pages are compressed here, 16 at once, and each one goes to a
bench.gpu worker (EEFFOC_GPU) as soon as it's done, so compressing and
embedding run at the same time: the next volume compresses while the GPU
finishes the last. The vectors are the AVIF pages' (what's kept), as fp16:

    .local/library/<model>-fine/<series>/<archive>.npy   [pages, 171, 128]
    .local/library/<model>-fine/<series>/<archive>.json  members, unreadable pages

Volumes already done are skipped, so a stopped run picks up where it was.
Without EEFFOC_GPU it only compresses (and a later run with it embeds the
AVIF volumes). Vectors belong to the model that made them: after training a
new one, embed again into a new <model>-fine folder.

    EEFFOC_GPU=http://127.0.0.1:8813 .venv/bin/python -m bench.library --series Veil "Sayonara Eri"
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing
import os
import time
import zipfile
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import libarchive
import numpy as np
from PIL import Image

from bench.catalog import COLOUR_CHROMA, LOCAL, MANGA, chroma  # light imports only: the encoders import this module

PLAN = MANGA / ".logs/compression-plan.csv"
AVIF_ROOT = Path.home() / "MANGA-avif"
VECTORS = LOCAL / "library"
ARCHIVES = {".cbz", ".cbr", ".zip"}
IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}
AVIF = dict(quality=55, speed=10, max_threads=1)  # as good as AVIF q50 at speed 6, 7.5x faster (bzk-v1)
# Long side these series' pages are shrunk to first: ~25 MP grainy scans, where AVIF at full size saves
# little (Akira 2.7 -> 2.4 MB a page) or grows them (Nausicaä 2.2 -> 2.4); at 3000 px, 0.7-0.8 MB.
MAX_SIDE = {"Akira": 3000, "Kaze no Tani no Nausicaä": 3000}
ENCODERS = 16


def planned(path: Path) -> list[str]:
    return [row["title"] for row in csv.DictReader(path.open()) if row["decision"] == "compress"]


def read_pages(archive: Path) -> list[tuple[str, bytes]]:
    """(member name, file) of every image in the archive, in name order."""
    with libarchive.file_reader(str(archive)) as entries:
        pages = [(e.pathname, b"".join(e.get_blocks())) for e in entries
                 if e.isfile and Path(e.pathname).suffix.lower() in IMAGES]
    return sorted(pages)


def encode(data: bytes, max_side: int | None = None) -> bytes | None:
    """The page as AVIF, at most max_side on its long side, or None if PIL can't read it."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except OSError:
        return None
    if image.mode not in ("L", "RGB", "RGBA"):  # palette or 1-bit: grey or RGB
        image = image.convert("RGB")
        if chroma(image) <= COLOUR_CHROMA:
            image = image.convert("L")
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    options = dict(AVIF)
    if image.mode == "L":
        options["subsampling"] = "4:0:0"
    out = io.BytesIO()
    image.save(out, format="AVIF", **options)
    return out.getvalue()


def compress(archive: Path, avif_path: Path, encoders: ProcessPoolExecutor, embed, max_side: int | None) -> dict:
    """Write the volume as a .cbz of AVIF pages, handing each page to embed as it's done.
    A page whose AVIF isn't smaller (big palette PNG scans: Nausicaä) is kept as it was."""
    pages = read_pages(archive)
    unfinished = avif_path.with_name(avif_path.name + ".partial")
    unfinished.parent.mkdir(parents=True, exist_ok=True)
    kept, failed, originals, size_out = [], [], 0, 0
    with zipfile.ZipFile(unfinished, "w", zipfile.ZIP_STORED) as out:
        encoded = encoders.map(partial(encode, max_side=max_side), [data for _, data in pages], chunksize=2)
        for (name, original), avif in zip(pages, encoded):
            if avif is None:
                failed.append(name)
                continue
            if len(avif) < len(original):
                member, data = str(Path(name).with_suffix(".avif")), avif
            else:
                member, data, originals = name, original, originals + 1
            out.writestr(member, data)
            kept.append(member)
            size_out += len(data)
            embed(data)
    os.replace(unfinished, avif_path)
    return {"pages": len(pages), "members": kept, "failed": failed, "originals": originals,
            "size_in": sum(len(data) for _, data in pages), "size_out": size_out}


def save_vectors(futures: list[Future], record: dict, path: Path) -> None:
    """Wait for the worker, then write the vectors, then the record (which marks the volume done)."""
    vectors = np.stack([f.result() for f in futures]).astype(np.float16) if futures else np.zeros((0,), np.float16)
    unfinished = path.with_name(path.name + ".partial")
    with unfinished.open("wb") as out:
        np.save(out, vectors)
    os.replace(unfinished, path.with_suffix(".npy"))
    path.write_text(json.dumps(record, indent=1))


def main() -> None:
    from bench import systems  # torch: here, not in the encoder processes

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--series", nargs="+", help="these series only (default: every one the plan compresses)")
    parser.add_argument("--model", default="manga-ft1")
    args = parser.parse_args()
    series_names = args.series or planned(args.plan)
    worker = None
    if systems.GPU_WORKER:
        from bench.gpu import Worker
        worker = Worker(systems.GPU_WORKER, systems.checkpoint(args.model), systems.FINE_GRIDS)
    vectors_root = VECTORS / f"{args.model}-fine"
    print(json.dumps({"series": len(series_names), "avif": str(AVIF_ROOT),
                      "vectors": str(vectors_root) if worker else None}), flush=True)

    encoders = ProcessPoolExecutor(ENCODERS, mp_context=multiprocessing.get_context("spawn"))
    finisher, finishing = ThreadPoolExecutor(1), []
    started, total = time.perf_counter(), {"pages": 0, "size_in": 0, "size_out": 0}
    for series in series_names:
        for archive in sorted(p for p in (MANGA / series).rglob("*") if p.suffix.lower() in ARCHIVES):
            relative = archive.relative_to(MANGA)
            avif_path = (AVIF_ROOT / relative).with_suffix(".cbz")
            record_path = (vectors_root / relative).with_suffix(".json")
            if avif_path.exists() and (worker is None or record_path.exists()):
                continue
            t, futures = time.perf_counter(), []
            embed = (lambda avif, futures=futures: futures.append(worker.pool.submit(worker.embed, avif))) if worker \
                else (lambda avif: None)
            if avif_path.exists():  # compressed before, without a worker: embed what's kept
                pages = read_pages(avif_path)
                for _, avif in pages:
                    embed(avif)
                done = {"pages": len(pages), "members": [n for n, _ in pages], "failed": [], "originals": 0,
                        "size_in": 0, "size_out": 0}
            else:
                done = compress(archive, avif_path, encoders, embed, MAX_SIDE.get(series))
            if worker:
                record = {"archive": str(relative), "avif": str(avif_path), "members": done["members"],
                          "unreadable": done["failed"]}
                record_path.parent.mkdir(parents=True, exist_ok=True)
                finishing.append(finisher.submit(save_vectors, futures, record, record_path))
                if len(finishing) > 1:
                    finishing[-2].result()  # fail fast, and at most one volume waits on the GPU
            for key in total:
                total[key] += done[key]
            print(f"{series} / {archive.stem}: {done['pages']} pages, {done['size_in'] / 1e6:.0f} -> "
                  f"{done['size_out'] / 1e6:.0f} MB, {done['pages'] / (time.perf_counter() - t):.1f} pages/s"
                  + (f", {done['originals']} kept as they were" if done["originals"] else "")
                  + (f", {len(done['failed'])} unreadable" if done["failed"] else ""), flush=True)
    for f in finishing:
        f.result()
    print(f"done: {total['pages']} pages in {(time.perf_counter() - started) / 60:.0f} min, "
          f"{total['size_in'] / 1e9:.1f} -> {total['size_out'] / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
