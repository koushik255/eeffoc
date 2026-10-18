"""Run search systems on the query suite and write a report.

    .venv/bin/python -m bench.run --suite sbr-v2
    .venv/bin/python -m bench.run --suite sbr-v2 --systems curtain <run-name> <run-name>+align

A system is "sift", "sscd", "curtain" or a fine-tuned run in .local/runs/<name>,
with "+fine" to index a finer tile grid as well, "+views" to also search with
content crops, strip windows and halves (bench.views) and "+align" to re-order
its top 10 by SIFT + RANSAC.

Scores per query: the right page first (exact), the right page or the page
before or after it first (neighbour: a site shows the match with the pages
around it, and a screenshot often catches two), the right page in the top 10,
the right volume first, the right series first. For two_page_view either
page counts.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

from bench import systems
from bench.catalog import LOCAL
from bench.suite import SUITES

RESULTS = LOCAL / "results"
LABELS = {"sift": "SIFT + RANSAC", "curtain": "curtain ResNet-18", "sscd": "SSCD ResNet-50"}


def label(key: str) -> str:
    name, *extras = key.split("+")
    names = {"fine": " + fine tiles", "views": " + views", "align": " + align 10"}
    return LABELS.get(name, name) + "".join(names[e] for e in extras)
METRICS = [("exact", "Exact page"), ("neighbour", "Page or its neighbour"), ("top10", "Right page in top 10"),
           ("volume", "Right volume"), ("series", "Right series")]


def build(key: str, set_name: str, pages: list[dict], sift_pages):
    if key == "sift":
        return systems.system_sift(pages, sift_pages)
    name, *extras = key.split("+")
    unknown = set(extras) - {"fine", "views", "align"}
    if unknown:
        raise ValueError(f"unknown system modifier(s) {unknown} in {key}")
    return systems.system_embedding(name, set_name, pages, sift_pages, align_top=10 if "align" in extras else 0,
                                    extra_views="views" in extras, fine="fine" in extras)


def score(found: list[int], targets: list[int], pages: list[dict]) -> dict[str, bool]:
    if not found:
        return {m: False for m, _ in METRICS}
    first, target = pages[found[0]], pages[targets[0]]
    same_volume = lambda t: (pages[t]["series"], pages[t]["volume"]) == (first["series"], first["volume"])
    return {
        "exact": found[0] in targets,
        "neighbour": any(abs(found[0] - t) <= 1 and same_volume(t) for t in targets),
        "top10": any(t in found[:10] for t in targets),
        "volume": (first["series"], first["volume"]) == (target["series"], target["volume"]),
        "series": first["series"] == target["series"],
    }


def table(report: dict, keys: list[str], metric: str, types: list[str]) -> list[str]:
    lines = ["| type | " + " | ".join(label(k) for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for kind in types + ["ALL"]:
        cells = []
        for k in keys:
            rows = [r for r in report["systems"][k]["rows"] if kind in ("ALL", r["type"])]
            cells.append(f"{sum(r[metric] for r in rows)}/{len(rows)}")
        lines.append(f"| {'**all**' if kind == 'ALL' else kind} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="sbr-v2")
    parser.add_argument("--systems", nargs="+", default=["curtain", "curtain+align"])
    args = parser.parse_args()
    torch.set_num_threads(16)

    suite = SUITES / args.suite
    manifest = json.loads((suite / "manifest.json").read_text())
    set_name = manifest.get("set", "v1")
    queries = manifest["queries"]
    images = [systems.upload(Image.open(suite / q["image"])) for q in queries]
    pages = systems.catalog(set_name)
    needs_sift = any(k == "sift" or k.endswith("+align") for k in args.systems)
    sift_pages = systems.SiftPages(pages, set_name) if needs_sift else None
    report = {"suite": args.suite, "set": set_name, "queries": len(queries), "pages": len(pages), "systems": {}}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for key in args.systems:
        started = time.perf_counter()
        print(f"{label(key)}: building", flush=True)
        run = build(key, set_name, pages, sift_pages)
        print(f"{label(key)}: ready in {time.perf_counter() - started:.0f}s", flush=True)
        rows, latencies = [], []
        for query, image in zip(queries, images):
            t = time.perf_counter()
            found = run(image)
            latencies.append(time.perf_counter() - t)
            rows.append({"type": query["type"], "image": query["image"], "targets": query["targets"],
                         "found": found, **score(found, query["targets"], pages)})
        exact = sum(r["exact"] for r in rows)
        print(f"{label(key)}: {exact}/{len(rows)} exact, median {np.median(latencies) * 1000:.0f} ms", flush=True)
        report["systems"][key] = {"label": label(key), "rows": rows,
                                  "median_ms": float(np.median(latencies) * 1000)}
        text = write(report, queries, pages, f"{args.suite}-{stamp}")  # after every system, so a stopped run keeps what it has
    print("\n" + text + f"\n\nsaved {RESULTS / args.suite}-{stamp}.json and .md")


def write(report: dict, queries: list[dict], pages: list[dict], stamp: str) -> str:
    keys = list(report["systems"])
    types = list(dict.fromkeys(q["type"] for q in queries))
    series = sorted({p["series"] for p in pages})
    lines = [f"# Benchmark {report['suite']}: {len(queries)} queries, {len(types)} types, {len(pages)} pages "
             f"({', '.join(series)})", ""]
    for metric, title in METRICS:
        lines += [f"## {title}", "", *table(report, keys, metric, types), ""]
    lines.append("Median search time: " + ", ".join(
        f"{label(k)} {report['systems'][k]['median_ms']:.0f} ms" for k in keys))
    text = "\n".join(lines)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stamp}.json").write_text(json.dumps(report, indent=1))
    (RESULTS / f"{stamp}.md").write_text(text + "\n")
    return text


if __name__ == "__main__":
    main()
