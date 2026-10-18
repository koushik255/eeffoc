"""Find where a screenshot comes from and show it on the page.

Searches the index the best way the benchmark knows (manga-ft1, fine tiles,
extra views, SIFT check on the top 10), then shows the matched page with the
page before and after it, since screenshots often catch two pages. On every
shown page that SIFT confirms, the part of the screenshot it matched is
outlined: the box around the screenshot points whose matches agree on one
affine map, carried onto the page by that map and clipped to the page.

    .venv/bin/python -m bench.find shot.jpg other.png --out .local/found
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from bench import systems
from bench.catalog import LOCAL, PAGES
from mangashot.core import font

SHOW_HEIGHT = 1000
MATCH, NEIGHBOUR = (230, 30, 30), (30, 120, 230)


class Finder:
    def __init__(self, set_name: str = "sbr", model: str = "manga-ft1"):
        self.pages = systems.catalog(set_name)
        self.sift_pages = systems.SiftPages(self.pages, set_name)
        self.search = systems.system_embedding(model, set_name, self.pages, self.sift_pages, align_top=10,
                                               extra_views=True, fine=True)

    def neighbours(self, page: int) -> list[int]:
        # The page before, the match, the page after, within one volume.
        same = lambda i: 0 <= i < len(self.pages) and self.pages[i]["volume"] == self.pages[page]["volume"]
        return [i for i in (page - 1, page, page + 1) if same(i)]

    def number(self, page: int) -> int:
        # 1-based page number within its volume.
        return sum(p["volume"] == self.pages[page]["volume"] for p in self.pages[:page]) + 1

    def outline(self, feats, page: int) -> tuple[int, np.ndarray | None]:
        """(inliers, 4 corners in page pixels) of the part of the query that matched this page."""
        points, matrix, matched = self.sift_pages.align(feats, page)
        if matrix is None or points < systems.VERIFIED_POINTS:
            return points, None
        (x0, y0), (x1, y1) = matched.min(axis=0), matched.max(axis=0)
        corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        w, h = self.pages[page]["size"]
        on_page = cv2.transform(corners[None], matrix)[0] / min(1.0, systems.SIFT_PAGE_SIDE / max(w, h))
        return points, np.clip(on_page, 0, [w, h])

    def find(self, image: Image.Image) -> tuple[dict, Image.Image]:
        query = systems.upload(image)
        with torch.inference_mode():
            found = self.search(query)
        best = found[0]
        feats = systems.sift(np.asarray(query), systems.SIFT_QUERY_SIDE)
        shown = []
        for page in self.neighbours(best):
            points, corners = self.outline(feats, page)
            shown.append({"page": page, "volume": self.pages[page]["volume"], "number": self.number(page),
                          "role": "match" if page == best else ("previous" if page < best else "next"),
                          "sift_points": points, "outline": corners.round().tolist() if corners is not None else None})
        return {"best": shown[[s["role"] for s in shown].index("match")], "pages": shown,
                "top10": found[:10]}, self.render(shown)

    def render(self, shown: list[dict]) -> Image.Image:
        tiles = []
        for s in shown:
            with Image.open(PAGES / self.pages[s["page"]]["path"]) as image:
                page = image.convert("RGB")
            scale = SHOW_HEIGHT / page.height
            page = page.resize((round(page.width * scale), SHOW_HEIGHT))
            draw = ImageDraw.Draw(page)
            colour = MATCH if s["role"] == "match" else NEIGHBOUR
            if s["outline"]:  # kept 3 px inside the page so a whole-page match still shows
                inside = np.clip(np.array(s["outline"]) * scale, 3, [page.width - 4, page.height - 4])
                draw.polygon([tuple(c) for c in inside], outline=colour, width=6)
            tile = Image.new("RGB", (page.width, SHOW_HEIGHT + 50), "white")
            tile.paste(page, (0, 50))
            label = f"{s['role']}: {s['volume']} p{s['number']}" + (" (outlined)" if s["outline"] else "")
            ImageDraw.Draw(tile).text((6, 10), label, fill=colour if s["outline"] else (90, 90, 90), font=font(28))
            tiles.append(tile)
        out = Image.new("RGB", (sum(t.width + 16 for t in tiles), SHOW_HEIGHT + 50), (240, 240, 240))
        x = 0
        for t in tiles:
            out.paste(t, (x, 0))
            x += t.width + 16
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=LOCAL / "found")
    parser.add_argument("--set", default="sbr")
    parser.add_argument("--model", default="manga-ft1")
    args = parser.parse_args()
    torch.set_num_threads(16)
    finder = Finder(args.set, args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        with Image.open(path) as image:
            result, picture = finder.find(image.convert("RGB"))
        name = f"{path.parent.name}-{path.stem}.jpg"
        picture.save(args.out / name, quality=88)
        best = result["best"]
        print(json.dumps({"query": str(path), "match": f"{best['volume']} p{best['number']}",
                          "outlined": [f"{s['role']} {s['volume']} p{s['number']}" for s in result["pages"] if s["outline"]],
                          "saved": str(args.out / name)}))


if __name__ == "__main__":
    main()
