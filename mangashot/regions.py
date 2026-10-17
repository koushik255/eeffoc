"""Which part of a page a query shows: whole page, crops, panels, strips."""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image

from mangashot.core import Box


def full(page: Image.Image) -> Box:
    return 0, 0, page.width, page.height


def crop(page: Image.Image, rng: random.Random, w: tuple[float, float], h: tuple[float, float]) -> Box:
    """A random box covering w x h of the page (fractions of its width and height)."""
    cw = max(8, int(page.width * rng.uniform(*w)))
    ch = max(8, int(page.height * rng.uniform(*h)))
    x, y = rng.randint(0, page.width - cw), rng.randint(0, page.height - ch)
    return x, y, x + cw, y + ch


def wide(page, rng):
    return crop(page, rng, (0.7, 0.9), (0.7, 0.9))


def medium(page, rng):
    return crop(page, rng, (0.4, 0.65), (0.4, 0.65))


def tight(page, rng):
    return crop(page, rng, (0.2, 0.35), (0.2, 0.35))


def strip(page: Image.Image, rng: random.Random) -> Box:
    """A tier of panels (wide and short) or a column (tall and narrow)."""
    if rng.random() < 0.6:
        return crop(page, rng, (0.85, 1.0), (0.18, 0.35))
    return crop(page, rng, (0.2, 0.35), (0.55, 0.9))


def panels(page: Image.Image) -> list[Box]:
    """Rough panel boxes: blobs of ink separated by white gutters.

    Not a real panel detector. It finds bordered panels on white pages well
    enough for panel-shaped queries; full-bleed pages give nothing."""
    scale = 800 / max(page.size)
    grey = np.asarray(page.convert("L").resize((round(page.width * scale), round(page.height * scale))))
    ink = cv2.morphologyEx((grey < 225).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    area = grey.shape[0] * grey.shape[1]
    boxes = []
    for x, y, w, h, _ in stats[1:count]:
        if 0.03 * area <= w * h <= 0.6 * area and w > 0.12 * grey.shape[1] and h > 0.08 * grey.shape[0]:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    inside = lambda a, b: a != b and a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]
    boxes = [a for a in boxes if not any(inside(a, b) for b in boxes)]
    return [tuple(round(v / scale) for v in box) for box in boxes]


def panel(page: Image.Image, rng: random.Random, found: list[Box] | None = None) -> Box | None:
    """One detected panel with its edges nudged a few pixels (a hand crop), or None."""
    found = panels(page) if found is None else found
    if not found:
        return None
    x0, y0, x1, y1 = rng.choice(found)
    nudge = lambda: rng.randint(-8, 8)
    return (max(0, x0 + nudge()), max(0, y0 + nudge()),
            min(page.width, x1 + nudge()), min(page.height, y1 + nudge()))


REGIONS = {"full": lambda p, r: full(p), "wide": wide, "medium": medium, "tight": tight, "strip": strip}
