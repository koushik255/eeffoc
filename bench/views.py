"""Query-side views: what of a query image gets embedded and searched.

curtain searches with the query and its four corner pieces. Two cases break
that, so queries can also be searched as:

  content   a screenshot (chat, social post, browser, photo) where the manga
            is a small part of the image: find the busy rectangle among the
            flat app colours and search with that crop as well
  windows   a strip (a tier of panels) squashed to a square is unrecognisable:
            search with overlapping squarish windows along it instead of
            corner pieces
  halves    a landscape image (two facing pages, a spread) is also searched
            as its left and right halves

Each group of views is scored the curtain way and a page keeps its best
group score.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

WHOLE_HANDICAP = 0.1   # curtain's: a whole-query match counts 0.1 less than the piece mean
WORK_SIDE = 512        # content detection runs at most this large
BUSY_STD = 14          # local grey-level spread above this: texture (art), below: flat UI
CONTENT_AREA = (0.06, 0.85)  # a found rectangle outside this share of the image isn't used
STRIP_ASPECT = 2.2     # wider (or taller) than this: search with windows along it
WINDOW_ASPECT = 1.4    # windows are this many short sides long
HALVES_ASPECT = (1.2, STRIP_ASPECT)  # landscape images in this width/height range get halves


def pieces(image: Image.Image) -> list[Image.Image]:
    """curtain's four overlapping corner pieces, each 60% of the width and height."""
    w, h = image.size
    pw, ph = int(w * 0.6), int(h * 0.6)
    return [image.crop((x, y, x + pw, y + ph)) for y in (0, h - ph) for x in (0, w - pw)]


def content_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    """The biggest textured rectangle in a screenshot, or None when the
    texture fills (or barely fills) the image: then it's the content already."""
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    scale = min(1.0, WORK_SIDE / max(grey.shape))
    small = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else grey
    mean = cv2.blur(small, (9, 9))
    spread = np.sqrt(np.maximum(cv2.blur(small * small, (9, 9)) - mean * mean, 0))
    busy = (spread > BUSY_STD).astype(np.uint8)
    busy = cv2.morphologyEx(busy, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(busy, connectivity=8)
    if count < 2:
        return None
    x, y, w, h, _ = max(stats[1:], key=lambda s: s[2] * s[3])
    share = w * h / (small.shape[0] * small.shape[1])
    if not CONTENT_AREA[0] <= share <= CONTENT_AREA[1]:
        return None
    return (int(x / scale), int(y / scale), int(min(grey.shape[1], (x + w) / scale)),
            int(min(grey.shape[0], (y + h) / scale)))


def windows(image: Image.Image) -> list[Image.Image] | None:
    """Overlapping windows along a strip, half a window apart, or None if it isn't one."""
    w, h = image.size
    long_side, short = max(w, h), min(w, h)
    if long_side < STRIP_ASPECT * short:
        return None
    size = int(short * WINDOW_ASPECT)
    count = max(2, int(np.ceil((long_side - size) / (size / 2))) + 1)
    starts = np.linspace(0, long_side - size, count).round().astype(int)
    if w >= h:
        return [image.crop((s, 0, s + size, h)) for s in starts]
    return [image.crop((0, s, w, s + size)) for s in starts]


def groups(image: Image.Image) -> list[list[Image.Image]]:
    """View groups for one query, each with its whole image first."""
    def group(view: Image.Image) -> list[Image.Image]:
        return [view, *(windows(view) or pieces(view))]
    box = content_box(image)
    out = []
    for view in [image] + ([image.crop(box)] if box else []):
        out.append(group(view))
        w, h = view.size
        if HALVES_ASPECT[0] <= w / h < HALVES_ASPECT[1]:
            out += [group(view.crop((0, 0, w // 2, h))), group(view.crop((w // 2, 0, w, h)))]
    return out


def page_scores(vectors: torch.Tensor, queries: np.ndarray) -> np.ndarray:
    """curtain's score per page for one group. vectors: [pages, tiles, d]
    float32, on the CPU or GPU; queries: [views, d], whole view first: the
    better of the mean over views and the whole view minus WHOLE_HANDICAP.
    One matrix multiply over all tiles, on the vectors' device (on the CPU,
    torch's is ~4x faster than numpy's einsum here)."""
    pages, tiles, dim = vectors.shape
    queries = torch.from_numpy(np.asarray(queries, dtype=np.float32)).to(vectors.device)
    best = (vectors.view(pages * tiles, dim) @ queries.T).view(pages, tiles, -1).amax(dim=1)
    return torch.maximum(best[:, 0] - WHOLE_HANDICAP, best.mean(dim=1)).cpu().numpy()


def rank(vectors: torch.Tensor, embedded: list[np.ndarray], top: int) -> list[int]:
    """Pages by their best group score."""
    score = np.max([page_scores(vectors, q) for q in embedded], axis=0)
    return np.argsort(-score)[:top].tolist()
