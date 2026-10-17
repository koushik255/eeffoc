"""Shared pieces: the Shot result, train/test variant pools, fonts, encoding."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Sequence, TypeVar

from PIL import Image, ImageFont

Box = tuple[int, int, int, int]  # left, top, right, bottom in page pixels
T = TypeVar("T")
PROFILES = ("train", "test")


@dataclass
class Shot:
    """A fake query and where it came from.

    boxes: (page number, box) for each page that shows up in the image: page 0
    is the page passed in, page 1 its neighbour (two-page views). A page that
    only peeks in at the edge (a reader scrolled between pages) isn't listed.
    """
    image: Image.Image
    boxes: list[tuple[int, Box]]
    kind: str
    steps: list[str] = field(default_factory=list)  # what was done, for debugging


def pool(items: Sequence[T], profile: str) -> list[T]:
    """Every third variant is held out for the test profile, so a model
    trained on "train" shots is tested on layouts, fonts and captions it
    never saw."""
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    if profile == "test":
        return list(items[2::3])
    return [x for i, x in enumerate(items) if i % 3 != 2]


def pick(rng: random.Random, items: Sequence[T], profile: str) -> T:
    return rng.choice(pool(items, profile))


# ------------------------------------------------------------------ fonts

FONT_DIRS = [Path("/usr/share/fonts"), Path.home() / ".local/share/fonts", Path.home() / ".fonts"]
# Latin fonts for captions, UI and English speech bubbles.
LATIN_FONTS = [
    "DejaVuSans-Bold.ttf", "NimbusSans-Bold.otf", "URWGothic-Demi.otf",
    "DejaVuSans.ttf", "NimbusSansNarrow-Bold.otf", "DejaVuSerif-Bold.ttf",
    "DejaVuSansCondensed-Bold.ttf", "NimbusSans-Regular.otf", "C059-Bold.otf",
    "NimbusSans-BoldItalic.otf", "URWBookman-Demi.otf", "NimbusRoman-Bold.otf",
]
CJK_FONTS = ["DroidSansFallbackFull.ttf", "NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"]


@cache
def _font_paths() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in FONT_DIRS:
        if root.is_dir():
            for path in root.rglob("*"):
                if path.suffix.lower() in {".ttf", ".otf", ".ttc"}:
                    found.setdefault(path.name, path)
    return found


def font(size: int, rng: random.Random | None = None, profile: str = "train",
         cjk: bool = False) -> ImageFont.ImageFont:
    """A font of this pixel size; with rng, a random one from the profile's pool.
    Falls back to PIL's built-in font when none of the files are installed."""
    size = max(8, int(size))
    names = CJK_FONTS if cjk else LATIN_FONTS
    available = [n for n in names if n in _font_paths()]
    if available:
        choices = (pool(available, profile) or available) if not cjk else available
        name = rng.choice(choices) if rng else choices[0]
        return ImageFont.truetype(str(_font_paths()[name]), size)
    return ImageFont.load_default(size=size)


def has_cjk_font() -> bool:
    return any(n in _font_paths() for n in CJK_FONTS)


# ------------------------------------------------------------------ images

def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def webp(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="WEBP", quality=int(quality))
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def fit(image: Image.Image, long_side: int) -> Image.Image:
    scale = long_side / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def fit_width(image: Image.Image, width: int) -> Image.Image:
    return image.resize((width, max(1, round(image.height * width / image.width))), Image.LANCZOS)


def fit_inside(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
