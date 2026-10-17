"""mangashot: fake the ways people screenshot manga.

One library for both sides of the model:

  make(kind, page, rng)      one named kind of query, one factor at a time
                             (the benchmark suite is built from these)
  random_shot(page, rng)     a random mix of region, edits, scene and
                             compression (training positives)

Both return a Shot: the image plus the page boxes it shows. Layouts, fonts,
captions, colours and backgrounds come in variant pools, and every third
variant is only used with profile="test", so a model trained on "train"
shots is tested on screenshot styles it hasn't seen.

    import random
    from PIL import Image
    import mangashot

    page = Image.open("page.jpg")
    shot = mangashot.make("social_post", page, random.Random(1), profile="test")
    shot.image.save("query.jpg"); print(shot.boxes)
"""

from __future__ import annotations

import random
from typing import Callable

from PIL import Image

from mangashot import edits, regions, scenes
from mangashot.core import Box, Shot, fit, jpeg, pool

__all__ = ["KINDS", "Shot", "make", "random_shot", "pool"]


def _final(image: Image.Image, rng: random.Random, quality: tuple[int, int] = (80, 92)) -> Image.Image:
    # Screenshots are often PNG (lossless); otherwise JPEG at a typical quality.
    return image.convert("RGB") if rng.random() < 0.25 else jpeg(image, rng.randint(*quality))


def _page_or_medium(page: Image.Image, rng: random.Random) -> Box:
    return regions.medium(page, rng) if rng.random() < 0.5 else regions.full(page)


def _region_kind(region: Callable) -> Callable:
    def build(page, rng, profile, neighbour, panels):
        box = region(page, rng)
        return jpeg(page.crop(box), 85), [(0, box)]
    return build


def _edit_kind(edit: Callable, region: Callable = regions.medium, after: Callable | None = None) -> Callable:
    def build(page, rng, profile, neighbour, panels):
        box = region(page, rng)
        image = edit(page.crop(box), rng, profile)
        return (after(image, rng) if after else jpeg(image, 85)), [(0, box)]
    return build


def _scene_kind(scene: Callable, region: Callable = _page_or_medium) -> Callable:
    def build(page, rng, profile, neighbour, panels):
        box = region(page, rng)
        return _final(scene(page.crop(box), rng, profile, neighbour), rng), [(0, box)]
    return build


def _full_page(page, rng, profile, neighbour, panels):
    return jpeg(fit(page, rng.randint(1000, 1400)), rng.randint(70, 90)), [(0, regions.full(page))]


def _panel(page, rng, profile, neighbour, panels):
    box = regions.panel(page, rng, panels)
    return (jpeg(page.crop(box), 85), [(0, box)]) if box else None


def _heavy(page, rng, profile, neighbour, panels):
    box = _page_or_medium(page, rng)
    return edits.heavy_compression(fit(page.crop(box), 1000), rng, profile), [(0, box)]


def _two_page(page, rng, profile, neighbour, panels):
    if neighbour is None:
        return None
    image = scenes.two_pages(page, neighbour, rng, profile)
    return jpeg(fit(image, rng.randint(1200, 1800)), 85), [(0, regions.full(page)), (1, regions.full(neighbour))]


# Benchmark kinds: each changes one thing. f(page, rng, profile, neighbour, panels) -> (image, boxes) or None.
KINDS: dict[str, Callable] = {
    # what part of the page
    "full_page": _full_page,
    "wide_crop": _region_kind(regions.wide),
    "medium_crop": _region_kind(regions.medium),
    "tight_crop": _region_kind(regions.tight),
    "panel": _panel,
    "strip": _region_kind(regions.strip),
    # image quality
    "low_res": _edit_kind(edits.low_res, _page_or_medium),
    "heavy_compression": _heavy,
    "upscaled": _edit_kind(edits.upscale),
    "greyscale": _edit_kind(edits.greyscale, _page_or_medium),
    "colour_shift": _edit_kind(edits.colour_shift),
    # edits people make
    "speech_bubbles": _edit_kind(edits.bubbles),
    "watermark": _edit_kind(edits.watermark),
    "rotated": _edit_kind(edits.rotate),
    "mirrored": _edit_kind(edits.mirror),
    "stretched": _edit_kind(edits.squash),
    "border": _edit_kind(edits.border),
    # where it's shown
    "scroll_reader": _scene_kind(scenes.scroll_reader),
    "paged_reader": _scene_kind(scenes.paged_reader, regions.REGIONS["full"]),
    "social_post": _scene_kind(scenes.social_post),
    "chat": _scene_kind(scenes.chat),
    "browser": _scene_kind(scenes.browser),
    "photo": _scene_kind(scenes.photo),
    "two_page_view": _two_page,
}


def make(kind: str, page: Image.Image, rng: random.Random, profile: str = "test",
         neighbour: Image.Image | None = None, panels: list[Box] | None = None) -> Shot | None:
    """One query of a named kind, or None when the page can't make it (no
    panel found, no neighbour for a two-page view)."""
    built = KINDS[kind](page.convert("RGB"), rng, profile, neighbour.convert("RGB") if neighbour else None, panels)
    if built is None:
        return None
    image, boxes = built
    return Shot(image, boxes, kind, [kind])


# Training mix: probabilities of each step.
REGION_MIX = {"full": 0.2, "wide": 0.15, "medium": 0.2, "tight": 0.15, "panel": 0.15, "strip": 0.15}
EDIT_MIX = {"bubbles": 0.25, "watermark": 0.1, "mirror": 0.15, "rotate": 0.15, "squash": 0.1, "border": 0.1,
            "colour_shift": 0.2}
SCENE_MIX = {None: 0.45, "scroll_reader": 0.1, "paged_reader": 0.06, "social_post": 0.1, "chat": 0.08,
             "browser": 0.07, "photo": 0.12, "two_pages": 0.02}
QUALITY_MIX = {"low_res": 0.1, "upscale": 0.08, "blur_noise": 0.1, "heavy_compression": 0.1}


def random_shot(page: Image.Image, rng: random.Random, profile: str = "train",
                neighbour: Image.Image | None = None, panels: list[Box] | None = None) -> Shot:
    """A random query: a region of the page, some edits, maybe a scene around
    it, then compression. Several things at once, unlike make()."""
    page = page.convert("RGB")
    steps: list[str] = []
    scene = rng.choices(list(SCENE_MIX), weights=list(SCENE_MIX.values()))[0]
    if scene == "two_pages" and neighbour is not None:
        image = scenes.two_pages(page, neighbour.convert("RGB"), rng, profile)
        boxes = [(0, regions.full(page)), (1, regions.full(neighbour))]
        steps.append("two_pages")
    else:
        if scene == "two_pages":
            scene = None
        name = rng.choices(list(REGION_MIX), weights=list(REGION_MIX.values()))[0]
        box = regions.panel(page, rng, panels) if name == "panel" else regions.REGIONS[name](page, rng)
        if box is None:
            name, box = "tight", regions.tight(page, rng)
        steps.append(name)
        image = page.crop(box)
        for edit, p in EDIT_MIX.items():
            if rng.random() < p:
                image = edits.EDITS[edit](image, rng, profile)
                steps.append(edit)
        if scene:
            image = scenes.SCENES[scene](image, rng, profile, neighbour.convert("RGB") if neighbour else None)
            steps.append(scene)
        boxes = [(0, box)]
    for edit, p in QUALITY_MIX.items():
        if rng.random() < p:
            image = edits.EDITS[edit](image, rng, profile)
            steps.append(edit)
    image = _final(image, rng, (60, 95))
    return Shot(image, boxes, "random", steps)
