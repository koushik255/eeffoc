"""Edits to the shown region: scanlation text, watermarks, geometry, image quality.

Every edit is edit(image, rng, profile) -> image.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from mangashot.core import font, has_cjk_font, jpeg, pick, pool, webp

ENGLISH_LINES = [
    "WHAT ARE", "YOU DOING", "HERE?!", "I WON'T", "LOSE...", "THAT'S", "IMPOSSIBLE", "WAIT!",
    "IT'S NOT", "OVER YET", "STAY BACK", "DAMN IT...", "RUN!", "WHO ARE", "YOU?", "THIS IS", "MY FIGHT",
    "HOLD ON", "LOOK OUT!", "NO WAY...", "GET DOWN!", "I SEE...", "BEHIND YOU", "HA HA HA",
]
JAPANESE_LINES = ["なんだと", "まさか", "逃げろ", "待て", "行くぞ", "ふざけるな", "どうして", "終わりだ", "見ろ"]
WATERMARKS = ["SCANS", "read at mangasite", "TL: anon", "raw provider", "@scanteam", "do not repost"]


def bubbles(image: Image.Image, rng: random.Random, profile: str = "train") -> Image.Image:
    """1-3 speech bubbles pasted over the art: English or Japanese text, oval or boxed."""
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    japanese = has_cjk_font() and rng.random() < 0.3
    for _ in range(rng.randint(1, 3)):
        bw, bh = int(image.width * rng.uniform(0.18, 0.35)), int(image.height * rng.uniform(0.12, 0.25))
        x, y = rng.randint(0, image.width - bw), rng.randint(0, image.height - bh)
        outline = max(2, bw // 60)
        if rng.random() < 0.7:
            draw.ellipse((x, y, x + bw, y + bh), fill="white", outline="black", width=outline)
        else:  # narration box
            draw.rectangle((x, y, x + bw, y + bh), fill="white", outline="black", width=outline)
        if japanese:  # vertical columns, right to left
            columns = rng.sample(pool(JAPANESE_LINES, profile), rng.randint(1, 2))
            size = max(8, min(bw // 4, bh // 6))
            f = font(size, cjk=True)
            for c, text in enumerate(columns):
                cx = x + bw * (0.6 - 0.25 * c)
                for i, ch in enumerate(text[: max(1, int(bh * 0.8 // size))]):
                    draw.text((cx, y + bh * 0.15 + i * size), ch, fill="black", font=f, anchor="mt")
        else:
            text = "\n".join(rng.sample(pool(ENGLISH_LINES, profile), rng.randint(1, 3)))
            draw.multiline_text((x + bw / 2, y + bh / 2), text, fill="black", anchor="mm", align="center",
                                font=font(bh // 7, rng, profile))
    return image


def watermark(image: Image.Image, rng: random.Random, profile: str = "train") -> Image.Image:
    """A scanlator watermark: translucent text in a corner or across the middle."""
    image = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    size = int(image.width * rng.uniform(0.03, 0.08))
    text = pick(rng, WATERMARKS, profile)
    colour = rng.choice([(0, 0, 0), (255, 255, 255), (200, 30, 30)]) + (rng.randint(60, 160),)
    f = font(size, rng, profile)
    if rng.random() < 0.5:
        x = rng.choice([0.03, 0.97]) * image.width
        draw.text((x, image.height * rng.uniform(0.03, 0.97)), text, fill=colour, font=f,
                  anchor="lm" if x < image.width / 2 else "rm")
    else:
        draw.text((image.width / 2, image.height / 2), text, fill=colour, font=font(size * 2, rng, profile),
                  anchor="mm")
        layer = layer.rotate(rng.uniform(-35, 35), resample=Image.BICUBIC)
    return Image.alpha_composite(image, layer).convert("RGB")


def mirror(image, rng, profile="train"):
    return ImageOps.mirror(image)


def rotate(image: Image.Image, rng: random.Random, profile: str = "train") -> Image.Image:
    angle = rng.choice([-1, 1]) * rng.uniform(2, 8)
    fill = rng.choice([(255, 255, 255), (0, 0, 0)])
    return image.convert("RGB").rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=fill)


def squash(image: Image.Image, rng: random.Random, profile: str = "train") -> Image.Image:
    # Stretched 0.7-0.85x or 1.15-1.35x in one direction, like a careless resize.
    factor = rng.choice([rng.uniform(0.7, 0.85), rng.uniform(1.15, 1.35)])
    if rng.random() < 0.5:
        return image.resize((max(1, round(image.width * factor)), image.height), Image.BICUBIC)
    return image.resize((image.width, max(1, round(image.height * factor))), Image.BICUBIC)


def border(image: Image.Image, rng: random.Random, profile: str = "train") -> Image.Image:
    # Padding someone added before posting.
    pad = int(max(image.size) * rng.uniform(0.02, 0.12))
    return ImageOps.expand(image.convert("RGB"), pad, fill=rng.choice([(255, 255, 255), (0, 0, 0), (24, 24, 27)]))


def greyscale(image, rng, profile="train"):
    return ImageOps.grayscale(image).convert("RGB")


def colour_shift(image: Image.Image, rng: random.Random, profile: str = "train") -> Image.Image:
    """Yellowed paper, gamma, contrast and brightness: a scan or a tired screen."""
    image = ImageEnhance.Brightness(image.convert("RGB")).enhance(rng.uniform(0.8, 1.15))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.7, 1.3))
    array = (np.asarray(image, dtype=np.float32) / 255) ** rng.uniform(0.7, 1.4)
    paper = pick(rng, [(1.0, 0.93, 0.78), (0.98, 0.95, 0.88), (0.9, 0.95, 1.0), (1.0, 0.97, 0.9),
                       (0.95, 0.9, 0.8), (0.92, 0.92, 0.96)], profile)
    return Image.fromarray((array * np.array(paper, np.float32) * 255).clip(0, 255).astype(np.uint8))


def low_res(image: Image.Image, rng: random.Random, profile: str = "train") -> Image.Image:
    scale = rng.randint(200, 350) / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.BILINEAR)


def upscale(image: Image.Image, rng: random.Random, profile: str = "train") -> Image.Image:
    """Shrunk, blown up 2-4x and smoothed: what an AI upscaler or waifu2x leaves."""
    small = image.convert("RGB").resize((max(1, image.width // rng.randint(2, 3)),
                                         max(1, image.height // rng.randint(2, 3))), Image.BILINEAR)
    big = small.resize((small.width * rng.randint(2, 4), small.height * rng.randint(2, 4)), Image.BICUBIC)
    smooth = cv2.bilateralFilter(np.asarray(big), 7, 40, 7)
    return Image.fromarray(smooth).filter(ImageFilter.UnsharpMask(radius=2, percent=120))


def heavy_compression(image: Image.Image, rng: random.Random, profile: str = "train") -> Image.Image:
    if rng.random() < 0.7:
        return jpeg(image, rng.randint(10, 20))
    return webp(image, rng.randint(5, 20))


def blur_noise(image: Image.Image, rng: random.Random, profile: str = "train") -> Image.Image:
    array = cv2.GaussianBlur(np.asarray(image.convert("RGB"), dtype=np.float32), (0, 0), rng.uniform(0.6, 1.6))
    array += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2, 8), array.shape)
    return Image.fromarray(array.clip(0, 255).astype(np.uint8))


def moire(image: Image.Image, rng: random.Random, profile: str = "train") -> Image.Image:
    # Interference stripes from photographing a pixel grid.
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    yy, xx = np.mgrid[0:array.shape[0], 0:array.shape[1]].astype(np.float32)
    period, theta = rng.uniform(3, 7), rng.uniform(0, math.pi)
    wave = np.sin(2 * math.pi * (xx * math.cos(theta) + yy * math.sin(theta)) / period)
    return Image.fromarray((array + rng.uniform(4, 10) * wave[..., None]).clip(0, 255).astype(np.uint8))


EDITS = {
    "bubbles": bubbles, "watermark": watermark, "mirror": mirror, "rotate": rotate, "squash": squash,
    "border": border, "greyscale": greyscale, "colour_shift": colour_shift, "low_res": low_res,
    "upscale": upscale, "heavy_compression": heavy_compression, "blur_noise": blur_noise,
}
