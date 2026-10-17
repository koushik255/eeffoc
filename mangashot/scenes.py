"""Where the query is shown: phone apps, desktop readers, photos of screens.

Every scene is scene(content, rng, profile, neighbour=None) -> image. The
content (an already edited region of a page) is always shown whole; the
neighbour (another page of the same volume) may peek in around it.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

from mangashot.core import fit_inside, fit_width, font, pick

PHONE = (750, 1624)   # a phone screen at half resolution
# (background, ink, bar) colours. Every third is test-only (see core.pool).
THEMES = [
    ((0, 0, 0), (255, 255, 255), (28, 28, 30)),
    ((255, 255, 255), (0, 0, 0), (242, 242, 247)),
    ((18, 18, 18), (230, 230, 230), (40, 40, 40)),
    ((250, 246, 236), (40, 30, 20), (235, 225, 205)),
    ((21, 32, 43), (255, 255, 255), (30, 45, 60)),
    ((245, 245, 245), (20, 20, 20), (225, 225, 225)),
]
CAPTIONS = [
    "this panel lives in my head rent free", "peak fiction no debate", "who drew this?? source pls",
    "rereading this and it still hits", "what chapter is this from", "the art in this one is insane",
    "never recovered from this page", "sauce?", "best fight in the series imo",
]
NAMES = ["@manga_fan", "@panelhunter", "@readerbot", "@inkdrop", "@tl_notes", "@nightreader"]
CHAT = ["look at this", "WHAT", "bro read this chapter", "lmaooo", "no way", "page 12 omg",
        "ok this is peak", "spoilers!!", "which volume?"]
BACKGROUNDS = [(40, 35, 30), (120, 90, 60), (200, 200, 195), (15, 15, 20), (90, 100, 110), (170, 150, 120)]


def status_bar(draw: ImageDraw.ImageDraw, width: int, height: int, ink, rng: random.Random, profile: str) -> None:
    f = font(height // 2, rng, profile)
    draw.text((width * 0.08, height / 2), f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d}", fill=ink, font=f,
              anchor="lm")
    bx = width * 0.84
    draw.rounded_rectangle((bx, height * 0.32, bx + width * 0.07, height * 0.68), radius=3, outline=ink, width=2)
    draw.rectangle((bx + 3, height * 0.32 + 3, bx + width * 0.07 * rng.uniform(0.3, 0.95), height * 0.68 - 3),
                   fill=ink)
    for i in range(4):
        x = width * 0.72 + i * 7
        draw.rectangle((x, height * (0.65 - 0.08 * i), x + 4, height * 0.68), fill=ink)


def rounded(image: Image.Image, radius: int) -> tuple[Image.Image, Image.Image]:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, image.width - 1, image.height - 1), radius=radius, fill=255)
    return image, mask


def scroll_reader(content, rng, profile, neighbour=None):
    """A phone reader scrolled so the content sits somewhere down the screen,
    the neighbouring page running into it from above or below."""
    W, H = PHONE
    bg, ink, bar = pick(rng, THEMES, profile)
    canvas = Image.new("RGB", (W, H), bg)
    shown = fit_width(content, W)
    top = rng.randint(-shown.height // 4, max(60, H - shown.height)) if shown.height < H else rng.randint(-200, 60)
    if neighbour is not None:
        other = fit_width(neighbour, W)
        gap = rng.choice([0, 8, 24])
        if rng.random() < 0.5:
            canvas.paste(other, (0, top - other.height - gap))
        else:
            canvas.paste(other, (0, top + shown.height + gap))
    canvas.paste(shown, (0, top))
    draw = ImageDraw.Draw(canvas)
    if rng.random() < 0.6:  # toolbar overlay
        overlay = Image.new("RGBA", (W, 90), bar + (rng.randint(170, 240),))
        canvas.paste(overlay, (0, H - 90), overlay)
        draw.text((W / 2, H - 45), f"{rng.randint(2, 40)} / {rng.randint(41, 220)}", fill=ink,
                  font=font(30, rng, profile), anchor="mm")
    draw.rectangle((0, 0, W, 50), fill=bg)
    status_bar(draw, W, 50, ink, rng, profile)
    return canvas


def paged_reader(content, rng, profile, neighbour=None):
    """A phone reader showing one page fitted to the screen, title and slider bars visible."""
    W, H = PHONE
    bg, ink, bar = pick(rng, THEMES, profile)
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    chrome = rng.random() < 0.7
    top, bottom = (140, 150) if chrome else (50, 0)
    shown = fit_inside(content, W, H - top - bottom)
    canvas.paste(shown, ((W - shown.width) // 2, top + (H - top - bottom - shown.height) // 2))
    if chrome:
        draw.rectangle((0, 0, W, 140), fill=bar)
        draw.text((30, 95), f"Chapter {rng.randint(1, 120)}", fill=ink, font=font(32, rng, profile), anchor="lm")
        draw.rectangle((0, H - 150, W, H), fill=bar)
        y = H - 95
        draw.line((60, y, W - 60, y), fill=ink, width=4)
        knob = rng.uniform(60, W - 60)
        draw.ellipse((knob - 14, y - 14, knob + 14, y + 14), fill=ink)
    status_bar(draw, W, 50, ink, rng, profile)
    return canvas


def social_post(content, rng, profile, neighbour=None):
    """A post in a feed: name and caption above, the image, buttons below."""
    W, H = PHONE
    bg, ink, bar = pick(rng, THEMES, profile)
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 50, W, 140), fill=bar)
    draw.text((30, 95), rng.choice(["Post", "Home", "Thread"]), fill=ink, font=font(36, rng, profile), anchor="lm")
    y = rng.randint(150, 260)
    draw.ellipse((30, y, 100, y + 70), fill=(rng.randint(60, 200), rng.randint(60, 200), rng.randint(60, 200)))
    draw.text((120, y + 20), pick(rng, NAMES, profile) + str(rng.randint(1, 999)), fill=ink,
              font=font(28, rng, profile), anchor="lm")
    draw.text((30, y + 100), pick(rng, CAPTIONS, profile), fill=ink, font=font(30, rng, profile), anchor="lm")
    margin = rng.choice([0, 0, 24, 40])
    box_top = y + 150
    shown = fit_inside(content, W - 2 * margin, H - box_top - 160)
    if rng.random() < 0.3:  # square media slot, content letterboxed inside it
        side = W - 2 * margin
        slot = Image.new("RGB", (side, side), (0, 0, 0) if rng.random() < 0.5 else bar)
        inner = fit_inside(content, side, side)
        slot.paste(inner, ((side - inner.width) // 2, (side - inner.height) // 2))
        shown = slot
    image, mask = rounded(shown, rng.choice([0, 12, 24]))
    canvas.paste(image, ((W - shown.width) // 2, box_top), mask)
    y = box_top + shown.height + 20
    for i in range(4):
        draw.ellipse((40 + i * 150, y, 80 + i * 150, y + 40), outline=ink, width=3)
    draw.rectangle((0, 0, W, 50), fill=bg)
    status_bar(draw, W, 50, ink, rng, profile)
    return canvas


def chat(content, rng, profile, neighbour=None):
    """A messaging app: a few messages, then the image in a bubble."""
    W, H = PHONE
    bg, ink, bar = pick(rng, THEMES, profile)
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 50, W, 150), fill=bar)
    draw.text((W / 2, 100), rng.choice(["Group chat", "Alex", "manga club"]), fill=ink,
              font=font(32, rng, profile), anchor="mm")
    mine = (0, 132, 255) if rng.random() < 0.5 else (52, 199, 89)
    y = 180
    for _ in range(rng.randint(0, 3)):
        text = pick(rng, CHAT, profile)
        f = font(30, rng, profile)
        tw = draw.textlength(text, font=f) + 40
        right = rng.random() < 0.5
        x0 = W - 30 - tw if right else 30
        draw.rounded_rectangle((x0, y, x0 + tw, y + 60), radius=28, fill=mine if right else bar)
        draw.text((x0 + 20, y + 30), text, fill="white" if right else ink, font=f, anchor="lm")
        y += 80
    shown = fit_inside(content, int(W * rng.uniform(0.6, 0.78)), H - y - 160)
    image, mask = rounded(shown, 28)
    x = W - 30 - shown.width if rng.random() < 0.5 else 30
    canvas.paste(image, (x, y + 10), mask)
    draw.rectangle((0, H - 110, W, H), fill=bar)
    draw.rounded_rectangle((30, H - 85, W - 120, H - 30), radius=26, outline=ink, width=2)
    draw.rectangle((0, 0, W, 50), fill=bg)
    status_bar(draw, W, 50, ink, rng, profile)
    return canvas


def browser(content, rng, profile, neighbour=None):
    """A desktop browser window with the page shown at fit-to-height in the middle."""
    W, H = rng.choice([(1600, 1000), (1440, 900), (1920, 1080)])
    bg, ink, bar = pick(rng, THEMES, profile)
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, W, 44), fill=bar)
    for i in range(rng.randint(1, 5)):  # tabs
        draw.rounded_rectangle((12 + i * 230, 8, 230 + i * 230, 44), radius=8,
                               fill=bg if i == 0 else bar, outline=ink if i == 0 else None)
    draw.rectangle((0, 44, W, 92), fill=bg)
    draw.rounded_rectangle((120, 52, W - 200, 84), radius=16, fill=bar)
    draw.text((140, 68), "https://" + rng.choice(["reader", "mangadex", "comics", "scans"]) + ".example/ch/"
              + str(rng.randint(1, 300)), fill=ink, font=font(18, rng, profile), anchor="lm")
    shown = fit_inside(content, int(W * rng.uniform(0.35, 0.9)), H - 92 - rng.randint(0, 60))
    canvas.paste(shown, ((W - shown.width) // 2, 92 + rng.randint(0, max(0, H - 92 - shown.height))))
    return canvas


def photo(content, rng, profile, neighbour=None):
    """A phone photo of a screen or a printed page: tilted, blurred, unevenly lit."""
    content = fit_inside(content, 900, 900)
    w, h = content.size
    pad = int(max(w, h) * rng.uniform(0.12, 0.3))
    W, H = w + 2 * pad, h + 2 * pad
    jitter = lambda: rng.uniform(-0.07, 0.07) * max(w, h)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[pad + jitter(), pad + jitter()], [pad + w + jitter(), pad + jitter()],
                      [pad + w + jitter(), pad + h + jitter()], [pad + jitter(), pad + h + jitter()]])
    background = pick(rng, BACKGROUNDS, profile)
    image = cv2.warpPerspective(np.asarray(content.convert("RGB")), cv2.getPerspectiveTransform(src, dst), (W, H),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=background).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    angle = rng.uniform(0, 2 * math.pi)
    ramp = (xx * math.cos(angle) + yy * math.sin(angle)) / max(W, H)
    image *= (rng.uniform(0.75, 0.95) + rng.uniform(0.1, 0.3) * (ramp - ramp.min()))[..., None]
    if rng.random() < 0.5:  # a screen: moire from the pixel grid
        period, theta = rng.uniform(3, 7), rng.uniform(0, math.pi)
        image += rng.uniform(4, 10) * np.sin(2 * math.pi * (xx * math.cos(theta) + yy * math.sin(theta)) / period)[..., None]
    image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.8, 1.8))
    image += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(3, 8), image.shape)
    return Image.fromarray(image.clip(0, 255).astype(np.uint8))


def two_pages(left: Image.Image, right: Image.Image, rng: random.Random, profile: str) -> Image.Image:
    """Facing pages in a desktop or tablet reader, toolbar on top."""
    h = 1100
    a, b = (p.resize((max(1, round(p.width * h / p.height)), h), Image.LANCZOS) for p in (left, right))
    bg, ink, bar = pick(rng, THEMES, profile)
    gap, margin = rng.choice([0, 8]), rng.randint(20, 120)
    W = a.width + b.width + gap + 2 * margin
    canvas = Image.new("RGB", (W, h + 140), bg)
    canvas.paste(a, (margin, 90))
    canvas.paste(b, (margin + a.width + gap, 90))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, W, 60), fill=bar)
    draw.text((20, 30), "Reader", fill=ink, font=font(28, rng, profile), anchor="lm")
    return canvas


SCENES = {"scroll_reader": scroll_reader, "paged_reader": paged_reader, "social_post": social_post,
          "chat": chat, "browser": browser, "photo": photo}
