"""Fine-tune a curtain encoder on manga with mangashot positives.

Each training page gives one pair:

  positive  mangashot.random_shot(page): a region of the page with edits and
            maybe a phone / browser / photo scene around it (train profile)
  anchor    the clean view the index stores for that region: of the page's
            49 half-stride tiles and the whole page, the one that overlaps
            the shot's region best

Every other page in the batch is a negative (symmetric InfoNCE, as in
curtain). After every epoch the model is scored the way bench.run scores it
on held-out pages: shared-trunk tiles for the pages, query + 4 pieces for
test-profile shots, exact page first.

Runs on CUDA when there is one, else CPU (only sensible for a smoke test):

    .venv/bin/python -m training.finetune --data .local --output .local/runs/smoke \\
        --init ~/curtain/.local/trained_models/combined-rocm-50k-v1/best.pt --epochs 1 --steps 3 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import mangashot
from bench.systems import (GRID, IMAGENET_MEAN, IMAGENET_STD, STEPS, CurtainEncoder, curtain_page,
                           curtain_views, rank_tiles)


@dataclass(frozen=True)
class Config:
    data_dir: str                 # holds train-pages/pages.json
    output_dir: str
    init: str | None = None       # checkpoint to start from (curtain's)
    epochs: int = 15
    steps: int | None = None      # steps per epoch; default: one pass over the training pages
    batch_size: int = 256
    image_size: int = 192
    embedding_dim: int = 128
    learning_rate: float = 1.5e-4
    temperature: float = 0.07
    weight_decay: float = 1e-4
    workers: int = 16
    validation_shots: int = 400
    seed: int = 42


def tile_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """The index's 49 half-stride tiles, then the whole page (same order as curtain_page)."""
    tw, th = width / GRID, height / GRID
    boxes = [(round(x * tw / 2), round(y * th / 2), round(x * tw / 2 + tw), round(y * th / 2 + th))
             for y in range(STEPS) for x in range(STEPS)]
    return boxes + [(0, 0, width, height)]


def best_tile(box, width: int, height: int):
    def iou(a, b):
        w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = w * h
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return max(tile_boxes(width, height), key=lambda t: iou(box, t))


def square(image: Image.Image, size: int) -> torch.Tensor:
    # Grey, squashed to size x size like the index and the query side: uint8 [1, size, size].
    grey = image.convert("L").resize((size, size), Image.BILINEAR, reducing_gap=3.0)
    return torch.from_numpy(np.asarray(grey, dtype=np.uint8).copy())[None]


class ShotPairs(Dataset):
    """(anchor, positive) uint8 tensors for one page per item."""

    def __init__(self, root: Path, pages: list[dict], neighbours: list[int | None], size: int):
        self.root, self.pages, self.neighbours, self.size = root, pages, neighbours, size

    def __len__(self) -> int:
        return len(self.pages)

    def open(self, index: int) -> Image.Image:
        with Image.open(self.root / self.pages[index]["path"]) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int):
        rng = random.Random(random.getrandbits(64))
        page = self.open(index)
        n = self.neighbours[index]
        shot = mangashot.random_shot(page, rng, "train", neighbour=self.open(n) if n is not None else None)
        anchor = page.crop(best_tile(shot.boxes[0][1], page.width, page.height))
        return square(anchor, self.size), square(shot.image, self.size)


def seed_worker(worker: int) -> None:
    random.seed(torch.initial_seed() % 2**63)
    np.random.seed(torch.initial_seed() % 2**32)


def contrastive_loss(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    # Symmetric in-batch InfoNCE (curtain's).
    batch = first.shape[0]
    embeddings = torch.cat((first, second)).float()
    logits = embeddings @ embeddings.T / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = (torch.arange(2 * batch, device=logits.device) + batch) % (2 * batch)
    return functional.cross_entropy(logits, targets)


def to_input(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = batch.to(device, non_blocking=True).float().div_(255).expand(-1, 3, -1, -1)
    return (images - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def neighbours_of(pages: list[dict]) -> list[int | None]:
    # The next page of the same volume, for two-page views and readers.
    return [i + 1 if i + 1 < len(pages) and (pages[i + 1]["series"], pages[i + 1]["volume"])
            == (p["series"], p["volume"]) else None for i, p in enumerate(pages)]


class Validation:
    """Held-out pages scored like bench.run: exact page first over test-profile shots."""

    def __init__(self, root: Path, pages: list[dict], count: int, seed: int):
        self.root, self.pages = root, pages
        rng = random.Random(seed)
        neighbours = neighbours_of(pages)
        self.queries = []
        kinds = list(mangashot.KINDS)
        while len(self.queries) < count:
            i = rng.randrange(len(pages))
            n = neighbours[i]
            shot = mangashot.make(kinds[len(self.queries) % len(kinds)], self.open(i), rng, "test",
                                  neighbour=self.open(n) if n is not None else None)
            if shot is not None:
                targets = [i, n]
                self.queries.append((shot.kind, shot.image.convert("L"), [targets[k] for k, _ in shot.boxes]))

    def open(self, index: int) -> Image.Image:
        with Image.open(self.root / self.pages[index]["path"]) as image:
            return image.convert("RGB")

    @torch.inference_mode()
    def __call__(self, model: CurtainEncoder) -> dict[str, float]:
        model.eval()
        vectors = np.stack([curtain_page(model, self.open(i).convert("L")) for i in range(len(self.pages))])
        hits: dict[str, list[bool]] = {}
        for kind, image, targets in self.queries:
            found = rank_tiles(vectors, curtain_views(model, image))
            hits.setdefault(kind, []).append(found[0] in targets)
        model.train()
        exact = float(np.mean([h for v in hits.values() for h in v]))
        return {"validation_exact": exact, **{f"exact_{k}": float(np.mean(v)) for k, v in hits.items()}}


def train(config: Config, commit=lambda: None) -> dict:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    root = Path(config.data_dir) / "train-pages"
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    pages = json.loads((root / "pages.json").read_text())["pages"]
    train_pages = [p for p in pages if p["split"] == "train"]
    held_out = [p for p in pages if p["split"] == "validation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CurtainEncoder(config.embedding_dim, config.image_size)
    if config.init:
        model.load_state_dict(torch.load(config.init, map_location="cpu", weights_only=False)["model"])
    model.to(device).train()
    loader = DataLoader(ShotPairs(root, train_pages, neighbours_of(train_pages), config.image_size),
                        batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=config.workers,
                        worker_init_fn=seed_worker, persistent_workers=config.workers > 0,
                        pin_memory=device.type == "cuda", prefetch_factor=4 if config.workers else None)
    steps = min(config.steps or len(loader), len(loader))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total = steps * config.epochs
    warmup = max(1, min(steps, total // 10))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total - warmup))))
    validate = Validation(root, held_out, config.validation_shots, config.seed + 1)
    print(json.dumps({"train_pages": len(train_pages), "validation_pages": len(held_out),
                      "validation_shots": len(validate.queries), "steps_per_epoch": steps,
                      "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}),
          flush=True)
    if (output / "last.pt").exists():
        # A restarted run (Modal preempts GPUs): carry on after the last finished epoch.
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        log = json.loads((output / "log.json").read_text())
        best = max(m["validation_exact"] for m in log)
        print(json.dumps({"resumed_after_epoch": saved["metrics"]["epoch"], "best": best}), flush=True)
    else:
        started = time.perf_counter()
        metrics = {"epoch": 0, **validate(model), "seconds": time.perf_counter() - started}
        print(json.dumps(metrics), flush=True)
        best = metrics["validation_exact"]
        log = [metrics]
    for epoch in range(log[-1]["epoch"] + 1, config.epochs + 1):
        started = time.perf_counter()
        losses = []
        for step, (anchors, positives) in enumerate(loader):
            if step >= steps:
                break
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                embeddings = model(torch.cat([to_input(anchors, device), to_input(positives, device)]))
            first, second = embeddings.float().split(len(anchors))
            loss = contrastive_loss(first, second, config.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            losses.append(loss.item())
        training = time.perf_counter() - started
        metrics = {"epoch": epoch, "loss": float(np.mean(losses)), "learning_rate": float(schedule.get_last_lr()[0]),
                   "pairs_per_second": steps * config.batch_size / training, **validate(model),
                   "seconds": time.perf_counter() - started}
        print(json.dumps(metrics), flush=True)
        log.append(metrics)
        saved = {"model": model.state_dict(), "config": asdict(config), "metrics": metrics}
        if metrics["validation_exact"] > best:
            best = metrics["validation_exact"]
            torch.save(saved, output / "best.pt")
        torch.save({**saved, "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict()},
                   output / "last.pt")
        (output / "log.json").write_text(json.dumps(log, indent=1))
        commit()
    if not (output / "best.pt").exists():  # never beat the starting point: keep the last one anyway
        torch.save({"model": model.state_dict(), "config": asdict(config), "metrics": log[-1]},
                   output / "best.pt")
    return {"best_validation_exact": best, "start_validation_exact": log[0]["validation_exact"], **log[-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--validation-shots", type=int, default=400)
    args = parser.parse_args()
    print(train(Config(data_dir=args.data, output_dir=args.output, init=args.init, epochs=args.epochs,
                       steps=args.steps, batch_size=args.batch_size, workers=args.workers,
                       learning_rate=args.learning_rate, validation_shots=args.validation_shots)))


if __name__ == "__main__":
    main()
