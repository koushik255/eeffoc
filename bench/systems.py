"""Search systems compared by the benchmark.

All of them see pages and queries in grey (the index is colour-blind on
purpose: Steel Ball Run also exists in black and white) and get queries the
way a site would: shrunk to UPLOAD_SIDE on the long side, JPEG 90.

  sift     RootSIFT features of every page in one FAISS index; each query
           feature votes for the page of its nearest neighbour (ratio test
           against the nearest other page), then the best 20 pages are
           checked with RANSAC.
  curtain  curtain's combined-rocm-50k-v1 ResNet-18, exactly as the movie
           site uses it: 49 shared-trunk tiles + whole page, query + 4 pieces.
  sscd     Meta's SSCD (sscd_disc_mixup, ResNet-50) zero-shot, with the
           same tiles and pieces (each tile run separately).
  *+align  the embedding's top 10 re-ordered by SIFT + RANSAC, as on curtain.

With an NVIDIA GPU (`uv sync --extra cuda`) the curtain-architecture model,
the page vectors and the SIFT descriptor matching run on it (DEVICE); SIFT
features, RANSAC and SSCD stay on the CPU.
"""

from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from bench import catalog as catalogs
from bench import views
from bench.catalog import LOCAL, PAGES

INDEX = LOCAL / "index"
# Curtain-architecture checkpoints (ResNet-18 + 128-d head), by system name.
CHECKPOINTS = {
    "curtain": Path.home() / "curtain/.local/trained_models/combined-rocm-50k-v1/best.pt",
}
SSCD_MODEL = LOCAL / "models/sscd_disc_mixup.torchscript.pt"
UPLOAD_SIDE = 1024
TOP = 10
GRID = 4                    # tiles per side of a page
STEPS = 2 * GRID - 1        # half-stride tile positions per side: 7 x 7 = 49 tiles
FINE_GRIDS = (4, 6)         # "+fine": also a 6 x 6 grid (11 x 11 = 121 tiles, ~2.8% of the page each)
SIFT_PAGE_SIDE = 1280       # pages are described at this long side
SIFT_QUERY_SIDE = 1024
SIFT_FEATURES = 3000
RATIO = 0.8
VERIFIED_POINTS = 15        # RANSAC inliers that count as a confirmed match
STRONG_POINTS = 50          # this many on the first candidate: confirmed, the rest aren't checked
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
POOL = ThreadPoolExecutor(16)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_WORKER = os.environ.get("EEFFOC_GPU")  # a bench.gpu worker (http://host:port) to build indexes on


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue()))


def upload(image: Image.Image) -> Image.Image:
    # What a browser would send: at most UPLOAD_SIDE on the long side, JPEG 90, then grey.
    scale = UPLOAD_SIDE / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    return jpeg(image, 90).convert("L")


def catalog(set_name: str) -> list[dict]:
    return catalogs.load(set_name)


def checkpoint(name: str) -> Path:
    """A named checkpoint, or a fine-tuned run in .local/runs/<name>/best.pt."""
    return CHECKPOINTS.get(name, LOCAL / "runs" / name / "best.pt")


def page_grey(page: dict) -> Image.Image:
    with Image.open(PAGES / page["path"]) as image:
        return image.convert("L")


def pieces(image: Image.Image) -> list[Image.Image]:
    """curtain's four overlapping corner pieces, each 60% of the width and height."""
    w, h = image.size
    pw, ph = int(w * 0.6), int(h * 0.6)
    return [image.crop((x, y, x + pw, y + ph)) for y in (0, h - ph) for x in (0, w - pw)]


def tensor(image: Image.Image) -> torch.Tensor:
    # Grey PIL image -> [1, 3, H, W] float in 0..1.
    array = torch.from_numpy(np.asarray(image.convert("L"), dtype=np.uint8).copy())
    return array.float().div_(255)[None, None].expand(1, 3, -1, -1)


def normalized(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    images = functional.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=True)
    return (images - IMAGENET_MEAN.to(images.device)) / IMAGENET_STD.to(images.device)


def rank_tiles(vectors: torch.Tensor, queries: np.ndarray) -> list[int]:
    """curtain's scoring (bench.views.page_scores) for the query and its pieces."""
    return np.argsort(-views.page_scores(vectors, queries))[:TOP].tolist()


# ------------------------------------------------------------------ SIFT

def sift(grey: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    """RootSIFT points (in the resized image's pixels) and descriptors."""
    scale = min(1.0, side / max(grey.shape))
    if scale < 1:
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=SIFT_FEATURES).detectAndCompute(grey, None)
    if descriptors is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
    return np.float32([k.pt for k in keypoints]), np.sqrt(descriptors).astype(np.float32)


class SiftPages:
    """Every page's SIFT features, cached in .local/index/<set>."""

    def __init__(self, pages: list[dict], set_name: str):
        path = INDEX / set_name / f"sift-{SIFT_PAGE_SIDE}-{SIFT_FEATURES}.npz"
        if not path.exists():
            started = time.perf_counter()
            found = list(POOL.map(lambda p: sift(np.asarray(page_grey(p)), SIFT_PAGE_SIDE), pages))
            counts = np.array([len(f[0]) for f in found])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, points=np.concatenate([f[0] for f in found]),
                     descriptors=np.concatenate([f[1] for f in found]).astype(np.float16),
                     offsets=np.concatenate([[0], np.cumsum(counts)]))
            print(f"  SIFT for {len(pages)} pages in {time.perf_counter() - started:.0f}s", flush=True)
        data = np.load(path)
        self.points, self.offsets = data["points"], data["offsets"]
        self.descriptors = data["descriptors"]  # fp16; a page's rows become fp32 when matched
        self.owner = np.repeat(np.arange(len(pages)), np.diff(self.offsets))

    def of(self, page: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.offsets[page], self.offsets[page + 1]
        return self.points[a:b], self.descriptors[a:b].astype(np.float32)

    def align(self, query: tuple[np.ndarray, np.ndarray], page: int):
        """(inliers, matrix, matched): the affine map from query SIFT pixels to
        the page's SIFT pixels (the page at SIFT_PAGE_SIDE) that most matches
        agree on, and the query points of those matches."""
        (qp, qd), (pp, pd) = query, self.of(page)
        if len(qd) < 4 or len(pd) < 4:
            return 0, None, None
        query_rows, page_rows = ratio_matches(qd, pd)
        if len(query_rows) < 6:
            return len(query_rows) // 2, None, None
        source = qp[query_rows]
        matrix, mask = cv2.estimateAffine2D(source, pp[page_rows], method=cv2.RANSAC, ransacReprojThreshold=6.0)
        if matrix is None:
            return 0, None, None
        return int(mask.sum()), matrix, source[mask.ravel() == 1]

    def inliers(self, query: tuple[np.ndarray, np.ndarray], page: int) -> int:
        """Matches that agree on one affine map from the query into the page."""
        return self.align(query, page)[0]

    def rerank(self, query: tuple[np.ndarray, np.ndarray], candidates: list[int]) -> list[int]:
        """Confirmed pages (VERIFIED_POINTS+ inliers) first, most inliers first;
        the rest keep their order. When the first candidate alone gets
        STRONG_POINTS (most queries), the others aren't checked."""
        first = self.inliers(query, candidates[0])
        if first >= STRONG_POINTS:
            return candidates
        points = [first] + list(POOL.map(lambda c: self.inliers(query, c), candidates[1:]))
        confirmed = sorted((i for i, n in enumerate(points) if n >= VERIFIED_POINTS), key=lambda i: -points[i])
        return [candidates[i] for i in confirmed] + [c for i, c in enumerate(candidates) if i not in confirmed]


def ratio_matches(query: np.ndarray, page: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(query rows, page rows): each query descriptor's nearest page descriptor,
    kept when it's nearer than RATIO x the second nearest."""
    if DEVICE.type == "cpu":
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query, page, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
        return np.array([m.queryIdx for m in good], int), np.array([m.trainIdx for m in good], int)
    # The same brute-force match as one distance matrix on the GPU (~3000 x 3000).
    query, page = torch.from_numpy(query).to(DEVICE), torch.from_numpy(page).to(DEVICE)
    distances, nearest = torch.cdist(query, page).topk(2, dim=1, largest=False)
    keep = distances[:, 0] < RATIO * distances[:, 1]
    return keep.nonzero()[:, 0].cpu().numpy(), nearest[keep, 0].cpu().numpy()


def system_sift(pages: list[dict], sift_pages: SiftPages):
    started = time.perf_counter()
    quantizer = faiss.IndexFlatL2(128)
    index = faiss.IndexIVFFlat(quantizer, 128, 2048)
    sample = np.random.default_rng(0).choice(len(sift_pages.descriptors), 200_000, replace=False)
    index.train(sift_pages.descriptors[sample].astype(np.float32))
    for start in range(0, len(sift_pages.descriptors), 1 << 20):
        index.add(sift_pages.descriptors[start:start + (1 << 20)].astype(np.float32))
    index.nprobe = 24
    print(f"  FAISS over {index.ntotal:,} SIFT features in {time.perf_counter() - started:.0f}s", flush=True)

    def run(image: Image.Image) -> list[int]:
        query = sift(np.asarray(image), SIFT_QUERY_SIDE)
        if len(query[1]) == 0:
            return []
        distances, ids = index.search(query[1], 8)
        owners = np.where(ids >= 0, sift_pages.owner[np.maximum(ids, 0)], -1)
        # Ratio test against the nearest neighbour on a different page (squared distances).
        other = (owners != owners[:, :1]) & (owners >= 0)
        second = np.where(other.any(axis=1), distances[np.arange(len(ids)), other.argmax(axis=1)], np.inf)
        keep = (owners[:, 0] >= 0) & (distances[:, 0] < RATIO ** 2 * second)
        votes = np.bincount(owners[keep, 0], minlength=len(pages))
        candidates = np.argsort(-votes, kind="stable")[:20].tolist()
        return sift_pages.rerank(query, candidates)[:TOP]
    return run


# ------------------------------------------------------------------ curtain

class CurtainEncoder(nn.Module):
    # Same layout as curtain_ml.model.CurtainEncoder, so its checkpoint loads as is.
    def __init__(self, embedding_dim: int, image_size: int):
        super().__init__()
        self.image_size = image_size
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.backbone(images), dim=1)

    def trunk(self, images: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        for layer in (b.conv1, b.bn1, b.relu, b.maxpool, b.layer1, b.layer2, b.layer3, b.layer4):
            images = layer(images)
        return images


def load_curtain(path: Path) -> CurtainEncoder:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = CurtainEncoder(int(config["embedding_dim"]), int(config["image_size"]))
    model.load_state_dict(saved["model"])
    return model.to(DEVICE).eval()


def curtain_page(model: CurtainEncoder, page: Image.Image, grids: tuple[int, ...] = (GRID,)) -> np.ndarray:
    """Tile vectors, then the whole page: [50, 128] for the default 4 x 4 grid.

    For each grid g, one trunk pass at g x 192 (rcmindex's way) and the
    (2g - 1)^2 half-stride windows of g x g cells. Like curtain, the page is
    squashed to a square."""
    size = model.image_size
    image = tensor(page).to(next(model.parameters()).device)
    tiles = []
    for grid in grids:
        features = model.trunk(normalized(image, (grid * size, grid * size)))[0]
        cells = features.shape[-1] // grid
        stride = cells // 2
        steps = 2 * grid - 1
        pooled = torch.stack([features[:, y * stride:y * stride + cells, x * stride:x * stride + cells].mean((1, 2))
                              for y in range(steps) for x in range(steps)])
        tiles.append(functional.normalize(model.backbone.fc(pooled), dim=1))
    whole = model(normalized(image, (size, size)))
    return torch.cat([*tiles, whole]).float().cpu().numpy()


def curtain_views(model: CurtainEncoder, image: Image.Image, views: list[Image.Image] | None = None) -> np.ndarray:
    """Embeddings of the query and its corner pieces, or of the given views."""
    size = model.image_size
    views = views or [image, *pieces(image)]
    device = next(model.parameters()).device
    return model(torch.cat([normalized(tensor(v).to(device), (size, size)) for v in views])).float().cpu().numpy()


def curtain_groups(model: CurtainEncoder, image: Image.Image) -> list[np.ndarray]:
    """Embeddings of each of bench.views' groups of views of the image, all in one forward pass."""
    groups = views.groups(image)
    embedded = curtain_views(model, image, [v for g in groups for v in g])
    return np.split(embedded, np.cumsum([len(g) for g in groups])[:-1])


# ------------------------------------------------------------------ SSCD

SSCD_SIDE = 288  # SSCD's evaluation size: the short side resized to 288


def sscd_size(w: int, h: int) -> tuple[int, int]:
    scale = SSCD_SIDE / min(w, h)
    return round(h * scale), round(w * scale)


def sscd_embed(model, images: list[Image.Image]) -> torch.Tensor:
    # Images of one size, keeping their aspect ratio (SSCD isn't trained on squashed inputs).
    size = sscd_size(*images[0].size)
    batch = torch.cat([normalized(tensor(i), size) for i in images])
    return functional.normalize(model(batch), dim=1)


def sscd_page_per_tile(model, page: Image.Image) -> np.ndarray:
    """[50, 512]: the 49 half-stride tiles, each run separately, then the whole page.
    The reference for sscd_page; ~10 s a page on CPU."""
    w, h = page.size
    tw, th = w // GRID, h // GRID
    tiles = [page.crop((round(x * tw / 2), round(y * th / 2), round(x * tw / 2) + tw, round(y * th / 2) + th))
             for y in range(STEPS) for x in range(STEPS)]
    return torch.cat([sscd_embed(model, tiles), sscd_embed(model, [page])]).numpy()


def sscd_page(model, page: Image.Image) -> np.ndarray:
    """[50, 512] like sscd_page_per_tile, from one trunk pass at GRID x the tile
    size: GeM-pool each tile's window of the feature map, then SSCD's head."""
    th, tw = sscd_size(page.width // GRID, page.height // GRID)
    features = model.backbone(normalized(tensor(page), (GRID * th, GRID * tw)))
    fh, fw = features.shape[-2:]
    rows = [(round(y * fh / (2 * GRID)), round((y + 2) * fh / (2 * GRID))) for y in range(STEPS)]
    cols = [(round(x * fw / (2 * GRID)), round((x + 2) * fw / (2 * GRID))) for x in range(STEPS)]
    tiles = torch.cat([model.embeddings(features[:, :, r0:r1, c0:c1]) for r0, r1 in rows for c0, c1 in cols])
    return torch.cat([tiles, sscd_embed(model, [page])]).numpy()


def sscd_views(model, image: Image.Image) -> np.ndarray:
    return torch.cat([sscd_embed(model, [image]), sscd_embed(model, pieces(image))]).numpy()


# ------------------------------------------------------------------ embedding systems

def vectors_path(name: str, set_name: str) -> Path:
    return INDEX / set_name / f"{name}.npy"


def page_vectors(name: str, set_name: str, pages: list[dict], embed_page, map_pages=map) -> np.ndarray:
    """[pages, vectors per page, d] float32, cached in vectors_path(name, set_name).
    embed_page(page dict) runs over the pages with map_pages (a worker's pool.map runs several at once)."""
    path = vectors_path(name, set_name)
    if path.exists():
        return np.load(path)
    started = time.perf_counter()
    out = []
    with torch.inference_mode():
        for i, vectors in enumerate(map_pages(embed_page, pages)):
            out.append(vectors)
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.perf_counter() - started)
                print(f"  {name}: {i + 1}/{len(pages)} pages, {rate:.1f}/s", flush=True)
    vectors = np.stack(out).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    print(f"  {name}: {len(pages)} pages in {time.perf_counter() - started:.0f}s", flush=True)
    return vectors


def system_embedding(name: str, set_name: str, pages: list[dict], sift_pages: SiftPages | None,
                     align_top: int = 0, extra_views: bool = False, fine: bool = False):
    """name: "sscd", or a curtain-architecture checkpoint (see checkpoint()).
    extra_views: also search with the content crop, strip windows and halves (bench.views).
    fine: index pages with FINE_GRIDS instead of the 4 x 4 grid alone."""
    grids, map_pages = FINE_GRIDS if fine else (GRID,), map
    if (extra_views or fine) and name == "sscd":
        raise ValueError("extra views and fine tiles are only wired up for curtain-architecture models")
    if name == "sscd":
        model = torch.jit.load(str(SSCD_MODEL), map_location="cpu").eval()
        embed_page, embed_views = (lambda p: sscd_page(model, page_grey(p))), (lambda i: sscd_views(model, i))
    else:
        model = load_curtain(checkpoint(name))
        embed_page = lambda p: curtain_page(model, page_grey(p), grids)  # noqa: E731
        embed_views = lambda i: curtain_views(model, i)  # noqa: E731
        embed_groups = lambda i: curtain_groups(model, i)  # noqa: E731
    index_name = name + ("-fine" if fine else "")
    if GPU_WORKER and name != "sscd" and not vectors_path(index_name, set_name).exists():
        from bench.gpu import Worker  # build the index on the GPU machine; searches stay here
        worker = Worker(GPU_WORKER, checkpoint(name), grids)
        embed_page, map_pages = worker.page, worker.pool.map
    # One tensor on DEVICE for the whole search (318 MB with fine tiles), scored there.
    vectors = page_vectors(index_name, set_name, pages, embed_page, map_pages)
    vectors = torch.from_numpy(vectors).to(DEVICE)

    def run(image: Image.Image) -> list[int]:
        with torch.inference_mode():
            if extra_views:
                found = views.rank(vectors, embed_groups(image), TOP)
            else:
                found = rank_tiles(vectors, embed_views(image))
        if align_top:
            query = sift(np.asarray(image), SIFT_QUERY_SIDE)
            found = sift_pages.rerank(query, found[:align_top]) + found[align_top:]
        return found
    return run
