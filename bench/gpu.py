"""Embed pages on a GPU machine for the others on the network, like a
serverless GPU: it keeps nothing but the model in memory.

The GPU machine needs only this repo and `uv sync --extra cuda`; no pages,
index or checkpoints. Start the worker there:

    .venv/bin/python -m bench.gpu --port 8800

and point bench.run or bench.find at it from the machine that has .local:

    EEFFOC_GPU=http://gpu-box:8800 .venv/bin/python -m bench.run --suite sbr-v3 --systems manga-ft1+fine

When an index isn't built yet, the client sends the model's weights (the
worker keeps them until it stops), then each page's file, and gets its
vectors back. Searches always run on the client. There's no
authentication: run the worker only on a network you trust, or listen on
127.0.0.1 and reach it through `ssh -L`.

    GET  /model/<id>               200 if the worker has the model, else 404
    PUT  /model/<id>               the model: {"embedding_dim", "image_size", "state"} (torch.save)
    POST /page/<id>?grids=4,6      a page image -> .npy [tiles + 1, d] (systems.curtain_page)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image

from bench import systems
from bench.catalog import PAGES

MAX_BODY = 100 << 20  # bytes: the model is ~45 MB
IN_FLIGHT = 4         # requests a client keeps going at once while indexing
TIMEOUT = 120         # seconds a request may take (a page takes well under 1)
RETRY_WAIT, MAX_WAIT = 5, 300  # seconds between tries when the worker can't be reached, doubling up to MAX_WAIT
# Every request gets its own thread, but the model runs on this one: CUDA work from each new
# thread leaks ~4 MB of memory on Windows (27 GB after 7,000 pages).
GPU = ThreadPoolExecutor(1)


# ------------------------------------------------------------------ worker (the GPU machine)

def embed_page(model: systems.CurtainEncoder, image: Image.Image, grids: tuple[int, ...]) -> np.ndarray:
    with torch.inference_mode():  # per thread, so here on GPU's
        return systems.curtain_page(model, image, grids)


class Handler(BaseHTTPRequestHandler):
    models: ClassVar[dict[str, systems.CurtainEncoder]] = {}  # by id, shared by every request

    def route(self) -> tuple[str, str, dict[str, list[str]]]:
        url = urlparse(self.path)
        kind, model_id = (url.path.strip("/").split("/") + [""])[:2]
        return kind, model_id, parse_qs(url.query)

    def body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_BODY:
            raise ValueError(f"send 1 byte to {MAX_BODY >> 20} MB")
        return self.rfile.read(length)

    def reply(self, data: bytes = b"") -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        kind, model_id, _ = self.route()
        if kind != "model" or model_id not in self.models:
            return self.send_error(404, "unknown model")
        self.reply()

    def do_PUT(self) -> None:
        kind, model_id, _ = self.route()
        if kind != "model" or not model_id:
            return self.send_error(404)
        try:
            saved = torch.load(io.BytesIO(self.body()), map_location="cpu", weights_only=True)  # tensors only, no code
            model = systems.CurtainEncoder(int(saved["embedding_dim"]), int(saved["image_size"]))
            model.load_state_dict(saved["state"])
        except Exception as error:  # noqa: BLE001 - whatever was sent, it isn't a model
            return self.send_error(400, f"not a model: {error}")
        self.models[model_id] = GPU.submit(lambda: model.to(systems.DEVICE).eval()).result()
        print(json.dumps({"loaded": model_id}), flush=True)
        self.reply()

    def do_POST(self) -> None:
        kind, model_id, params = self.route()
        if kind != "page":
            return self.send_error(404)
        if model_id not in self.models:
            return self.send_error(404, "unknown model")
        model, out = self.models[model_id], io.BytesIO()
        try:
            image = Image.open(io.BytesIO(self.body())).convert("L")
        except (ValueError, OSError, RuntimeError) as error:  # too big, or PIL can't read it
            return self.send_error(400, f"not an image: {error}")
        grids = tuple(int(g) for g in params.get("grids", ["4"])[0].split(","))
        np.save(out, GPU.submit(embed_page, model, image, grids).result())
        self.reply(out.getvalue())

    def log_message(self, *args) -> None:  # one line a request is too many while indexing
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0: every network interface")
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", "device": str(systems.DEVICE),
                      "gpu": torch.cuda.get_device_name(0) if systems.DEVICE.type == "cuda" else None}), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


# ------------------------------------------------------------------ client (the machine with .local)

class Worker:
    """A bench.gpu worker at url, embedding pages with checkpoint the way
    systems.curtain_page does locally. When the worker can't be reached (its
    machine asleep, the tunnel down), requests wait and try again; when it
    has restarted without the model, the model is sent again."""

    def __init__(self, url: str, checkpoint: Path, grids: tuple[int, ...]):
        self.url, self.grids = url.rstrip("/"), ",".join(map(str, grids))
        self.checkpoint, self.id = checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16]
        self.pool, self.sending = ThreadPoolExecutor(IN_FLIGHT), Lock()
        self.send_model()

    def send_model(self) -> None:
        with self.sending:  # once, however many requests found it missing
            try:
                self.request("GET", "model")
                return
            except HTTPError as error:
                if error.code != 404:
                    raise
            saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)  # our own checkpoint
            model = io.BytesIO()
            torch.save({"embedding_dim": int(saved["config"]["embedding_dim"]),
                        "image_size": int(saved["config"]["image_size"]), "state": saved["model"]}, model)
            self.request("PUT", "model", model.getvalue())

    def request(self, method: str, kind: str, body: bytes | None = None, **params) -> bytes:
        url = f"{self.url}/{kind}/{self.id}" + (f"?{urlencode(params)}" if params else "")
        for attempt in itertools.count():
            try:
                with urlopen(Request(url, data=body, method=method), timeout=TIMEOUT) as response:
                    return response.read()
            except HTTPError as error:
                if error.code != 404 or kind == "model":
                    raise
                self.send_model()  # the worker restarted
            except (URLError, OSError) as error:  # unreachable, refused, timed out
                wait = min(RETRY_WAIT * 2 ** attempt, MAX_WAIT)
                print(f"  worker unreachable ({error}); trying again in {wait}s", flush=True)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def embed(self, image_file: bytes) -> np.ndarray:
        """systems.curtain_page of an image file (any format PIL reads), run on the worker."""
        return np.load(io.BytesIO(self.request("POST", "page", image_file, grids=self.grids)))

    def page(self, page: dict) -> np.ndarray:
        # The page's file as it is: the worker reads it the way systems.page_grey does.
        return self.embed((PAGES / page["path"]).read_bytes())


if __name__ == "__main__":
    main()
