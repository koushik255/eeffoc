"""Fine-tune on Modal.

The volume holds the packed training pages and the starting checkpoint:

    modal volume put eeffoc-training-data .local/train-pages.tar /datasets/train-pages.tar
    modal volume put eeffoc-training-data ~/curtain/.local/trained_models/combined-rocm-50k-v1/best.pt /init/curtain.pt

Then:

    modal run cloud/train.py --run-name manga-ft1 --epochs 15
    modal volume get eeffoc-training-data /runs/manga-ft1/best.pt .local/runs/manga-ft1/best.pt
"""

from __future__ import annotations

import modal

app = modal.App("eeffoc-manga-training")
volume = modal.Volume.from_name("eeffoc-training-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # mangashot draws captions and bubbles with these (DejaVu, URW, Droid CJK), as on the dev machine.
    .apt_install("fonts-dejavu-core", "fonts-urw-base35", "fonts-droid-fallback")
    .uv_pip_install(
        "faiss-cpu>=1.9,<2",
        "numpy>=2.2,<3",
        "opencv-python-headless>=4.10,<5",
        "pillow>=11,<13",
        "torch>=2.8,<3",
        "torchvision>=0.23,<1",
    )
    .add_local_python_source("mangashot", "bench", "training")
)


@app.function(image=image, gpu="L4", cpu=32, memory=49152, timeout=6 * 60 * 60, volumes={"/data": volume})
def finetune(run_name: str, epochs: int, batch_size: int, learning_rate: float, steps: int | None,
             workers: int) -> dict:
    import tarfile
    from pathlib import Path

    from training.finetune import Config, train

    data = Path("/tmp/data")
    data.mkdir(parents=True, exist_ok=True)
    with tarfile.open("/data/datasets/train-pages.tar") as archive:
        archive.extractall(data, filter="data")
    return train(Config(data_dir=str(data), output_dir=f"/data/runs/{run_name}", init="/data/init/curtain.pt",
                        epochs=epochs, steps=steps, batch_size=batch_size, learning_rate=learning_rate,
                        workers=workers),
                 commit=volume.commit)


@app.local_entrypoint()
def main(run_name: str = "manga-smoke", epochs: int = 1, batch_size: int = 256, learning_rate: float = 1.5e-4,
         steps: int = 0, workers: int = 31) -> None:
    print(finetune.remote(run_name, epochs, batch_size, learning_rate, steps or None, workers))
