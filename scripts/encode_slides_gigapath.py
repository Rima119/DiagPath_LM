#!/usr/bin/env python
"""
encode_slides_gigapath.py

* Scan slide headers recursively, cut tiles, encode with Prov-GigaPath encoder
  (HF hub: prov-gigapath/prov-gigapath), and save HDF5 containing:
    - features: (N, 1536)  float16
    - coords:   (N, 2)     int32 (x, y)
* Safe-resume: valid H5 skipped; corrupted重新生成。
"""
# ---------------------------------------------------------------------
# 0. HuggingFace credentials & mirror
# ---------------------------------------------------------------------
import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------------------------------------------------------------------
# 1. Std imports
# ---------------------------------------------------------------------
import argparse
import multiprocessing as mp
import warnings
from pathlib import Path
from functools import partial
from typing import List, Optional

import h5py
import numpy as np
import openslide
import timm
import torch
import torchvision.transforms as T
import tqdm

# ---------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------

def is_valid_h5(fp: Path) -> bool:
    try:
        with h5py.File(fp, "r") as h5:
            return (
                "features" in h5 and "coords" in h5 and
                h5["features"].shape[0] > 0 and h5["coords"].shape[0] > 0
            )
    except Exception:
        return False


def infer_batch(img_tensors, model, device):
    with torch.no_grad():
        batch = torch.stack(img_tensors).to(device).half()
        return model(batch).cpu().numpy()   # (B, 1536)

# ---------------------------------------------------------------------
# 3. Worker
# ---------------------------------------------------------------------

def process_slide(
    header_path: str,
    model_name: str,
    gpu_ids: List[int],
    level: int,
    tile: int,
    batch_size: int,
    max_tiles: Optional[int],
    outdir: Path,
):
    header = Path(header_path)
    sid    = header.stem
    out_p  = outdir / f"{sid}.h5"

    # resume-safe
    if out_p.exists():
        if is_valid_h5(out_p):
            return f"skip  {sid}"
        warnings.warn(f"Corrupted {out_p}, rebuilding…")
        out_p.unlink(missing_ok=True)

    # device
    pid   = os.getpid()
    gpu   = gpu_ids[pid % len(gpu_ids)] if gpu_ids else None
    device = torch.device(f"cuda:{gpu}" if gpu is not None else "cpu")

    # encoder & transform (lazy build per process)
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device).half()

    img_size = getattr(model.patch_embed, "img_size", 224)
    tfm = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # open slide
    try:
        slide = openslide.OpenSlide(str(header))
    except Exception as e:
        return f"fail  {sid}: {e}"

    W, H = slide.level_dimensions[level]
    feats, coords = [], []
    buf_imgs, buf_xy = [], []

    for y in range(0, H, tile):
        for x in range(0, W, tile):
            img = slide.read_region((x, y), level, (tile, tile)).convert("RGB")
            if np.mean(img) > 240:
                continue
            buf_imgs.append(tfm(img))
            buf_xy.append((x, y))

            if len(buf_imgs) >= batch_size:
                feats.append(infer_batch(buf_imgs, model, device))
                coords.extend(buf_xy)
                buf_imgs, buf_xy = [], []
            if max_tiles and len(coords) >= max_tiles:
                break
        if max_tiles and len(coords) >= max_tiles:
            break

    if buf_imgs:
        feats.append(infer_batch(buf_imgs, model, device))
        coords.extend(buf_xy)

    if not feats:
        return f"empty {sid}"

    feats  = np.vstack(feats)[: max_tiles]
    coords = np.asarray(coords, np.int32)[: max_tiles]

    with h5py.File(out_p, "w") as h5:
        h5.create_dataset("features", data=feats, compression="gzip")
        h5.create_dataset("coords",   data=coords, compression="gzip")
    return f"done  {sid}: {coords.shape[0]} tiles"

# ---------------------------------------------------------------------
# 4. Discover headers
# ---------------------------------------------------------------------

def collect_headers(root: Path) -> List[str]:
    pats = ["*.mrxs", "Index.dat", "index.dat", "*.xml"]
    hdrs = []
    for p in pats:
        hdrs.extend(root.rglob(p))
    return list(dict.fromkeys(map(str, hdrs)))

# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------

def main():
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--tile",  type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu_ids", default="0")
    ap.add_argument("--model_name", default="hf_hub:prov-gigapath/prov-gigapath")
    ap.add_argument("--max_tiles", type=int, default=None)
    args = ap.parse_args()

    root   = Path(args.root).expanduser()
    outdir = Path(args.out).expanduser(); outdir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(i) for i in args.gpu_ids.split(',')] if torch.cuda.is_available() and args.gpu_ids else []
    headers = collect_headers(root)
    print(f"Found {len(headers)} slide headers under '{root}'.")

    worker = partial(process_slide,
                     model_name=args.model_name,
                     gpu_ids=gpu_ids,
                     level=args.level,
                     tile=args.tile,
                     batch_size=args.batch_size,
                     max_tiles=args.max_tiles,
                     outdir=outdir)

    num_workers = min(args.workers, len(gpu_ids) or args.workers)
    print(f"Starting {num_workers} worker(s)")
    with mp.get_context("spawn").Pool(num_workers) as pool:
        for msg in tqdm.tqdm(pool.imap_unordered(worker, headers), total=len(headers)):
            print(msg)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="openslide")
    main()