#!/usr/bin/env python3
# coding: utf-8
"""
encode_slides_2560.py -- re-revised 2025-06-13

Recursively scan for slide container files, cut tiles, encode with
`prov-gigapath/prov-gigapath-2560` 2560-d model, and write one HDF5 per
slide with:

    - features (N × 2560) float16
    - coords   (N × 2)    int32

Safe-resume: skip only if the .h5 exists and passes integrity check.

Usage:

    export HUGGINGFACEHUB_API_TOKEN="<YOUR_TOKEN_HERE>"
    python encode_slides_2560.py \
      --root /media/mxl/Extreme\ Pro \
      --out outputs/level2_tile128_h5_2560 \
      --model_name hf_hub:prov-gigapath/prov-gigapath-2560 \
      --level 2 \
      --tile 128 \
      --batch_size 64 \
      --workers 4 \
      --gpu_ids 0,1,2,3 \
      --max_tiles 1024
"""
import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import os, warnings, argparse, multiprocessing as mp
from pathlib import Path
from functools import partial
import h5py, numpy as np, timm, torch, torchvision.transforms as T, tqdm
import openslide
from openslide import OpenSlideError

# ---------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------
EXPECTED_DIM = 2560

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def is_valid_h5(fp: Path) -> bool:
    """True if fp exists and has correct datasets with expected dims."""
    try:
        with h5py.File(fp, 'r') as h5:
            if 'features' not in h5 or 'coords' not in h5:
                return False
            feats = h5['features']
            coords = h5['coords']
            return (
                feats.ndim == 2 and coords.ndim == 2 and
                feats.shape[0] > 0 and feats.shape[1] == EXPECTED_DIM and
                coords.shape[0] == feats.shape[0] and coords.shape[1] == 2
            )
    except Exception:
        return False


def infer_batch(imgs, model, device):
    """Encode a batch of preprocessed tiles into 2560-d embeddings."""
    with torch.no_grad():
        batch = torch.stack(imgs).to(device).half()
        out = model(batch)
        return out.cpu().numpy().astype('float16')

# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------

def process_slide(
    slide_path,
    model_name,
    gpu_ids,
    level,
    tile,
    batch_size,
    max_tiles,
    outdir
):
    slide_path = Path(slide_path)
    sid = slide_path.stem
    out_p = outdir / f"{sid}.h5"
    try:
        # skip existing valid
        if out_p.exists() and is_valid_h5(out_p):
            return f"skip  {sid}"
        if out_p.exists():
            out_p.unlink()
        # select device round-robin
        pid = os.getpid()
        gpu = gpu_ids[pid % len(gpu_ids)] if gpu_ids else None
        device = torch.device(f"cuda:{gpu}" if gpu is not None else "cpu")
        # load model + prep
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model.eval().to(device).half()
        img_size = model.patch_embed.img_size if hasattr(model, 'patch_embed') else 224
        tfm = T.Compose([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        # open slide
        try:
            slide = openslide.OpenSlide(str(slide_path))
        except OpenSlideError as e:
            return f"fail  {sid}: {e}"
        if level >= slide.level_count:
            slide.close()
            return f"skip  {sid}: only {slide.level_count} levels"
        W, H = slide.level_dimensions[level]
        feats_list, coords_list = [], []
        buf_imgs, buf_xy = [], []
        # tile loop
        for y in range(0, H, tile):
            for x in range(0, W, tile):
                try:
                    img = slide.read_region((x,y), level, (tile,tile)).convert('RGB')
                except Exception:
                    continue
                if np.mean(img) > 240:
                    continue
                buf_imgs.append(tfm(img))
                buf_xy.append((x,y))
                if len(buf_imgs) >= batch_size:
                    feats_list.append(infer_batch(buf_imgs, model, device))
                    coords_list.extend(buf_xy)
                    buf_imgs, buf_xy = [], []
                if max_tiles is not None and len(coords_list) >= max_tiles:
                    break
            if max_tiles is not None and len(coords_list) >= max_tiles:
                break
        if buf_imgs:
            feats_list.append(infer_batch(buf_imgs, model, device))
            coords_list.extend(buf_xy)
        slide.close()
        if not feats_list:
            return f"empty {sid}"
        feats = np.vstack(feats_list)
        coords = np.asarray(coords_list, np.int32)
        if max_tiles is not None:
            feats = feats[:max_tiles]
            coords = coords[:max_tiles]
        outdir.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_p, 'w') as h5:
            h5.create_dataset('features', data=feats, compression='gzip')
            h5.create_dataset('coords',   data=coords, compression='gzip')
        return f"done  {sid}: {coords.shape[0]} tiles"
    except Exception as e:
        return f"error {sid}: {type(e).__name__}: {e}"

# ---------------------------------------------------------------------
# Slide discovery
# ---------------------------------------------------------------------

def collect_headers(root: Path):
    exts = {'.mrxs','.svs','.tif','.tiff','.ndpi','.vms','.vmu','.scn','.isyntax','.bif'}
    return [str(p) for p in root.rglob('*') if p.suffix.lower() in exts]

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    mp.set_start_method('spawn', force=True)
    ap = argparse.ArgumentParser(description='Encode slides with prov-gigapath-2560.')
    ap.add_argument('--root',       required=True, help='Slide root dir')
    ap.add_argument('--out',        default='outputs/level2_tile128_h5_2560')
    ap.add_argument('--model_name', default='hf_hub:prov-gigapath/prov-gigapath-2560')
    ap.add_argument('--level',      type=int,   default=2)
    ap.add_argument('--tile',       type=int,   default=128)
    ap.add_argument('--batch_size', type=int,   default=64)
    ap.add_argument('--workers',    type=int,   default=4)
    ap.add_argument('--gpu_ids',    default='0')
    ap.add_argument('--max_tiles',  type=int,   default=None)
    args = ap.parse_args()

    # ensure HF auth
    token = os.environ.get('HUGGINGFACEHUB_API_TOKEN') or os.environ.get('HF_TOKEN')
    if not token:
        print('Error: set HUGGINGFACEHUB_API_TOKEN to access the 2560-d model'); exit(1)
    root = Path(args.root).expanduser()
    outdir = Path(args.out).expanduser()
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')] if torch.cuda.is_available() else []

    headers = collect_headers(root)
    processed = {p.stem for p in outdir.glob('*.h5') if is_valid_h5(p)}
    to_process = [h for h in headers if Path(h).stem not in processed]

    print(f"🖼️  Found {len(headers)} slides under {root}")
    print(f"▶️  Encoding {len(to_process)} / {len(headers)} slides")

    worker = partial(
        process_slide,
        model_name=args.model_name,
        gpu_ids=gpu_ids,
        level=args.level,
        tile=args.tile,
        batch_size=args.batch_size,
        max_tiles=args.max_tiles,
        outdir=outdir,
    )
    num_workers = min(args.workers, len(to_process))
    print(f"🚀 Launching {num_workers} worker(s)")

    if num_workers < 1:
        for path in tqdm.tqdm(to_process): print(worker(path))
    else:
        with mp.get_context('spawn').Pool(num_workers) as pool:
            for msg in tqdm.tqdm(pool.imap_unordered(worker, to_process), total=len(to_process)):
                print(msg)

if __name__ == '__main__':
    warnings.filterwarnings('ignore', module='openslide')
    main()
