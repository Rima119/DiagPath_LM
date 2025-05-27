#!/usr/bin/env python
"""
Recursively scan --root for slide headers:
  • Index.dat / index.dat (folder-based MRXS)
  • *.mrxs files
Treat each as a slide, cut tiles, batch-encode with Prov-GigaPath,
and write HDF5. Supports both batch and demo mode.
"""
import argparse
import multiprocessing as mp
from pathlib import Path
import h5py
import numpy as np
import torch
import openslide
import tqdm
import timm
import torchvision.transforms as T
from functools import partial


def demo_open_slide(header_path: Path, tile: int, level: int):
    """Demo: open a single slide header, read one tile and display its info."""
    header = header_path
    print(f"Opening header: {header}")
    try:
        slide = openslide.OpenSlide(str(header))
    except Exception as e:
        print(f"Failed to open slide: {e}")
        return
    W, H = slide.level_dimensions[level]
    print(f"Slide dimensions (level {level}): {W}x{H}")
    x = max(0, W//2 - tile//2)
    y = max(0, H//2 - tile//2)
    img = slide.read_region((x, y), level, (tile, tile)).convert('RGB')
    arr = np.array(img)
    print(f"Tile shape: {arr.shape}, mean pixel: {arr.mean():.2f}")
    try:
        img.show()
    except Exception:
        pass


def process_slide(header_path: str, device, model, tfm, level, tile, max_tiles, batch_size, outdir):
    header = Path(header_path)
    sid = header.stem
    out_p = outdir / f"{sid}.h5"
    if out_p.exists():
        return f"skip  {sid}"
    try:
        slide = openslide.OpenSlide(str(header))
    except Exception as e:
        return f"fail  {sid}: {e}"

    W, H = slide.level_dimensions[level]
    all_feats = []
    all_coords = []
    batch_imgs = []
    batch_coords = []

    # slide through tiles
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            img = slide.read_region((x, y), level, (tile, tile)).convert('RGB')
            if np.mean(img) > 235:
                continue
            t = tfm(img)
            batch_imgs.append(t)
            batch_coords.append((x, y))
            # if batch full or last
            if len(batch_imgs) >= batch_size:
                batch_tensor = torch.stack(batch_imgs).half().to(device)
                with torch.no_grad():
                    feats = model(batch_tensor).cpu().numpy()
                all_feats.append(feats)
                all_coords.extend(batch_coords)
                batch_imgs.clear()
                batch_coords.clear()
            if len(all_coords) >= max_tiles:
                break
        if len(all_coords) >= max_tiles:
            break

    # process leftover
    if batch_imgs and len(all_coords) < max_tiles:
        batch_tensor = torch.stack(batch_imgs).half().to(device)
        with torch.no_grad():
            feats = model(batch_tensor).cpu().numpy()
        all_feats.append(feats)
        all_coords.extend(batch_coords)

    # concatenate and limit
    feats_arr = np.vstack(all_feats)[:max_tiles]
    coords_arr = np.array(all_coords, dtype=np.int32)[:max_tiles]

    if feats_arr.size == 0:
        return f"empty {sid}: no tiles"

    with h5py.File(out_p, 'w') as h5:
        h5.create_dataset('features', data=feats_arr, compression='gzip')
        h5.create_dataset('coords',   data=coords_arr, compression='gzip')
    return f"done  {sid}: {coords_arr.shape[0]} tiles"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',   help='Root directory to scan')
    parser.add_argument('--out',    help='Output HDF5 folder')
    parser.add_argument('--tile',   type=int, default=256, help='Tile size')
    parser.add_argument('--level',  type=int, default=0, help='OpenSlide level')
    parser.add_argument('--max_tiles', type=int, default=100000, help='Max tiles per slide')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for GPU inference')
    parser.add_argument('--workers', type=int, default=0, help='Parallel workers; 0 for serial')
    parser.add_argument('--demo',   action='store_true', help='Demo single slide open')
    parser.add_argument('--path',   type=str, help='Path to single slide header for demo')
    args = parser.parse_args()

    if args.demo:
        if not args.path:
            print("Please provide --path for demo mode.")
            return
        demo_open_slide(Path(args.path), args.tile, args.level)
        return

    if not args.root or not args.out:
        print("Please provide --root and --out for batch mode.")
        return

    ROOT = Path(args.root).expanduser()
    OUTDIR = Path(args.out).expanduser()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
    model.eval().to(device).half()
    tfm = T.Compose([
        T.Resize(args.tile, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # collect slide headers
    headers = []
    headers += list(map(str, ROOT.rglob('*.mrxs')))
    headers += list(map(str, ROOT.rglob('Index.dat')))
    headers += list(map(str, ROOT.rglob('index.dat')))
    headers += list(map(str, ROOT.rglob('*.xml')))
    headers = list(dict.fromkeys(headers))
    print(f"Found {len(headers)} slide headers under '{ROOT}'")

    process_fn = partial(
        process_slide,
        device=device,
        model=model,
        tfm=tfm,
        level=args.level,
        tile=args.tile,
        max_tiles=args.max_tiles,
        batch_size=args.batch_size,
        outdir=OUTDIR,
    )
    if args.workers > 0:
        with mp.get_context('spawn').Pool(args.workers) as pool:
            for msg in tqdm.tqdm(pool.imap_unordered(process_fn, headers), total=len(headers)):
                print(msg)
    else:
        for hdr in tqdm.tqdm(headers):
            print(process_fn(hdr))


if __name__ == "__main__":
    main()
