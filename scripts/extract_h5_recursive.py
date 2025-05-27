#!/usr/bin/env python
"""
Recursively scan --root for slide headers (MRXS / Index.dat / *.xml),
cut tiles, batch-encode with a ViT-tiny backbone, and save HDF5.

↺ 2025-05-28  增强断点续跑：重启后已完成文件自动跳过，损坏文件会重算。
"""
import argparse, multiprocessing as mp, os, warnings
from pathlib import Path
import h5py, numpy as np, torch, openslide, tqdm, timm
import torchvision.transforms as T
from functools import partial


# ------------------------- demo util ---------------------------------
def demo_open_slide(header_path: Path, tile: int, level: int):
    """Demo: open a single slide header, read one tile and display its info."""
    try:
        slide = openslide.OpenSlide(str(header_path))
    except Exception as e:
        print(f"Failed to open slide: {e}")
        return
    W, H = slide.level_dimensions[level]
    print(f"Slide dim (level {level}): {W}×{H}")
    x, y = max(0, W // 2 - tile // 2), max(0, H // 2 - tile // 2)
    img = slide.read_region((x, y), level, (tile, tile)).convert("RGB")
    arr = np.asarray(img)
    print(f"Tile shape {arr.shape}, mean {arr.mean():.2f}")
    try:
        img.show()
    except Exception:
        pass


# ---------------------- main processing fn ---------------------------
def is_valid_h5(fp: Path) -> bool:
    """H5 是否完整有效（含非空 features / coords）"""
    try:
        with h5py.File(fp, "r") as h5:
            return (
                "features" in h5
                and "coords" in h5
                and h5["features"].shape[0] > 0
                and h5["coords"].shape[0] > 0
            )
    except Exception:
        return False


def process_slide(
    header_path: str,
    device,
    model,
    tfm,
    level,
    tile,
    max_tiles,
    batch_size,
    outdir,
):
    header = Path(header_path)
    sid = header.stem
    out_p = outdir / f"{sid}.h5"

    # ↺ 断点续跑：完整文件直接跳过；坏文件先删
    if out_p.exists():
        if is_valid_h5(out_p):
            return f"skip  {sid}"
        else:
            warnings.warn(f"Corrupted {out_p}, rebuilding…")
            try:
                out_p.unlink()  # 删除损坏文件
            except Exception:
                pass

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
            if np.mean(img) > 235:  # 跳过空白
                continue
            buf_imgs.append(tfm(img))
            buf_xy.append((x, y))

            if len(buf_imgs) >= batch_size:
                f = infer_batch(buf_imgs, device, model)
                feats.append(f)
                coords.extend(buf_xy)
                buf_imgs, buf_xy = [], []
            if len(coords) >= max_tiles:
                break
        if len(coords) >= max_tiles:
            break

    if buf_imgs and len(coords) < max_tiles:
        feats.append(infer_batch(buf_imgs, device, model))
        coords.extend(buf_xy)

    if not feats:
        return f"empty {sid}"

    feats = np.vstack(feats)[: max_tiles]
    coords = np.asarray(coords, np.int32)[: max_tiles]
    with h5py.File(out_p, "w") as h5:
        h5.create_dataset("features", data=feats, compression="gzip")
        h5.create_dataset("coords", data=coords, compression="gzip")
    return f"done  {sid}: {len(coords)} tiles"


def infer_batch(img_list, device, model):
    """batched GPU forward"""
    with torch.no_grad():
        return (
            model(torch.stack(img_list).half().to(device)).cpu().numpy()
        )


# ------------------------------ main ---------------------------------
def build_model(device):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
    model.eval().to(device).half()
    tfm = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, tfm


def collect_headers(root: Path):
    pats = ["*.mrxs", "Index.dat", "index.dat", "*.xml"]
    headers = []
    for p in pats:
        headers += root.rglob(p)
    return list(dict.fromkeys(map(str, headers)))  # 去重


def main():
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root to scan")
    ap.add_argument("--out", required=True, help="Output folder for H5")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--max_tiles", type=int, default=100000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--path", help="Single slide header for --demo")
    args = ap.parse_args()

    # demo mode --------------------------------------------------------
    if args.demo:
        if not args.path:
            ap.error("--demo 需要配合 --path")
        demo_open_slide(Path(args.path), args.tile, args.level)
        return

    root = Path(args.root).expanduser()
    outdir = Path(args.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tfm = build_model(device)

    headers = collect_headers(root)
    print(f"Found {len(headers)} slide headers under '{root}'")

    fn = partial(
        process_slide,
        device=device,
        model=model,
        tfm=tfm,
        level=args.level,
        tile=args.tile,
        max_tiles=args.max_tiles,
        batch_size=args.batch_size,
        outdir=outdir,
    )

    # 运行 -------------------------------------------------------------
    if args.workers > 0:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for msg in tqdm.tqdm(pool.imap_unordered(fn, headers), total=len(headers)):
                print(msg)
    else:
        for h in tqdm.tqdm(headers):
            print(fn(h))


if __name__ == "__main__":
    # 避免 openslide 多进程时输出警告
    warnings.filterwarnings("ignore", category=UserWarning, module="openslide")
    main()
