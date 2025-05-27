#!/usr/bin/env python
"""
1. 递归 rglob 搜索 **/Index.dat 或 **/index.dat
2. 把 parent 目录认作一个 slide 目录（无需再检查 Data*.dat）
3. 用 OpenSlide 打开该目录，切 tile → Prov-GigaPath tile-encoder
4. 写成  <out_dir>/<folder_name>.h5   含 features+coords
"""
import argparse, multiprocessing as mp, os
from pathlib import Path
import h5py, numpy as np, torch, openslide, tqdm, timm
import torchvision.transforms as T

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument('--root',  required=True, help='根目录，里面套 N 层子文件夹')
p.add_argument('--out',   required=True, help='输出 h5 的文件夹')
p.add_argument('--tile',  type=int, default=256)
p.add_argument('--level', type=int, default=0)
p.add_argument('--max_tiles', type=int, default=100000)
p.add_argument('--workers',   type=int, default=max(1, mp.cpu_count()//2))
args = p.parse_args()

ROOT   = Path(args.root).expanduser()
OUTDIR = Path(args.out).expanduser(); OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- 模型 ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
model.eval().half().to(device)
tfm = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# ---------- 单 slide 处理 ----------
def run_slide(idx_path: str):
    slide_dir = Path(idx_path).parent
    sid = slide_dir.name
    out_p = OUTDIR / f"{sid}.h5"
    if out_p.exists():
        return f"skip  {sid}"
    try:
        slide = openslide.OpenSlide(str(slide_dir))  # 目录直接传给 OpenSlide
    except Exception as e:
        return f"fail  {sid}: {e}"
    W, H = slide.level_dimensions[args.level]
    feats, coords = [], []; ts = args.tile
    for y in range(0, H, ts):
        for x in range(0, W, ts):
            img = slide.read_region((x, y), args.level, (ts, ts)).convert("RGB")
            if np.mean(img) > 235:   # 过滤近纯白背景
                continue
            t = tfm(img).unsqueeze(0).half().to(device)
            with torch.no_grad():
                f = model(t).cpu().numpy()[0]
            feats.append(f); coords.append((x, y))
            if len(feats) >= args.max_tiles: break
        if len(feats) >= args.max_tiles: break
    feats  = np.asarray(feats,  dtype=np.float16)
    coords = np.asarray(coords, dtype=np.int32)
    with h5py.File(out_p, 'w') as h:
        h.create_dataset("features", data=feats,  compression='gzip')
        h.create_dataset("coords",   data=coords, compression='gzip')
    return f"done  {sid}: {len(feats)} tiles"

# ---------- 收集 Index.dat ----------
index_files = list(ROOT.rglob('Index.dat')) + list(ROOT.rglob('index.dat'))
print(f"Found {len(index_files)} Index.dat files under {ROOT}")

# ---------- 多进程跑 ----------
with mp.get_context("spawn").Pool(args.workers) as pool:
    for msg in tqdm.tqdm(pool.imap_unordered(run_slide, map(str, index_files)),
                         total=len(index_files)):
        print(msg)
