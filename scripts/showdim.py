# 文件：inspect_h5_full.py

import h5py
import glob
import os

# 要检查的目录
h5_dir = "../outputs/level2_tile128_h5_2560"

# 期望的 feature 维度
EXPECTED_DIM = 2560

correct = 0
incorrect = 0
bad_files = []

for path in glob.glob(os.path.join(h5_dir, "*.h5")):
    try:
        with h5py.File(path, "r") as f:
            if "features" not in f:
                print(f"[WARN] no 'features' dataset in {path}")
                incorrect += 1
                bad_files.append((path, None))
                continue

            shape = f["features"].shape
            # shape 应该是 (N_tiles, EXPECTED_DIM)
            if len(shape) == 2 and shape[1] == EXPECTED_DIM:
                correct += 1
            else:
                incorrect += 1
                bad_files.append((path, shape))
    except Exception as e:
        print(f"[ERROR] failed to read {path}: {e}")
        incorrect += 1
        bad_files.append((path, None))

print(f"\n== 检查结果 ==")
print(f"总文件数： {correct + incorrect}")
print(f"✔ 正确 ({EXPECTED_DIM} 维)： {correct}")
print(f"✘ 错误（非 {EXPECTED_DIM} 维 或 读取失败）： {incorrect}")

if bad_files:
    print("\n错误文件详情：")
    for p, shape in bad_files:
        print(f"  - {os.path.basename(p)}  →  features shape = {shape}")
