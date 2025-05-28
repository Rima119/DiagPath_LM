import h5py
import numpy as np
from pathlib import Path

# 1) 把每个 H5 的 mean-pool 特征读出来
slide_dir = Path("data/ultralowres_h5")
feats = []
for p in slide_dir.glob("*.h5"):
    with h5py.File(p, "r") as h5:
        arr = h5["features"][:]        # (N_tiles, FEAT_DIM)
        feats.append(arr.mean(axis=0))  # (FEAT_DIM,)

feats = np.stack(feats)  # (num_slides, FEAT_DIM)

# 2) 查看每个维度的方差
var = feats.var(axis=0)
print(f"Per-dim variance → mean: {var.mean():.4f}, min: {var.min():.4f}, max: {var.max():.4f}")

# 3) 计算两两余弦相似度
normed = feats / np.linalg.norm(feats, axis=1, keepdims=True)
sims = normed @ normed.T      # (num_slides, num_slides)
# 只取上三角（不含对角线）：
i, j = np.triu_indices(sims.shape[0], k=1)
cosines = sims[i, j]
print(f"Cosine sim → mean: {cosines.mean():.4f}, std: {cosines.std():.4f}")

# 4) 如果想更直观，可以画个 histogram（需 matplotlib）
import matplotlib.pyplot as plt
plt.hist(var, bins=50)
plt.title("Feature Variance per Dimension")
plt.show()

plt.hist(cosines, bins=50)
plt.title("Pairwise Cosine Similarity")
plt.show()
