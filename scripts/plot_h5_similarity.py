#!/usr/bin/env python3
# coding: utf-8
"""
plot_h5_similarity.py

Compute slide-level feature similarities for a directory of H5 files
and plot the distribution of pairwise cosine similarities.
"""
import os
import argparse
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_mean_feature(path: str) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        feats = f['features'][:]
    return feats.mean(axis=0)

def main():
    parser = argparse.ArgumentParser(
        description="Plot distribution of slide-to-slide feature similarities"
    )
    parser.add_argument(
        '--h5_dir', required=True,
        help="Directory containing .h5 files with a 'features' dataset"
    )
    parser.add_argument(
        '--max_pairs', type=int, default=100000,
        help="If total pairs > max_pairs, randomly sample this many"
    )
    parser.add_argument(
        '--bins', type=int, default=50,
        help="Number of histogram bins"
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help="Path to save the histogram image (PNG). If not set, display interactively."
    )
    args = parser.parse_args()

    # 1) collect files and load mean features
    files = [
        f for f in os.listdir(args.h5_dir)
        if f.lower().endswith('.h5')
    ]
    if not files:
        print(f"No .h5 files found in {args.h5_dir}")
        return

    print(f"Loading mean features for {len(files)} files...")
    means = []
    for fn in files:
        path = os.path.join(args.h5_dir, fn)
        means.append(load_mean_feature(path))
    means = np.vstack(means)

    # 2) normalize and compute full cosine similarity matrix
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    means_norm = means / (norms + 1e-12)
    sim_matrix = means_norm @ means_norm.T

    # 3) extract upper-triangle similarities
    n = sim_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    sims = sim_matrix[iu]

    # 4) sample if too many
    total_pairs = len(sims)
    if total_pairs > args.max_pairs:
        print(f"Sampling {args.max_pairs} / {total_pairs} pairs for plotting")
        sims = random.sample(list(sims), args.max_pairs)

    # 5) plot histogram
    plt.figure(figsize=(6,4))
    plt.hist(sims, bins=args.bins, edgecolor='black')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Slide‐to‐Slide Feature Similarity Distribution")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Histogram saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
