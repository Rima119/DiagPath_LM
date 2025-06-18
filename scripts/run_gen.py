#!/usr/bin/env python
# coding: utf-8
"""
生成最终（200-epoch）模型的预测
"""

import os, sys, types, argparse, json, importlib, torch, inspect
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from train_slide2text_prism import Slide2TextDataset, generate_for_dataset
from huggingface_hub import snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN", "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")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)             # outputs/…/final
    ap.add_argument("--slide_dir", required=True)
    ap.add_argument("--reports",   required=True)
    ap.add_argument("--out",       default="final_predictions.jsonl")
    ap.add_argument("--max_tiles", type=int, default=128)
    ap.add_argument("--orig_repo", default="paige-ai/Prism")  # 源码仓库
    args = ap.parse_args()

    # ─── 1. 注入 PRISM 源码 ──────────────────────────────────
    repo = snapshot_download(args.orig_repo, local_files_only=True, token=HF_TOKEN)
    prism_pkg = types.ModuleType("prism_local"); prism_pkg.__path__ = [repo]
    sys.modules["prism_local"] = prism_pkg; sys.path.insert(0, repo)
    mod   = importlib.import_module("prism_local.modeling_prism")
    Prism = next(getattr(mod, n) for n in ("PrismForCausalLM","Prism","PrismModel") if hasattr(mod,n))

    # 必须把 has_no_defaults_at_init 打开，再读 config
    cfg_mod = importlib.import_module("prism_local.configuring_prism")
    cfg_mod.PrismConfig.has_no_defaults_at_init = True
    with open(os.path.join(args.model_dir, "config.json"), encoding="utf-8") as f:
        cfg = cfg_mod.PrismConfig.from_dict(json.load(f))
        cfg.has_no_defaults_at_init = True

    # ─── 2. 构建空模型并加载权重 ───────────────────────────────
    print("[INFO] building model && loading weights …")
    model = Prism(cfg).cuda().eval()        # 不再 from_pretrained！
    sd    = torch.load(os.path.join(args.model_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    del sd

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)

    # ─── 3. 数据集 & 生成 ────────────────────────────────────
    ds = Slide2TextDataset(args.slide_dir, args.reports, tok, args.max_tiles)
    vision_arg = next(p for p in inspect.signature(model.forward).parameters if p != "input_ids")

    generate_for_dataset(model, tok, ds, vision_arg, args.out)
    print("✅  Predictions saved →", args.out)

if __name__ == "__main__":
    main()
