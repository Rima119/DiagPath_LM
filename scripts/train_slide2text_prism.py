#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Slide ➜ Report · PRISM finetune
· 支持 --no_amp (全 FP32)
· epoch01 / epochLast 推断 & 每个 epoch 保存 ckpt
"""

import os, sys, types, importlib, argparse, json, glob, inspect, h5py, pathlib, logging
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# ───────── HF 离线镜像 ─────────
os.environ["HF_TOKEN"]                 = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["HF_ENDPOINT"]              = "https://hf-mirror.com"
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

# ╭──────────────── Dataset ───────────────╮
def norm_id(fn: str) -> str:
    stem = pathlib.Path(fn).stem
    if stem.lower().startswith("s"):
        stem = stem[1:]
    return "-".join(stem.split("-")[:2])

class Slide2TextDataset(Dataset):
    def __init__(self, slide_dir, reports_json, tok, max_tiles=128):
        with open(reports_json, encoding="utf-8") as f:
            rep = {norm_id(d["slide_id"]): f'{d["diagnosis"]} {d.get("findings","")}'.strip()
                   for d in json.load(f)}
        self.items = [(p, rep[norm_id(p)]) for p in glob.glob(f"{slide_dir}/*.h5")
                      if norm_id(p) in rep]
        self.tok, self.max_tiles = tok, max_tiles
        self.bos = tok.bos_token_id or tok.cls_token_id or 0

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, txt = self.items[idx]
        with h5py.File(path) as h5:
            feat = h5["features"][:].astype(np.float32)
        real_n = min(len(feat), self.max_tiles)
        pad_n  = self.max_tiles - real_n
        feat_padded = np.vstack([feat[:self.max_tiles],
                                 np.zeros((pad_n, feat.shape[1]), np.float32)])
        mask = np.zeros(self.max_tiles, np.bool_); mask[:real_n] = True
        ids = torch.cat([
            torch.tensor([self.bos]),
            self.tok(txt, add_special_tokens=False, truncation=True,
                     max_length=511, padding="max_length",
                     return_tensors="pt").input_ids.squeeze(0)
        ])
        return {"features": torch.from_numpy(feat_padded),
                "mask":     torch.from_numpy(mask),
                "input_ids": ids,
                "slide": pathlib.Path(path).name}

    @staticmethod
    def collate(batch):
        return {"features": torch.stack([b["features"] for b in batch]),
                "mask":     torch.stack([b["mask"]     for b in batch]),
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "meta": [b["slide"] for b in batch]}
# ╰─────────────────────────────────────────╯

# ───────── helper ─────────
def extract_logits(x):
    if isinstance(x, tuple):
        return x[0]
    if hasattr(x, "logits"):
        return x.logits
    return x["logits"]

def get_kv(out):
    if isinstance(out, tuple) and len(out) > 1:
        return out[1]
    if isinstance(out, dict):
        for k in ("past_key_values", "key_value_states", "kv", "image_latents"):
            if k in out and out[k] is not None:
                return out[k]
    if hasattr(out, "past_key_values"):
        return out.past_key_values
    return None

def get_vocab_size(decoder):
    for p in decoder.parameters():
        if p.dim() == 2:
            return p.size(0)
    raise RuntimeError("embedding matrix not found")

def save_model(model, tok, path):
    os.makedirs(path, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
               os.path.join(path, "pytorch_model.bin"))
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, ensure_ascii=False, indent=2)
    tok.save_pretrained(path)

def generate_for_dataset(model, tok, dl, out_path, *,
                         amp=True, eos_id=0, bos_id=0,
                         vision_arg, mask_arg):
    """推断并保存 jsonl"""
    gen_cfg = dict(max_new_tokens=128, eos_token_id=eos_id, bos_token_id=bos_id)
    model.eval()
    with torch.no_grad(), open(out_path, "w", encoding="utf-8") as fw, \
         torch.cuda.amp.autocast(enabled=amp):
        for b in tqdm(dl, total=len(dl), desc="⤬ generating"):
            feats = b["features"].cuda().to(model.dtype)
            masks = b["mask"].cuda()
            bos   = torch.full((feats.size(0),1), bos_id,
                               device=feats.device, dtype=torch.long)
            out   = model(**{vision_arg: feats, mask_arg: masks},
                          input_ids=bos, use_cache=True)
            kv    = get_kv(out)
            gens  = model.generate(key_value_states=kv, inputs=bos, **gen_cfg)
            dec   = tok.batch_decode(gens, skip_special_tokens=True)
            for sid, rp in zip(b["meta"], dec):
                fw.write(json.dumps({"slide": sid, "report_pred": rp},
                                    ensure_ascii=False) + "\n")
    model.train()

# ───────── CLI ─────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slide_dir", required=True)
    p.add_argument("--reports", required=True)
    p.add_argument("--model_name", default="paige-ai/Prism")
    p.add_argument("--tokenizer_name", default="microsoft/BioGPT-Large")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_tiles", type=int, default=128)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()

# ───────── main ─────────
def main():
    args = get_args(); os.makedirs(args.output_dir, exist_ok=True)

    repo = snapshot_download(args.model_name, local_files_only=True)
    prism_pkg = types.ModuleType("prism_local"); prism_pkg.__path__ = [repo]
    sys.modules["prism_local"] = prism_pkg; sys.path.insert(0, repo)

    cfg_mod = importlib.import_module("prism_local.configuring_prism")
    cfg_mod.PrismConfig.has_no_defaults_at_init = True
    with open(os.path.join(repo, "config.json")) as f:
        cfg = cfg_mod.PrismConfig.from_dict(json.load(f))

    tok   = AutoTokenizer.from_pretrained(args.tokenizer_name, local_files_only=True)
    Prism = importlib.import_module("prism_local.modeling_prism").Prism
    model = Prism.from_pretrained(repo, config=cfg,
                                  trust_remote_code=True,
                                  local_files_only=True).cuda()
    if args.no_amp: model = model.float()

    sig = inspect.signature(model.forward).parameters
    vision_arg = next(p for p in sig if p.startswith("tile") and "embed" in p)
    mask_arg   = next(p for p in sig if p.startswith("tile") and "mask"  in p)
    print(f"[INFO] vision_arg = '{vision_arg}', mask_arg = '{mask_arg}'")

    ds = Slide2TextDataset(args.slide_dir, args.reports, tok, args.max_tiles)
    print(f"[DATA] matched {len(ds)} / {len(glob.glob(f'{args.slide_dir}/*.h5'))} slides.")
    for itm in [ds[i] for i in range(min(3, len(ds)))]:
        print(f" ├─ {itm['slide']:<22}: {tok.decode(itm['input_ids'][:40])}...")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=Slide2TextDataset.collate)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, foreach=False)
    total = len(dl) * args.epochs // args.gradient_accumulation
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=total)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    v_model = get_vocab_size(model.text_decoder)
    unk_id  = tok.unk_token_id or 0
    pad_id  = tok.pad_token_id if tok.pad_token_id is not None else -100

    # —— epoch01 baseline ——
    ep_dir = f"{args.output_dir}/epoch01"
    save_model(model, tok, f"{ep_dir}/model")
    generate_for_dataset(model, tok, dl, f"{ep_dir}/preds.jsonl",
                         amp=not args.no_amp, eos_id=tok.eos_token_id or 0,
                         bos_id=tok.bos_token_id or 0,
                         vision_arg=vision_arg, mask_arg=mask_arg)

    # —— training ——
    for ep in range(1, args.epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {ep}/{args.epochs}")
        for i, batch in enumerate(pbar, 1):
            feats = batch["features"].cuda().to(model.dtype)
            masks = batch["mask"].cuda()
            ids   = batch["input_ids"].cuda()
            ids_in = ids.clone(); ids_in[ids_in >= v_model] = unk_id
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                out = model(**{vision_arg: feats, mask_arg: masks},
                            input_ids=ids_in, use_cache=False)
                logits = extract_logits(out)
                loss = F.cross_entropy(
                    logits.reshape(-1, v_model),
                    ids_in.reshape(-1),
                    ignore_index=pad_id
                )
            scaler.scale(loss).backward()
            if i % args.gradient_accumulation == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(); sched.step()
            pbar.set_postfix(g_step=i, loss=f"{loss.item():.4f}")

        # —— save every epoch ——
        ep_dir = f"{args.output_dir}/epoch{str(ep+1).zfill(2)}"
        save_model(model, tok, f"{ep_dir}/model")

        # —— last epoch inference ——
        if ep == args.epochs:
            generate_for_dataset(model, tok, dl, f"{ep_dir}/preds.jsonl",
                                 amp=not args.no_amp, eos_id=tok.eos_token_id or 0,
                                 bos_id=tok.bos_token_id or 0,
                                 vision_arg=vision_arg, mask_arg=mask_arg)

    print("✅ All done, outputs in", args.output_dir)

# ───────── standard entry point ─────────
if __name__ == "__main__":
    main()
