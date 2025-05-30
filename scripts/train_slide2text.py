#!/usr/bin/env python
# coding: utf-8
"""
scripts/train_slide2text.py

Train a GPT-based slide-to-text model using H5 features with prefix mapping,
then generate outputs on the training set and save them in JSON format (no immuno).

支持 JSON 数组 或 JSONL 作为报告输入；支持可配置的特征维度与 H5 目录。
在推理阶段可加高斯噪声、调节 temperature/top_p/top_k 采样。
"""
import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import json
import h5py
import torch
import argparse
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

MAX_LEN = 512

def load_reports(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    try:
        arr = json.loads(text)
        if not isinstance(arr, list):
            raise ValueError
    except Exception:
        arr = []
        for idx, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ Skip invalid JSONL at line {idx}")
                continue
            arr.append(obj)
    pattern = re.compile(r"^S?(\d{4}-\d{6})")
    id2text = {}
    for obj in arr:
        sid = obj.get("slide_id", "")
        m = pattern.match(sid)
        if not m:
            continue
        base = m.group(1)
        diag = obj.get("diagnosis", "").strip()
        fin  = obj.get("findings", "").strip()
        if diag and fin:
            id2text[base] = {"diagnosis": diag, "findings": fin}
    return id2text

def build_records(slide_dir: Path, id2text: dict):
    recs = []
    for h5_path in slide_dir.glob("*.h5"):
        m = re.match(r"^S?(\d{4}-\d{6})", h5_path.stem)
        if not m:
            continue
        base = m.group(1)
        info = id2text.get(base)
        if info:
            recs.append({
                "slide_id": base,
                "feat_path": str(h5_path),
                **info
            })
    return recs

def load_features(ex):
    with h5py.File(ex["feat_path"], "r") as h5:
        feats = h5["features"][:]
    mf = feats.mean(axis=0)
    return {
        "feat": mf.astype("float32"),
        "slide_id": ex["slide_id"],
        "diagnosis": ex["diagnosis"],
        "findings": ex["findings"]
    }

def collate_fn(batch, tokenizer, max_len):
    feats = torch.tensor([b["feat"] for b in batch], dtype=torch.float32)
    texts = [f"{b['diagnosis']}。{b['findings']}" for b in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask
    labels = input_ids.clone()
    # 在 prefix 位置填 -100，令它不计入 loss
    prefix = torch.full((labels.size(0), 1), -100, dtype=labels.dtype)
    labels = torch.cat([prefix, labels], dim=1)
    return {
        "feat": feats,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

class Slide2Text(torch.nn.Module):
    def __init__(self, lm, mapper):
        super().__init__()
        self.lm     = lm
        self.mapper = mapper
        self.emb    = lm.get_input_embeddings()

    def forward(self, feat, input_ids=None, attention_mask=None, labels=None):
        # 映射前缀向量
        pref = self.mapper(feat).unsqueeze(1)           # (B,1,D)
        # 取原始 token 嵌入
        emb_tokens = self.emb(input_ids)                # (B,L,D)
        emb = torch.cat([pref, emb_tokens], dim=1)      # (B,L+1,D)
        if attention_mask is not None:
            one = torch.ones((attention_mask.size(0),1),
                             device=attention_mask.device)
            attention_mask = torch.cat([one, attention_mask], dim=1)
        return self.lm(inputs_embeds=emb,
                       attention_mask=attention_mask,
                       labels=labels)

def main():
    parser = argparse.ArgumentParser(description="Train slide2text model and save outputs")
    parser.add_argument("--slide_dir",      type=str,
                        default="outputs/level2_tile128_h5",
                        help="Folder containing .h5 feature files")
    parser.add_argument("--feat_dim",       type=int, default=1536,
                        help="Dimensionality of each H5 feature vector")
    parser.add_argument("--reports",        type=str,
                        default="data/HCC_translated.json",
                        help="Path to JSON/JSONL reports file")
    parser.add_argument("--model_name",     type=str, default="microsoft/BioGPT-Large")
    parser.add_argument("--epochs",         type=int, default=3)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--learning_rate",  type=float, default=2e-5)
    parser.add_argument("--output_dir",     type=str, default="outputs/slide2text")
    parser.add_argument("--fp16",           action="store_true")
    parser.add_argument("--bf16",           action="store_true")
    parser.add_argument("--save_strategy",  type=str, default="epoch")
    parser.add_argument("--eval_strategy",  type=str, default="no")
    parser.add_argument("--logging_steps",  type=int, default=50)
    # 采样与噪声
    parser.add_argument("--noise_std",      type=float, default=0.0,
                        help="Gaussian noise σ added to prefix embedding at generation")
    parser.add_argument("--temp",           type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p",          type=float, default=0.9,
                        help="Nucleus sampling p")
    parser.add_argument("--top_k",          type=int,   default=50,
                        help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty")
    args = parser.parse_args()

    SLIDE_DIR = Path(args.slide_dir)
    REPORTS   = Path(args.reports)
    FEAT_DIM  = args.feat_dim

    # 1) 读报告、匹配 slide_id
    id2text = load_reports(REPORTS)
    recs    = build_records(SLIDE_DIR, id2text)
    print(f"🔗 matched {len(recs)} slide-report pairs")
    if not recs:
        print("❌ No matched pairs. Check slide_dir & reports.")
        return

    # 2) 构造 Dataset
    ds = Dataset.from_list(recs)
    ds = ds.map(load_features, new_fingerprint="feat_loaded")
    ds = ds.remove_columns(["feat_path"])

    # 3) Tokenizer & LM
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=MAX_LEN,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_name)
    prefix_dim = config.hidden_size

    lm     = AutoModelForCausalLM.from_pretrained(args.model_name)
    mapper = torch.nn.Linear(FEAT_DIM, prefix_dim, bias=False)
    model  = Slide2Text(lm, mapper)
    lm.gradient_checkpointing_enable()

    # —— **不再冻结任何参数** ——

    # 4) 训练参数 & Trainer
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation,
        fp16=args.fp16,
        bf16=args.bf16,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.eval_strategy,
        logging_steps=args.logging_steps,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        save_safetensors=False,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=lambda batch: collate_fn(batch, tokenizer, MAX_LEN),
    )
    trainer.train()

    # 5) 在训练集上推理并保存，加入噪声 & 采样参数
    model.eval()
    outputs = []
    for ex in recs:
        feat = load_features(ex)["feat"]
        f_t  = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(train_args.device)
        with torch.no_grad():
            prefix_emb = mapper(f_t).unsqueeze(1)
            if args.noise_std > 0:
                noise = torch.randn_like(prefix_emb) * args.noise_std
                prefix_emb = prefix_emb + noise
            gen_ids = lm.generate(
                inputs_embeds=prefix_emb,
                max_new_tokens=MAX_LEN,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=args.temp,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        outputs.append({"slide_id": ex["slide_id"], "report": text})

    out_path = Path(args.output_dir) / "train_outputs.json"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"✅ Outputs saved to {out_path}")

if __name__ == "__main__":
    main()
