#!/usr/bin/env python
# coding: utf-8
"""
scripts/train_slide2text.py

Demo for training a GPT-based slide-to-text model using H5 features with prefix mapping,
and then generating outputs on the training set and saving them in original JSON format
(without immuno field).
Supports both JSON array and JSONL report formats as input.
"""
import os
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

# —— 用户配置 —— #
SLIDE_DIR = Path("./data/ultralowres_h5")
REPORTS   = Path("./data/HCC_translated.json")
FEAT_DIM  = 192
MAX_LEN   = 512
# —————————————— #

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
            recs.append({"slide_id": base, "feat_path": str(h5_path), **info})
    return recs


def load_features(ex):
    with h5py.File(ex["feat_path"], "r") as h5:
        feats = h5["features"][:]
    mf = feats.mean(axis=0)
    return {"feat": mf.astype("float32"), "slide_id": ex["slide_id"],
            "diagnosis": ex["diagnosis"], "findings": ex["findings"]}


def collate_fn(batch, tokenizer, max_len):
    feats = torch.tensor([b['feat'] for b in batch], dtype=torch.float32)
    texts = [f"{b['diagnosis']}。{b['findings']}" for b in batch]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask
    labels = input_ids.clone()
    prefix = torch.full((labels.size(0), 1), -100, dtype=labels.dtype)
    labels = torch.cat([prefix, labels], dim=1)
    return {
        'feat': feats,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Train slide2text model and save outputs")
    parser.add_argument('--model_name', type=str, default='gpt2-large')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--output_dir', type=str, default='outputs/slide2text')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--save_strategy', type=str, default='epoch')
    parser.add_argument('--eval_strategy', type=str, default='no')
    parser.add_argument('--logging_steps', type=int, default=50)
    args = parser.parse_args()

    # load and prepare data
    id2text = load_reports(REPORTS)
    recs    = build_records(SLIDE_DIR, id2text)
    print(f"🔗 matched {len(recs)} slide-report pairs")
    if not recs:
        print("❌ No matched pairs. Check REPORTS and slide_id formats.")
        return

    ds = Dataset.from_list(recs)
    ds = ds.map(load_features, new_fingerprint="feat_loaded")
    ds = ds.remove_columns(["feat_path"])

    # tokenizer and model setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=MAX_LEN, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_name)
    prefix_dim = config.hidden_size

    lm = AutoModelForCausalLM.from_pretrained(args.model_name)
    mapper = torch.nn.Linear(FEAT_DIM, prefix_dim, bias=False)
    class Slide2Text(torch.nn.Module):
        def __init__(self, lm, mapper):
            super().__init__()
            self.lm     = lm
            self.mapper = mapper
        def forward(self, feat, input_ids=None, attention_mask=None, labels=None):
            pref = self.mapper(feat).unsqueeze(1)
            emb  = self.lm.transformer.wte(input_ids)
            emb  = torch.cat([pref, emb], dim=1)
            if attention_mask is not None:
                one = torch.ones((attention_mask.size(0),1), device=attention_mask.device)
                attention_mask = torch.cat([one, attention_mask], dim=1)
            return self.lm(inputs_embeds=emb, attention_mask=attention_mask, labels=labels)
    model = Slide2Text(lm, mapper)
    lm.gradient_checkpointing_enable()

    # training arguments
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

    # inference on training set
    model.eval()
    outputs = []
    for example in recs:
        feat = load_features(example)['feat']
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(train_args.device)
        with torch.no_grad():
            prefix_emb = mapper(feat_tensor).unsqueeze(1)
            gen_ids = lm.generate(inputs_embeds=prefix_emb, max_new_tokens=MAX_LEN, pad_token_id=tokenizer.eos_token_id)
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        outputs.append({
            "slide_id": example["slide_id"],
            "report": gen_text
        })

    # save as JSON array (no immuno)
    out_path = Path(args.output_dir) / "train_outputs.json"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"✅ Outputs saved to {out_path}")

if __name__ == "__main__":
    main()
