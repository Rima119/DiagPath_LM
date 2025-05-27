#!/usr/bin/env python
# coding: utf-8
"""
scripts/train_slide2text.py

最小 Demo：用 H5 特征（Mean Pool）+ 线性映射 前缀化 送入 GPT2 生成报告
支持两种报告文件格式：
 1. JSON 数组: [ {…}, {…}, … ]
 2. JSONL:    每行一个 {…}
"""
import os
# ———— 在导入 HF/transformers 之前设置 ————
# 如果你在用内部镜像，不要把 token 传给它：
os.environ["HF_HUB_DISABLE_SSL_VERIFY"]  = "1"
os.environ["HF_ENDPOINT"]                = "https://hf-mirror.com"
import re
import json
import h5py
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# —— 用户配置区 —— #
SLIDE_DIR  = Path("./data/ultralowres_h5")      # H5 特征文件夹
REPORTS    = Path("./data/HCC_translated.json") # JSON 数组或 JSONL
FEAT_DIM   = 192                                # 特征维度
PREFIX_DIM = 768                                # GPT2 hidden size
MAX_LEN    = 512                                # 文本最大长度
BATCH_SIZE = 4                                  # Trainer batch size
EPOCHS     = 1                                  # 训练轮数
LR         = 2e-5                               # 学习率
OUTPUT_DIR = "outputs/slide2text_demo"          # 输出目录
MODEL_NAME = "gpt2"                             # English GPT-2
# ———————— #

def load_reports(path):
    """
    读取 REPORTS 文件，支持：
      - 整个是一个 JSON 数组
      - 或者每行一个 JSON 对象（JSONL）
    返回 { base_id: text }
    """
    text = path.read_text(encoding="utf-8").strip()
    # 先尝试当 JSON 数组
    try:
        arr = json.loads(text)
        if not isinstance(arr, list):
            raise ValueError
    except Exception:
        # 退回到 JSONL：按行 parse
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

    # 把 diagnosis + findings 拼成一句文本，忽略 immuno
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
            id2text[base] = f"{diag}。{fin}"
    return id2text

def build_records(slide_dir, id2text):
    """
    遍历 slide_dir/*.h5，按正则提取 base_id，
    如果在 id2text 里，就加入 records 列表。
    """
    recs = []
    for h5_path in slide_dir.glob("*.h5"):
        m = re.match(r"^S?(\d{4}-\d{6})", h5_path.stem)
        if not m:
            continue
        base = m.group(1)
        txt  = id2text.get(base)
        if txt:
            recs.append({"feat_path": str(h5_path), "text": txt})
    return recs

def load_features(ex):
    """
    Dataset.map 用函数：打开 H5，Mean Pool features
    """
    with h5py.File(ex["feat_path"], "r") as h5:
        feats = h5["features"][:]      # (N, FEAT_DIM)
        mf    = feats.mean(axis=0)     # (FEAT_DIM,)
    return {"feat": mf.astype("float32"), "text": ex["text"]}

def main():
    # 1) 读报告、匹配 slide_id
    id2text = load_reports(REPORTS)
    recs    = build_records(SLIDE_DIR, id2text)
    print(f"🔗 matched {len(recs)} slide-report pairs")
    if not recs:
        print("❌ 没有匹配到任何 pair，请检查 REPORTS 路径与 slide_id 格式")
        return

    # 2) 构造 HF Dataset
    ds = Dataset.from_list(recs)
    # ←—— 这里只 remove feat_path，保留 text 列 —— removed `"text"` from remove_columns
    ds = ds.map(load_features,
                remove_columns=["feat_path"],
                new_fingerprint="feat_loaded")

    # 3) Tokenizer & LM
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=MAX_LEN,
        use_fast=True
    )
    tok.pad_token = tok.eos_token

    lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME).half().cuda()

    # 4) 特征映射层 FEAT_DIM->PREFIX_DIM
    mapper = torch.nn.Linear(FEAT_DIM, PREFIX_DIM, bias=False).half().cuda()

    # 5) 封装 Module
    class Slide2Text(torch.nn.Module):
        def __init__(self, lm, mapper):
            super().__init__()
            self.lm     = lm
            self.mapper = mapper

        def forward(self, feat, input_ids=None, attention_mask=None, labels=None):
            pref = self.mapper(feat).unsqueeze(1)  # (B,1,D)
            emb  = self.lm.transformer.wte(input_ids)
            emb  = torch.cat([pref, emb], dim=1)
            if attention_mask is not None:
                one = torch.ones((attention_mask.size(0),1),
                                 device=attention_mask.device)
                attention_mask = torch.cat([one, attention_mask], dim=1)
            return self.lm(inputs_embeds=emb,
                           attention_mask=attention_mask,
                           labels=labels)

    model = Slide2Text(lm, mapper)

    # 6) Data collator
    def collate_fn(batch):
        feats = torch.tensor([b["feat"] for b in batch]).half().cuda()
        texts = [b["text"] for b in batch]       # now text still exists!
        enc   = tok(texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt")
        return {
            "feat":           feats,
            "input_ids":      enc.input_ids.cuda(),
            "attention_mask": enc.attention_mask.cuda(),
            "labels":         enc.input_ids.cuda(),
        }

    # 7) 训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=True,
        save_total_limit=2,
        logging_steps=10,
        report_to=["none"],
    )

    # 8) Trainer & 训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collate_fn,
    )
    trainer.train()
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
