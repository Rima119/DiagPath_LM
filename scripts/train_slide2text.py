#!/usr/bin/env python
# coding: utf-8
"""
scripts/train_slide2text.py   · 2025-06-18 改
--------------------------------------------
• 支持 JSON / JSONL 报告输入
• 自动多卡切分 (device_map="auto")
• 关闭 re-entrant gradient checkpointing
• 规避 DataParallel → 直接用 DDP/FSDP/Deepspeed
• 训练后直接用同一个模型 generate
"""

import os, re, json, argparse
from pathlib import Path
import h5py, torch
from datasets import Dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)

# =========================  全局常量  =========================
MAX_LEN  = 512
HF_TOKEN = "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"   # ← 你自己的 Token
HF_MIRROR= "https://hf-mirror.com"                   # 镜像端点 (可选)

os.environ.update({
    "HF_TOKEN": HF_TOKEN,
    "HUGGINGFACEHUB_API_TOKEN": HF_TOKEN,
    "HF_ENDPOINT": HF_MIRROR,
})

# =========================  数据集处理  =========================
def load_reports(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    try:        arr = json.loads(text)
    except:     arr = [json.loads(line) for line in text.splitlines() if line.strip()]
    pattern = re.compile(r"^S?(\d{4}-\d{6})")
    id2text = {}
    for obj in arr:
        sid  = obj.get("slide_id", "")
        m    = pattern.match(sid)
        if not m:           continue
        diag = obj.get("diagnosis", "").strip()
        fin  = obj.get("findings",  "").strip()
        if diag and fin:    id2text[m.group(1)] = {"diagnosis": diag, "findings": fin}
    return id2text

def build_records(slide_dir: Path, id2text: dict):
    recs = []
    for h5_path in slide_dir.glob("*.h5"):
        m = re.match(r"^S?(\d{4}-\d{6})", h5_path.stem)
        if not m: continue
        base = m.group(1)
        info = id2text.get(base)
        if info:
            recs.append({"slide_id": base, "feat_path": str(h5_path), **info})
    return recs

def load_features(ex):
    with h5py.File(ex["feat_path"], "r") as h5:
        feats = h5["features"][:]
    return {
        "feat":      feats.mean(axis=0, dtype="float32"),   # 均值池化
        "slide_id":  ex["slide_id"],
        "diagnosis": ex["diagnosis"],
        "findings":  ex["findings"],
    }

def collate_fn(batch, tokenizer, max_len):
    feats = torch.tensor([b["feat"] for b in batch], dtype=torch.float32)
    texts = [f"{b['diagnosis']}。{b['findings']}" for b in batch]
    enc   = tokenizer(texts, padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
    # labels = [-100] + input_ids
    labels = torch.cat(
        [torch.full((enc.input_ids.size(0), 1), -100, dtype=torch.long),
         enc.input_ids], dim=1)
    attn   = torch.cat(
        [torch.ones((enc.input_ids.size(0), 1), dtype=torch.long),
         enc.attention_mask], dim=1)
    return {"feat": feats, "input_ids": enc.input_ids,
            "attention_mask": attn, "labels": labels}

# =========================  模型封装  =========================
class Slide2Text(torch.nn.Module):
    def __init__(self, base_lm, feat_dim):
        super().__init__()
        self.lm      = base_lm
        self.mapper  = torch.nn.Linear(feat_dim, base_lm.config.hidden_size, bias=False)
        self.emb     = base_lm.get_input_embeddings()

    def forward(self, feat, input_ids=None, attention_mask=None, labels=None):
        pref = self.mapper(feat).unsqueeze(1)           # (B,1,H)
        emb_tok = self.emb(input_ids)                   # (B,T,H)
        emb     = torch.cat([pref, emb_tok], dim=1)     # (B,1+T,H)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [torch.ones((attention_mask.shape[0],1), device=attention_mask.device),
                 attention_mask], dim=1)

        return self.lm(inputs_embeds=emb,
                       attention_mask=attention_mask,
                       labels=labels)

    # === 直接继承 generate，方便 Trainer.save_model 后加载 ===
    def generate(self, feat, **kwargs):
        pref = self.mapper(feat).unsqueeze(1)
        return self.lm.generate(inputs_embeds=pref, **kwargs)

# =========================  主函数  =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_dir",   type=str, default="outputs/level2_tile128_h5")
    parser.add_argument("--feat_dim",    type=int, default=1536)
    parser.add_argument("--reports",     type=str, default="data/HCC_translated.json")
    parser.add_argument("--model_name",  type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--epochs",      type=int, default=4)
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--grad_accum",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--output_dir",  type=str, default="outputs/slide2text_llama7b")
    parser.add_argument("--fp16",        action="store_true")
    parser.add_argument("--bf16",        action="store_true")
    # 采样参数
    parser.add_argument("--temp", type=float, default=1.1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int,   default=50)
    parser.add_argument("--rep_penalty", type=float, default=1.05)
    args = parser.parse_args()

    # ----------  数据准备  ----------
    id2text = load_reports(Path(args.reports))
    recs    = build_records(Path(args.slide_dir), id2text)
    if not recs:
        raise RuntimeError("❌ No matched slide-report pairs.")
    ds = Dataset.from_list(recs)\
                .map(load_features, num_proc=4, desc="🔄 Load H5")\
                .remove_columns(["feat_path"])

    # ----------  模型加载  ----------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=MAX_LEN,
        pad_token=None, eos_token=None, use_fast=True,
        trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_lm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 or not args.bf16 else torch.bfloat16,
        device_map="auto",          # 自动分卡 / CPU offload
        trust_remote_code=True,
    )
    # 关闭 re-entrant 模式避坑
    base_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = Slide2Text(base_lm, args.feat_dim)

    # ----------  Trainer ----------
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        fp16=args.fp16, bf16=args.bf16,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=lambda b: collate_fn(b, tokenizer, MAX_LEN),
    )
    trainer.train()

    # ----------  生成 & 保存 ----------
    model.eval()
    outputs, gen_kwargs = [], dict(
        max_new_tokens=MAX_LEN,
        temperature=args.temp, top_p=args.top_p,
        top_k=args.top_k, repetition_penalty=args.rep_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
    for ex in recs:
        feat = torch.tensor(ex["feat"], dtype=torch.float32, device=trainer.args.device).unsqueeze(0)
        ids  = model.generate(feat=feat, **gen_kwargs)[0]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        outputs.append({"slide_id": ex["slide_id"], "report": text})

    out_path = Path(args.output_dir) / "train_outputs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(outputs, out_path.open("w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"✅ Outputs saved → {out_path.resolve()}")

if __name__ == "__main__":
    main()
