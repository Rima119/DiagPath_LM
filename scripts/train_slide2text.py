#!/usr/bin/env python
# coding: utf-8
"""
scripts/train_slide2text.py  · 2025-06-19
----------------------------------------
• 支持 JSON / JSONL 报告输入
• 自动多卡切分 (device_map="auto")
• 关闭 re-entrant gradient checkpointing
• 规避 DataParallel → 直接用 DDP/FSDP/Deepspeed
• 修复 attention_mask 与 inputs_embeds 长度差 1
• 修复 mapper(fp16) ← feat(float32) dtype 不一致
"""

import os, re, json, argparse
from pathlib import Path
import h5py, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)

# ============ 全局常量 ============
MAX_LEN   = 512
HF_TOKEN  = "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"
HF_MIRROR = "https://hf-mirror.com"

os.environ["HF_TOKEN"]                 = HF_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = HF_MIRROR     # 推荐变量名

# ============ 数据预处理 ============
def load_reports(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    try:
        arr = json.loads(text)
    except Exception:
        arr = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    patt = re.compile(r"^S?(\d{4}-\d{6})")
    id2txt = {}
    for obj in arr:
        m = patt.match(obj.get("slide_id", ""))
        if not m:
            continue
        diag = obj.get("diagnosis", "").strip()
        fin  = obj.get("findings",  "").strip()
        if diag and fin:
            id2txt[m.group(1)] = {"diagnosis": diag, "findings": fin}
    return id2txt

def build_records(slide_dir: Path, id2txt: dict):
    recs = []
    for h5f in slide_dir.glob("*.h5"):
        m = re.match(r"^S?(\d{4}-\d{6})", h5f.stem)
        if not m:
            continue
        info = id2txt.get(m.group(1))
        if info:
            recs.append({"slide_id": m.group(1), "feat_path": str(h5f), **info})
    return recs

def load_features(ex):
    with h5py.File(ex["feat_path"], "r") as h5:
        feats = h5["features"][:]
    return {
        "feat": feats.mean(axis=0, dtype="float32"),
        "slide_id": ex["slide_id"],
        "diagnosis": ex["diagnosis"],
        "findings": ex["findings"],
    }

def collate_fn(batch, tokenizer, max_len):
    feats = torch.tensor([b["feat"] for b in batch])           # 保留 float32
    texts = [f"{b['diagnosis']}。{b['findings']}" for b in batch]
    enc   = tokenizer(texts, padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")

    labels = torch.cat(
        [torch.full((enc.input_ids.size(0), 1), -100, dtype=torch.long),
         enc.input_ids], dim=1)

    return {
        "feat": feats,                        # (B,D)
        "input_ids": enc.input_ids,           # (B,N)
        "attention_mask": enc.attention_mask, # (B,N)
        "labels": labels                      # (B,N+1)
    }

# ============ 模型封装 ============
class Slide2Text(torch.nn.Module):
    def __init__(self, base_lm, feat_dim):
        super().__init__()
        self.lm     = base_lm
        self.mapper = torch.nn.Linear(feat_dim, base_lm.config.hidden_size, bias=False)
        self.emb    = base_lm.get_input_embeddings()

    def forward(self, feat, input_ids=None, attention_mask=None, labels=None):
        feat   = feat.to(self.mapper.weight.dtype)            # ★ dtype 对齐
        prefix = self.mapper(feat).unsqueeze(1)               # (B,1,H)
        embTok = self.emb(input_ids)                          # (B,N,H)
        emb    = torch.cat([prefix, embTok], dim=1)           # (B,N+1,H)

        if attention_mask is not None:
            ones = torch.ones((attention_mask.size(0), 1),
                              dtype=attention_mask.dtype,
                              device=attention_mask.device)
            attention_mask = torch.cat([ones, attention_mask], dim=1)

        return self.lm(inputs_embeds=emb,
                       attention_mask=attention_mask,
                       labels=labels)

    def generate(self, feat, **kwargs):
        pref = self.mapper(feat.to(self.mapper.weight.dtype)).unsqueeze(1)
        return self.lm.generate(inputs_embeds=pref, **kwargs)

# ============ 主函数 ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide_dir",   default="outputs/level2_tile128_h5")
    ap.add_argument("--feat_dim",    type=int, default=1536)
    ap.add_argument("--reports",     default="data/HCC_translated.json")
    ap.add_argument("--model_name",  default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--epochs",      type=int, default=4)
    ap.add_argument("--batch_size",  type=int, default=1)
    ap.add_argument("--grad_accum",  type=int, default=4)
    ap.add_argument("--lr",          type=float, default=2e-5)
    ap.add_argument("--output_dir",  default="outputs/slide2text_llama7b")
    ap.add_argument("--fp16",        action="store_true")
    ap.add_argument("--bf16",        action="store_true")
    # 采样
    ap.add_argument("--temp", type=float, default=1.1)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int,   default=50)
    ap.add_argument("--rep_penalty", type=float, default=1.05)
    args = ap.parse_args()

    # ---------- 数据 ----------
    id2txt = load_reports(Path(args.reports))
    recs   = build_records(Path(args.slide_dir), id2txt)
    if not recs:
        raise RuntimeError("❌ No matched slide-report pairs.")
    ds = Dataset.from_list(recs)\
         .map(load_features, num_proc=4, desc="🔄 Load H5")\
         .remove_columns(["feat_path"])

    # ---------- 模型 ----------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=MAX_LEN,
        use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_lm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype = torch.float16 if args.fp16 or not args.bf16 else torch.bfloat16,
        device_map  = "auto",
        trust_remote_code=True,
    )
    base_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = Slide2Text(base_lm, args.feat_dim)

    # ---------- Trainer ----------
    targs = TrainingArguments(
        output_dir                = args.output_dir,
        per_device_train_batch_size = args.batch_size,
        num_train_epochs          = args.epochs,
        learning_rate             = args.lr,
        gradient_accumulation_steps = args.grad_accum,
        fp16 = args.fp16, bf16 = args.bf16,
        logging_steps             = 50,
        save_strategy             = "epoch",
        report_to                 = "none",
        ddp_find_unused_parameters= False,
        remove_unused_columns     = False,
    )
    trainer = Trainer(
        model = model,
        args  = targs,
        train_dataset = ds,
        data_collator = lambda b: collate_fn(b, tokenizer, MAX_LEN),
    )
    trainer.train()

    # ---------- 生成 ----------
    model.eval()
    outputs = []
    gen_kw = dict(max_new_tokens=MAX_LEN,
                  temperature=args.temp, top_p=args.top_p,
                  top_k=args.top_k, repetition_penalty=args.rep_penalty,
                  pad_token_id=tokenizer.eos_token_id)
    for ex in recs:
        feat = torch.tensor(ex["feat"], device=trainer.args.device)\
                   .to(model.mapper.weight.dtype)\
                   .unsqueeze(0)
        ids  = model.generate(feat=feat, **gen_kw)[0]
        txt  = tokenizer.decode(ids, skip_special_tokens=True)
        outputs.append({"slide_id": ex["slide_id"], "report": txt})

    out = Path(args.output_dir) / "train_outputs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(outputs, out.open("w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"✅ Outputs saved → {out.resolve()}")

if __name__ == "__main__":
    main()
