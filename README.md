# DiagPath-LM

**DiagPath-LM** is a multimodal pathology large language model for **end-to-end slide-to-report generation** from whole-slide images (WSIs). Built upon GigaPath and LLaMA-2 backbones, the model aligns high-resolution pathology images with diagnostic text using lightweight adapters—without requiring pixel-level annotations.

🌐 **Repo**: https://github.com/Rima119/DiagPath_LM  
☁️ **Checkpoint**: [Tsinghua Cloud (adapter)](https://cloud.tsinghua.edu.cn/f/a69d2ae10b8c47449da3/)

---

## 🧬 About

DiagPath-LM introduces a **slide-to-text pipeline** designed for digital pathology. It leverages:

- **Visual backbone**: GigaPath ViT-Large (frozen)
- **Language model**: LLaMA-2, GPT2, or Prism
- **Adapter modules**: Lightweight, two-headed network aligning vision/text
- **Data**: ≈2,000 WSIs and diagnostic reports (e.g., hepatocellular carcinoma)

It uses a **bidirectional contrastive loss** to train cross-modal embeddings, enabling retrieval and full report generation from raw WSIs—**without supervision or manual annotations**.

---

## 🔧 Installation

```bash
# Clone the repo
git clone https://github.com/Rima119/DiagPath_LM.git
cd DiagPath_LM

# Install dependencies
pip install -r requirements.txt
```

## 📂 Project Structure
```bash
DiagPath_LM/
│
├── scripts/
│   ├── encode_slides_gigapath.py   # Slide → Patch → ViT features (HDF5)
│   └── train_slide2text.py         # Train LLM using extracted features + reports
│
├── data/
│   └── HCC_translation.json         # Paired WSI-report JSON file
│
├── outputs/
│   ├── level2_tile128_h5/          # Encoded slide features
│   └── slide2text_llama2_7b/       # Model checkpoints and generated reports
│
├── DiagPath_LM__FINAL_REPORT.pdf  # 📄 Full technical report
└── README.md
```

## 🧪 Example Usage

#### 1. Encode Slides to HDF5 Features

```bash
python scripts/encode_slides_gigapath.py \
  --root "path-folder" \
  --out "outputs/level2_tile128_h5" \
  --level 2 \
  --tile 128 \
  --batch_size 64 \
  --workers 8 \
  --gpu_ids 0
```

#### 2. Train the Slide-to-Text Model

Example script using Llama-2-7b-chat-hf:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python scripts/train_slide2text.py \
  --slide_dir outputs/level2_tile128_h5 \
  --feat_dim 1536 \
  --reports data/HCC_translation.json \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --epochs 100 \
  --batch_size 1 \
  --grad_accum 4 \
  --fp16 \
  --temp 1.1 \
  --top_p 0.95 \
  --top_k 50 \
  --rep_penalty 1.05 \
  --output_dir outputs/slide2text_llama2_7b
```

#### 📊 3. Evaluation
Evaluation includes both cross-modal retrieval and text generation metrics:
Retrieval: Recall@1/5/10, Median Rank
Text Generation: ROUGE-L, BLEU, BERTScore

```bash
python scripts/similarity.py
```
