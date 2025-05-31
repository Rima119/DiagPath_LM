import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from collections import OrderedDict

# Metric libraries
import torch
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import BERTScorer
import pandas as pd

import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device for BERTScore: {device}")

TRAIN_PATH = 'outputs/slide2text_gptneo2.7b/train_outputs.json'
TRANS_PATH = 'data/HCC_translated.json'

# Initialize BERTScorer
bert_scorer = BERTScorer(lang='en', rescale_with_baseline=True, device=device)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

train = load_json(TRAIN_PATH)
trans = load_json(TRANS_PATH)

train_dict = OrderedDict()
for entry in train:
    sid = entry['slide_id']
    if sid not in train_dict:
        train_dict[sid] = entry['report']

trans_dict = {}
for entry in trans:
    sid = entry['slide_id'].lstrip('S')
    trans_dict[sid] = entry['diagnosis'] + '. ' + entry['findings']

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

results = []
for sid, hyp in train_dict.items():
    if sid not in trans_dict:
        continue
    ref = trans_dict[sid]

    # Case-insensitive evaluation
    hyp_low = hyp.lower()
    ref_low = ref.lower()

    # ROUGE-L
    rouge_l = rouge.score(ref_low, hyp_low)['rougeL'].fmeasure

    # BLEU-4 (using sacrebleu)
    bleu4_score = sacrebleu.sentence_bleu(hyp_low, [ref_low]).score / 100
    
    # Sentence BLEU with smoothing
    sent_bleu = sacrebleu.sentence_bleu(
        hyp_low, 
        [ref_low],
        smooth_method='exp',
        smooth_value=0.1  # Added smoothing
    ).score / 100

    # Proper BERTScore calculation
    P, R, F1 = bert_scorer.score([hyp], [ref])
    bert_f1 = F1.item()

    results.append({
        'slide_id': sid,
        'rougeL_f1': rouge_l,
        'bleu4': bleu4_score,
        'sentence_bleu': sent_bleu,
        'bertscore_f1': bert_f1
    })


df = pd.DataFrame(results)

if __name__ == '__main__':
    print(df)
    print("\nAverage scores:")
    print(df[['rougeL_f1','bleu4','sentence_bleu','bertscore_f1']].mean())
