
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
from bert_score import score as bert_score
import pandas as pd

import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device for BERTScore: {device}")

TRAIN_PATH = '../outputs/slide2text_biogpt_multi_unfreeze_100/train_outputs.json'
TRANS_PATH = '../data/HCC_translated.json'

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

    hyp_low = hyp.lower()
    ref_low = ref.lower()

    rouge_l = rouge.score(ref, hyp)['rougeL'].fmeasure

    bleu4_score = sacrebleu.sentence_bleu(hyp_low, [ref_low]).score / 100

    sent_bleu_score = sacrebleu.sentence_bleu(
        hyp_low, [ref_low], smooth_method='exp'
    ).score / 100

    try:
        P, R, F1 = bert_score(
            [hyp], [ref], lang='en', rescale_with_baseline=True,
            device=device, batch_size=64
        )
        bert_f1 = F1[0].item()
    except Exception as e:
        print(f"BERTScore error for slide {sid}: {e}")
        bert_f1 = None

    results.append({
        'slide_id': sid,
        'rougeL_f1': rouge_l,
        'bleu4': bleu4_score,
        'sentence_bleu': sent_bleu_score,
        'bertscore_f1': bert_f1
    })


df = pd.DataFrame(results)

if __name__ == '__main__':
    print(df)
    print("\nAverage scores:")
    print(df[['rougeL_f1','bleu4','sentence_bleu','bertscore_f1']].mean())