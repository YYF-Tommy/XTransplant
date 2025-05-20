import jsonlines
import os
import numpy as np
import string
from collections import defaultdict
import argparse
from tqdm.auto import tqdm

def match(pred, gold):
    match_pattern = f"({gold})"
    others = [item for item in string.ascii_uppercase if item != gold]
    if match_pattern in pred.upper() or pred.upper().strip() == gold:
        for other in others:
            if f"({other})" in pred.upper() or pred.upper().strip() == other:
                return False
        return True
    return False


def get_golden(lang):
    id2label = {}
    for i, letter in enumerate(string.ascii_uppercase):
        id2label[i] = letter
    golden = []
    ls = []
    with jsonlines.open(f"/XTransplant/data/GlobalOpinionQA/GlobalOpinionQA_sample/split/{lang}.json", "r") as f:
        for line in f:
            probs = line["label"]
            label = id2label[np.argmax(probs)]
            golden.append(label)
    return golden
    

def eval(path, lang):
    golden = get_golden(lang)
    preds = []
    with jsonlines.open(path, "r") as f:
        for line in f:
            preds.append(line)
    acc = 0
    s = []
    for i, (pred, gold) in enumerate(zip(preds, golden)):
        # print(pred, gold)
        if match(pred, gold) == True:
            acc += 1
            s.append(i)
    # print(acc / len(golden))
    return acc / len(golden), s


def acc4lang(lang):
    all = {}
    for i in range(0, N):
        union_y = {}
        for j in range(0, N):
            acc, s = eval(f"/XTransplant/output_ffn/GlobalOpinionQA_sample/{model_name}/transplant_{i}to{j}_firsttoken/{lang}.json", lang)
            all[i, j] = s.copy()
    union = []
    for i in range(0, N):
        union_x = []
        union_y = []
        for j in range(0, N):
            union_y.extend(all[i, j].copy())
            union_x.extend(all[j, i].copy())
        
        all[i, N] = union_y
        all[N, i] = union_x
        
        union.extend(union_y)
        union.extend(union_x)
    all[N, N] = union
    
    return all


langs = ["am", "ar", "de", "el", "en", "es", "fr", "hi", "id", "ja", "pt", "ru", "sv", "sw", "tl", "tr", "uk", "ur", "vi", "zh-CN"]

num = {"Llama-2-7b-chat-hf": 32, "Qwen2-7B-Instruct": 28, "Mistral-7B-Instruct-v0.3": 32, "bloomz-7b1": 30, "chinese-alpaca-2-7b": 32}

for model_name in ["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.3", "Qwen2-7B-Instruct"]:
    print(model_name)

    N = num[model_name]

    grid_all = defaultdict(lambda: 0)
    uppers = 0
    d = {}
    each = []
    for lang in tqdm(langs):
        grid = acc4lang(lang)

        for i in range(N + 1):
            for j in range(N + 1):
                grid_all[i, j] += len(set(grid[i, j]))


    print("source")
    tmp = []
    for i in range(N):
        tmp.append(f"({i+1}, "+str(round((grid_all[i, N]) / (50 * len(langs)) * 100, 1)) + ")")
    print(" ".join(tmp))

    print("target")
    tmp = []
    for i in range(N):
        tmp.append(f"({i+1}, "+str(round((grid_all[N, i]) / (50  * len(langs)) * 100, 1)) + ")")
    print(" ".join(tmp))
    print()
    print()