import jsonlines
import os
import numpy as np
import string
from collections import defaultdict
import argparse
from tqdm.auto import tqdm
import json
from collections import Counter

def match(pred, gold, clean_tag):
    pred = pred.replace("（", "(").replace("）", ")")
    match_pattern = f"({gold})"
    others = [item for item in ['1', '2', '3'] if item != gold]
    other_tags = [clean_tag[item] for item in ['1', '2', '3'] if item != gold]
    if match_pattern in pred or pred.strip() == gold or clean_tag[gold] in pred:
        for other in others:
            if f"({other})" in pred or pred.strip() == other:
                return False
        for other in other_tags:
            if other in pred:
                return False
        return True
    return False


def get_golden(lang):
    golden = []
    with jsonlines.open(f"/home/yfye/AAAI-2025/LayerTransplant/data/XNLI/XNLI_sample/ans/{lang}.json", "r") as f:
        for line in f:
            golden.append(line)
    return golden


def eval(path, lang):
    with jsonlines.open(f"/home/yfye/AAAI-2025/LayerTransplant/data/XNLI/XNLI_sample/ans_lang/{lang}.json") as f:
        for line in f:
            clean_tag = line
    golden = get_golden(lang)

    preds = []
    with jsonlines.open(path, "r") as f:
        for line in f:
            preds.append(line)
        
    
    acc = 0
    s = []
    for i, (pred, gold) in enumerate(zip(preds, golden)):
        if match(pred, gold, clean_tag) == True:
            acc += 1
            s.append(i)
    return acc / len(golden), s


def acc4lang(lang):
    all = {}
    for i in range(0, N):
        union_y = {}
        for j in range(0, N):
            # print(lang, i, j)
            acc, s = eval(f"/home/yfye/ICLR2025/UpperBound/XNLI_sample_reverse/{model_name}_all/transplant_{i}to{j}_firsttoken/{lang}.json", lang)
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


def find_max_indices(grid, mode):
    max_value = 0
    for i in range(N):
        if mode == "overall":
            for j in range(N):
                if grid[i, j] > max_value:
                    max_value = grid[i, j]
        elif mode == "targetfirst":
            if grid[i, 0] > max_value:
                max_value = grid[i, 0]

    print(max_value)
    indices = []  # 用来存放所有最大值的下标
    
    for i in range(N):
        if mode == "overall":
            for j in range(N):
                if grid[i, j] == max_value:
                    indices.append((i, j))  # 找到最大值时，记录下标
        elif mode == "targetfirst":
            if grid[i, 0] == max_value:
                indices.append((i, 0))  # 找到最大值时，记录下标
    
    return indices

def find_best_index(max_indices):
    # Sort by j in ascending order first, if j is the same, sort by i in descending order
    max_indices.sort(key=lambda x: (x[1], -x[0]))
    print(max_indices)
    return max_indices[0]

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="Llama-2-7b-chat-hf")

args = parser.parse_args()

num = {"Llama-2-7b-chat-hf": 32, "Qwen2-7B-Instruct": 28, "Mistral-7B-Instruct-v0.3": 32, "bloomz-7b1": 30, "chinese-alpaca-2-7b": 32}

# mode = "overall"
mode = "targetfirst"
for model_name in ["Llama-2-7b-chat-hf", "Qwen2-7B-Instruct", "Mistral-7B-Instruct-v0.3"]:
    N = num[model_name]

    langs = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
    each = []
    grid_all = defaultdict(lambda: 0)
    uppers = 0
    d = {}
    correct_num = 0
    for lang in tqdm(langs):
        grid = acc4lang(lang)
        all = []
        ttt = []
        for i in range(N+1):
            for j in range(N+1):
                grid[i, j] = len(set(grid[i, j]))
        max_indices = find_max_indices(grid, mode)
        indices = find_best_index(max_indices)
        d[lang] = indices

    os.makedirs(f"/XTransplant/ApplyExp/saved_pairs/XNLI_sample_reverse/{mode}", exist_ok=True)
    with jsonlines.open(f"/home/yfye/ICLR2025/practice/pairs_most_sourcelast/XNLI_sample_reverse/{mode}/{model_name}.json", 'w') as f:
        f.write(d)


