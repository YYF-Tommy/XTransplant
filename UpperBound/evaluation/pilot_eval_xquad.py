import jsonlines
import os
import numpy as np
import string
from collections import defaultdict
import argparse
from tqdm.auto import tqdm

def get_golden(lang):
    golden = []
    with jsonlines.open(f"/XTransplant/data/XQuAD/XQuAD_sample/ans/xquad.{lang}.json", "r") as f:
        for line in f:
            golden.append(line)
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
        if gold.lower() in pred.lower():
            acc += 1
            s.append(i)
    return acc / len(golden), s


def acc4lang(lang):
    all = {}
    for i in range(0, N):
        union_y = {}
        for j in range(0, N):
            acc, s = eval(f"/XTransplant/UpperBound/XQuAD_sample/{model_name}/transplant_{i}to{j}/{lang}.json", lang)
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


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="Llama-2-7b-chat-hf")

args = parser.parse_args()

num = {"Llama-2-7b-chat-hf": 32, "Qwen2-7B-Instruct": 28, "Mistral-7B-Instruct-v0.3": 32, "bloomz-7b1": 30, "chinese-alpaca-2-7b": 32}

model_name = args.model

N = num[model_name]
langs = ["ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh"]

grid_all = defaultdict(lambda: 0)
uppers = 0
d = {}
each = []
for lang in langs:
    grid = acc4lang(lang)
    flattened = [((i, j), len(grid[i, j])) for i in range(N) for j in range(N)]

    for i in range(N):
        for j in range(N):
            grid_all[i, j] += len(grid[i, j])
    sorted_list = sorted(flattened, key=lambda x: x[1], reverse=True)
    # best_case = [item[0] for item in sorted_list[:3]]
    best_case = sorted_list[0]
    print(lang, best_case[0], best_case[1] / 50, "||| Upper Bound:", len(set(grid[N, N])) / 50) 
    each.append(len(set(grid[N, N])) / 50 * 100)
    uppers += len(set(grid[N, N]))
    d[lang] = best_case[0]

print()
print(f"\nOverall Upper Bound: {uppers / 600}")

print("\t".join(langs))
var = np.std(each)
each.append(uppers / 600 * 100)
each.append(var)
each = [str(item) for item in each]
print("\t".join(each))
print("Var:, ", var)


