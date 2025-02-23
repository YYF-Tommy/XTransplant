import jsonlines
import os
import numpy as np
import string
from collections import defaultdict
import argparse
from tqdm.auto import tqdm

def match(pred, gold):
    pred = pred.replace("（", "(").replace("）", ")")
    match_pattern = f"({gold})"
    others = [item for item in ['1', '2'] if item != gold]
    if match_pattern in pred or pred.strip() == gold:
        for other in others:
            if f"({other})" in pred or pred.strip() == other:
                return False
        return True
    return False


def get_golden(lang):
    golden = []
    with jsonlines.open(f"/XTransplant/data/XCOPA/XCOPA_sample/ans/{lang}.json", "r") as f:
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
        if match(pred, gold) == True:
            acc += 1
            s.append(i)
    return acc / len(golden), s


def acc4lang(lang):
    all = {}
    for i in range(0, N):
        union_y = {}
        for j in range(0, N):
            acc, s = eval(f"/XTransplant/UpperBound/XCOPA_sample/{model_name}/transplant_{i}to{j}/{lang}.json", lang)
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

langs = ["en", "et", "ht", "id", "it", "sw", "ta", "th", "tr", "vi", "zh"]
each = []
grid_all = defaultdict(lambda: 0)
uppers = 0
d = {}
for lang in tqdm(langs):
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
print(f"\nOverall Upper Bound: {uppers / 550}")

print("\t".join(langs))
var = np.std(each)
each.append(uppers / 550 * 100)
each.append(var)
each = [str(item) for item in each]
print("\t".join(each))
print("Var:, ", var)
