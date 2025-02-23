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
            acc, s = eval(f"/XTransplant/UpperBound/GlobalOpinionQA_sample/{model_name}/transplant_{i}to{j}/{lang}.json", lang)
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

langs = ["am", "ar", "bn", "de", "el", "en", "es", "fr", "hi", "id", "it", "ja", "ko", "nl", "pt", "ru", "sv", "sw", "tl", "tr", "uk", "ur", "vi", "zh-CN"]

num = {"Llama-2-7b-chat-hf": 32, "Qwen2-7B-Instruct": 28, "Mistral-7B-Instruct-v0.3": 32, "chinese-alpaca-2-7b": 32}
model_name = args.model

N = num[model_name]

grid_all = defaultdict(lambda: 0)
uppers = 0
d = {}
each = []
for lang in tqdm(langs):
    grid = acc4lang(lang)
    # print(lang, len(set(grid[N-1, N])) / 50)
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
print(f"\nOverall Upper Bound: {uppers / 1200 * 100}")

print("\t".join(langs))
var = np.std(each)
each.append(uppers / 1200 * 100)
each.append(var)
each = [str(item) for item in each]
print("\t".join(each))
print("Var:, ", var)



