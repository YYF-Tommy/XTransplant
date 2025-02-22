import jsonlines
import os
import numpy as np
import string


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
    with jsonlines.open(f"/XTransplant/data/XNLI/XNLI_practice/ans/{lang}.json", "r") as f:
        for line in f:
            golden.append(line)
    return golden


def eval(path, lang):
    with jsonlines.open(f"/XTransplant/data/XNLI/XNLI_practice/ans_lang/{lang}.json") as f:
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
    return acc / len(golden), s, len(golden)

# mode = "overall"
mode = "targetfirst"
for model_name in ["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.3", "Qwen2-7B-Instruct"]:

    all = []
    log = f"{model_name} + XNLI_unseen + {mode}\n\n"
    files = os.listdir(f"/XTransplant/ApplyExp/output/XNLI_unseen/{model_name}/{mode}")
    files.sort()
    files.remove("en.json")  # 删除元素
    files.insert(0, "en.json")  # 插入到第一位
    acc_count = 0
    acc_num = 0
    acc_all = []
    langs = []
    for file in files:
        if file.endswith(".txt") or file.startswith("all"):
            continue
        langs.append(file.split(".")[0])
        lang = file.split(".")[0]
        acc, s, num = eval(f"/XTransplant/ApplyExp/output/XNLI_unseen/{model_name}/{mode}/{file}", lang)
        acc_count += len(s)
        acc_num += num
        log += f"{file} Accurary: {acc}\n"
        acc_all.append(acc * 100)


    std = np.std(acc_all)
    log += f"*** Overall *** Accurary: {acc_count / acc_num}\n\n"
    acc_all.append(acc_count / acc_num * 100)
    acc_all.append(std)
    acc_all = [str(item) for item in acc_all]
    with open(f"/XTransplant/ApplyExp/output/XNLI_unseen/{model_name}/{mode}/acc_log.txt", 'w') as f_w:
        f_w.write(log)
        f_w.write("\t".join(langs))
        f_w.write("\n")
        f_w.write("\t".join(acc_all))

