import jsonlines
import os
import numpy as np
import string


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
    with jsonlines.open(f"/XTransplant/data/GlobalOpinionQA/GlobalOpinionQA_practice/split/{lang}.json", "r") as f:
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
        if match(pred, gold) == True:
            acc += 1
            s.append(i)
    return acc / len(golden), s, len(golden)


# mode = "overall"
mode = "targetfirst"
for model_name in ["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.3", "Qwen2-7B-Instruct"]:

    all = []
    log = f"{model_name} + GlobalOpinionQA_unseen + {mode}\n\n"
    files = os.listdir(f"/XTransplant/ApplyExp/output/GlobalOpinionQA_unseen/{model_name}/{mode}")
    files.sort()
    acc_count = 0
    acc_num = 0
    acc_all = []
    for file in files:
        if file.endswith(".txt") or file.startswith("all"):
            continue
        lang = file.split(".")[0]
        acc, s, num = eval(f"/XTransplant/ApplyExp/output/GlobalOpinionQA_unseen/{model_name}/{mode}/{file}", lang)
        acc_count += len(s)
        acc_num += num
        log += f"{file} Accurary: {acc}\n"
        acc_all.append(acc * 100)

    std = np.std(acc_all)
    log += f"*** Overall *** Accurary: {acc_count / acc_num}\n\n"
    acc_all.append(acc_count / acc_num * 100)
    acc_all.append(std)
    acc_all = [str(item) for item in acc_all]
    log += f"*** Overall *** Accurary: {acc_count / acc_num}\n\n"
    with open(f"/XTransplant/ApplyExp/output/GlobalOpinionQA_unseen/{model_name}/{mode}/acc_log.txt", 'w') as f_w:
        f_w.write(log)
        f_w.write("\t".join(acc_all))