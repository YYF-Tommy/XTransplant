import jsonlines
import os
import numpy as np
import string


def get_golden(lang):
    golden = []
    with jsonlines.open(f"/XTransplant/data/XQuAD/XQuAD_practice/ans/xquad.{lang}.json", "r") as f:
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
    return acc / len(golden), s, len(golden)


# mode = "overall"
mode = "targetfirst"
for model_name in ["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.3", "Qwen2-7B-Instruct"]:

    all = []
    log = f"{model_name} + XQuAD_unseen + {mode}\n\n"
    files = os.listdir(f"/XTransplant/ApplyExp/output/XQuAD_unseen/{model_name}/{mode}")
    files.sort()
    files.remove("en.json") 
    files.insert(0, "en.json")  
    acc_count = 0
    acc_num = 0
    acc_all = []
    for file in files:
        if file.endswith(".txt") or file.startswith("all"):
            continue
        lang = file.split(".")[0]
        acc, s, num = eval(f"/XTransplant/ApplyExp/output/XQuAD_unseen/{model_name}/{mode}/{file}", lang)
        acc_count += len(s)
        acc_num += num
        log += f"{file} Accurary: {acc}\n"
        acc_all.append(acc * 100)

    std = np.std(acc_all)
    log += f"*** Overall *** Accurary: {acc_count / acc_num}\n\n"
    acc_all.append(acc_count / acc_num * 100)
    acc_all.append(std)
    acc_all = [str(item) for item in acc_all]
    with open(f"/XTransplant/ApplyExp/output/XQuAD_unseen/{model_name}/{mode}/acc_log.txt", 'w') as f_w:
        f_w.write(log)
        f_w.write("\t".join(acc_all))