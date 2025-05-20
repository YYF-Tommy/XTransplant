import jsonlines
import os
import numpy as np
import string


def get_golden(lang):
    golden = []
    with jsonlines.open(f"/XTransplant/data/XQuAD/XQuAD_unseen/ans/xquad.{lang}.json", "r") as f:
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


for model_name in ["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.3", "Qwen2-7B-Instruct"]:
    all = []
    log = f""
    files = os.listdir(f"/XTransplant/output_ffn/XQuAD_unseen/{model_name}")
    files.sort()
    files.remove("en.json") 
    files.insert(0, "en.json")  
    acc_count = 0
    acc_num = 0
    acc_all = []
    langs = []
    for file in files:
        if file.endswith(".txt") or file.startswith("all"):
            continue
        langs.append(file.split(".")[0])
        lang = file.split(".")[0]
        acc, s, num = eval(f"/XTransplant/output_ffn/XQuAD_unseen/{model_name}/{file}", lang)
        acc_count += len(s)
        acc_num += num
        log += f"{file} Accurary: {acc}\n"
        acc_all.append(acc * 100)


    std = np.std(acc_all)
    log += f"*** Overall *** Accurary: {acc_count / acc_num}\n\n"
    acc_all.append(acc_count / acc_num * 100)
    acc_all.append(std)
    acc_all = [str(item) for item in acc_all]
    with open(f"/XTransplant/output_ffn/XQuAD_/XTransplant/{model_name}/acc_log.txt", 'w') as f_w:
        f_w.write(log)
        f_w.write("\t".join(langs))
        f_w.write("\n")
        f_w.write("\t".join(acc_all))