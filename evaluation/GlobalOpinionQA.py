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
    with jsonlines.open(f"/XTransplant/data/GlobalOpinionQA/GlobalOpinionQA_unseen/split/{lang}.json", "r") as f:
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


for model_name in ["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.3", "Qwen2-7B-Instruct", "Qwen2-7B-Instruct-sft"]:

    all = []
    log = f""
    files = os.listdir(f"/XTransplant/output_ffn/GlobalOpinionQA_unseen/{model_name}")
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
        acc, s, num = eval(f"/XTransplant/output_ffn/GlobalOpinionQA_unseen/{model_name}/{file}", lang)
        acc_count += len(s)
        acc_num += num
        log += f"{file} Accurary: {acc}\n"
        acc_all.append(acc * 100)


    std = np.std(acc_all)
    log += f"*** Overall *** Accurary: {acc_count / acc_num}\n\n"
    acc_all.append(acc_count / acc_num * 100)
    acc_all.append(std)
    acc_all = [str(item) for item in acc_all]
    with open(f"/XTransplant/output_ffn/GlobalOpinionQA_unseen/{model_name}/acc_log.txt", 'w') as f_w:
        f_w.write(log)
        f_w.write("\t".join(langs))
        f_w.write("\n")
        f_w.write("\t".join(acc_all))