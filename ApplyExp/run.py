import sys
sys.path.append("/XTransplant/modelwrapper")
from llama2wrapper import Llama7BHelper
from mistralwrapper import Mistral7BHelper
from qwen2wrapper import Qwen7BHelper
import torch
import torch.nn.functional as F
import os
import json
import jsonlines
from tqdm.auto import tqdm
import argparse
import time
import numpy as np
from collections import Counter

def get_xcopa_unseen(dataset_name, lang):
    prompts = []
    languages = []
    with jsonlines.open(f"/XTransplant/data/XCOPA/XCOPA_unseen/en/{lang}.json", 'r') as f:
        for line in f:
            prompts.append(line)
            languages.append(lang)
            # break
    native_inputs = []
    with jsonlines.open(f"/XTransplant/data/XCOPA/XCOPA_unseen/split/{lang}.json", 'r') as f:
        for line in f:
            native_inputs.append(line)
    return prompts, native_inputs


def get_xnli_unseen(dataset_name, lang):
    # ['ID', 'Country', 'Background', 'Axis', 'Subaxis', 'Value', 'Rule-of-Thumb', 'Story', 'Explanation', 'Gold Label']
    prompts = []
    languages = []
    with jsonlines.open(f"/XTransplant/data/XNLI/XNLI_unseen/split/en.json", 'r') as f:
        for line in f:
            prompts.append(line)
            languages.append(lang)
            # break
    native_inputs = []
    with jsonlines.open(f"/XTransplant/data/XNLI/XNLI_unseen/split/{lang}.json", 'r') as f:
        for line in f:
            native_inputs.append(line)
    return prompts, native_inputs


def get_xquad_unseen(dataset_name, lang):
    # ['ID', 'Country', 'Background', 'Axis', 'Subaxis', 'Value', 'Rule-of-Thumb', 'Story', 'Explanation', 'Gold Label']
    prompts = []
    languages = []
    with jsonlines.open(f"/XTransplant/data/XQuAD/XQuAD_unseen/split/xquad.en.json", 'r') as f:
        for line in f:
            prompts.append(line)
            languages.append(lang)
            # break
    native_inputs = []
    with jsonlines.open(f"/XTransplant/data/XQuAD/XQuAD_unseen/split/xquad.{lang}.json", 'r') as f:
        for line in f:
            native_inputs.append(line)
    return prompts, native_inputs


def get_globalopinionqa_unseen(dataset_name, lang):
    prompts = []
    native_inputs = []
    with jsonlines.open(f"/XTransplant/data/GlobalOpinionQA/GlobalOpinionQA_unseen/split/{lang}.json", 'r') as f:
        for line in f:
            question = line["input_text"]
            prompts.append(question)
            # break
    
    with jsonlines.open(f"/XTransplant/data/GlobalOpinionQA/GlobalOpinionQA_unseen/translated/{lang}.json", 'r') as f:
        for line in f:
            native_inputs.append(line)
        return prompts, native_inputs


def transplant_mlp(input_1, input_2, source_layer, target_layers, max_new_tokens):
    # llama = Llama7BHelper(model_name, load_kwargs, None)
    time1 = time.time()
    helper.reset_changes(None)
    _ = helper.generate_text(input_2, max_new_tokens=1)
    time2 = time.time()
    mlp_activations = helper.get_activation(list(range(source_layer, source_layer+1)), 'mlp')
    
    value = mlp_activations.pop(source_layer)
    for i in target_layers:
        mlp_activations[i] = value

    time3 = time.time()
    # print(mlp_activations)
    # print(len(mlp_activations[25]["mlp"]))
    helper.reset_changes(mlp_activations)
    helper.reset_now_token()
    time4 = time.time()
    output = helper.generate_text(input_1, max_new_tokens=max_new_tokens)
    time5 = time.time()
    # print(time2-time1, time3-time2, time4-time3, time5-time4)
    return output
    

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def transplant():
    raw_outputs = []
    outputs = []
    native_outputs = []

    if dataset_name == "globalopinionqa_unseen":
        print(f"******** Running Dataset: GlobalOpinionQA_unseen ********")
        prompts, native_inputs = get_globalopinionqa_unseen(dataset_name, lang)
    elif dataset_name == "xquad_unseen":
        print(f"******** Running Dataset: xquad_unseen ********")
        prompts, native_inputs = get_xquad_unseen(dataset_name, lang)
    elif dataset_name == "xnli_unseen":
        print(f"******** Running Dataset: xnli_unseen ********")
        prompts, native_inputs = get_xnli_unseen(dataset_name, lang)
    elif dataset_name == "xcopa_unseen":
        print(f"******** Running Dataset: xcopa_unseen ********")
        prompts, native_inputs = get_xcopa_unseen(dataset_name, lang)
    else:
        print("******** Unknown Dataset! ********")
        prompts = []

    multilingual = ["xquad_unseen", "xnli_unseen", "xcopa_unseen"]
    culture = ["globalopinionqa_unseen"]

    print(len(prompts))
    batch_size = 1
    prompts = batch_split(prompts, batch_size)
    
    with jsonlines.open(f"/XTransplant/ApplyExp/saved_pairs/{folder_name[dataset_name].replace('unseen', 'sample')}/{mode}/{model_folder}.json") as f:
        for line in f:
            pair = line[lang]
    print(pair)
    for i, prompt in tqdm(enumerate(prompts)):
        source_layer = pair[0]
        target_layers = [pair[1]]
        if dataset_name in multilingual:
            output = transplant_mlp(native_inputs[i*batch_size: (i+1)*batch_size], prompt, source_layer, target_layers, max_new_tokens=20)
            outputs.extend(output)
            helper.reset_mlp()
            helper.reset_now_token()
            helper.reset_changes(None)

        elif dataset_name in culture:
            output = transplant_mlp(prompt, native_inputs[i*batch_size: (i+1)*batch_size], source_layer, target_layers, max_new_tokens=20)
            outputs.extend(output)
            helper.reset_mlp()
            helper.reset_now_token()
            helper.reset_changes(None)

    os.makedirs(f"/XTransplant/ApplyExp/output/{folder_name[dataset_name]}/{model_folder}/{mode}", exist_ok=True)
    with jsonlines.open(f"/XTransplant/ApplyExp/output/{folder_name[dataset_name]}/{model_folder}/{mode}/{lang}.json", 'w') as f_write:
        for line in outputs:
            f_write.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-d", "--dataset", type=str, default="")
    parser.add_argument("-l", "--lang", type=str, default=None)
    parser.add_argument("-md", "--mode", type=str, default="overall")

    args = parser.parse_args()

    print(torch.cuda.is_available())

    dataset_name = args.dataset

    if args.lang != None:
        lang = args.lang
    else:
        lang = "target_first"

    mode = args.mode

    folder_name = {
                "xquad_unseen": "XQuAD_unseen",
                "xnli_unseen": "XNLI_unseen",
                "xcopa_unseen": "XCOPA_unseen",
                "globalopinionqa_unseen": "GlobalOpinionQA_unseen"
            }

    model_name = args.model
    model_folder = model_name.split('/')[-1]

    load_kwargs = dict(
        device_map="auto",
        trust_remote_code=True
    )

    if model_folder == "Llama-2-7b-chat-hf":
        print(f"******** Running Model: Llama-2-7b-chat-hf ********")
        helper = Llama7BHelper(model_name, load_kwargs, None)
        num_layers = helper.model.config.num_hidden_layers
    elif model_folder == "Mistral-7B-Instruct-v0.3":
        print(f"******** Running Model: Mistral-7B-Instruct-v0.3 ********")
        helper = Mistral7BHelper(model_name, load_kwargs, None)
        num_layers = helper.model.config.num_hidden_layers
    elif model_folder == "Qwen2-7B-Instruct":
        print(f"******** Running Model: Qwen2-7B-Instruct ********")
        helper = Qwen7BHelper(model_name, load_kwargs, None)
        num_layers = helper.model.config.num_hidden_layers
    else:
        print("******** Unknown Model! ********")

    print(f"Num Layers: {num_layers}")
    
    transplant()
