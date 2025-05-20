import sys
sys.path.append("/XTransplant/modelwrapper_attn")
from llama2wrapper_deactivate import Llama7BHelper
from mistralwrapper_deactivate import Mistral7BHelper
from qwen2wrapper_deactivate import Qwen7BHelper
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import jsonlines
from tqdm.auto import tqdm
import argparse
import time
import numpy as np
from collections import Counter

def get_xnli_unseen(dataset_name, lang):
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

def get_xnli_sample(dataset_name, lang):
    prompts = []
    languages = []
    with jsonlines.open(f"/XTransplant/data/XNLI/XNLI_sample/split/en.json", 'r') as f:
        for line in f:
            prompts.append(line)
            languages.append(lang)
            # break
    native_inputs = []
    with jsonlines.open(f"/XTransplant/data/XNLI/XNLI_sample/split/{lang}.json", 'r') as f:
        for line in f:
            native_inputs.append(line)
    return prompts, native_inputs


def raw_generate(main_input, max_new_tokens):
    output = helper.generate_text(main_input, max_new_tokens=max_new_tokens)
    return output


def transplant_attn(main_input, aux_input, source_layer, target_layers, max_new_tokens):
    """
    main_input: original main input
    aux_input: auxiliary input
    """
    time1 = time.time()
    helper.reset_changes(None)
    output_native = helper.generate_text(aux_input, max_new_tokens=1)
    # print(output_native)
    time2 = time.time()
    attn_activations = helper.get_activation(list(range(source_layer, source_layer+1)), 'attn')

    value = attn_activations.pop(source_layer)
    for i in target_layers:
        attn_activations[i] = value

    time3 = time.time()

    helper.reset_changes(attn_activations)
    helper.reset_now_token()
    time4 = time.time()
    output = helper.generate_text(main_input, max_new_tokens=max_new_tokens)
    time5 = time.time()
    # print(time2-time1, time3-time2, time4-time3, time5-time4)
    return output_native, output


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

    if  dataset_name == "xnli_unseen":
        print(f"******** Running Dataset: xnli_unseen ********")
        prompts, native_inputs = get_xnli_unseen(dataset_name, lang)
    elif dataset_name == "xnli_sample":
        print(f"******** Running Dataset: xnli_sample ********")
        prompts, native_inputs = get_xnli_sample(dataset_name, lang)
    else:
        print("******** Unknown Dataset! ********")
        prompts = []


    print(len(prompts))
    batch_size = 4
    prompts = batch_split(prompts, batch_size)
    for i, prompt in tqdm(enumerate(prompts)):
        print(f"******** From {source_layer} to {target_layers} ********")
        output_native, output = transplant_attn(native_inputs[i*batch_size: (i+1)*batch_size], prompt, source_layer, target_layers, max_new_tokens=20)
        outputs.extend(output)
        native_outputs.append(output_native)
        helper.reset_attn()

    
    os.makedirs(f"/XTransplant/output_attn/{folder_name[dataset_name]}/{model_folder}", exist_ok=True)
    with jsonlines.open(f"/XTransplant/output_attn/{folder_name[dataset_name]}/{model_folder}/{lang}.json", 'w') as f_write:
        for line in outputs:
            f_write.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-d", "--dataset", type=str, default="")
    parser.add_argument("-l", "--lang", type=str, default=None)

    args = parser.parse_args()

    print(torch.cuda.is_available())

    dataset_name = args.dataset

    load_kwargs = dict(
        device_map="auto",
        trust_remote_code=True
    )
    model_name = args.model
    model_folder = model_name.split('/')[-1]

    if model_folder == "Llama-2-7b-chat-hf":
        print(f"******** Running Model: Llama-2-7b-chat-hf ********")
        helper = Llama7BHelper(model_name, load_kwargs, None)
        num_layers = helper.model.config.num_hidden_layers
    elif model_folder == "chinese-alpaca-2-7b":
        print(f"******** Running Model: chinese-alpaca-2-7b ********")
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
    elif model_folder == "Qwen2-7B-Instruct-sft":
        print(f"******** Running Model: Qwen2-7B-Instruct ********")
        helper = Qwen7BHelper(model_name, load_kwargs, None)
        num_layers = helper.model.config.num_hidden_layers
    else:
        print("******** Unknown Model! ********")

    print(f"Num Layers: {num_layers}")

    # or you can change the configuration manually
    source_layer = num_layers-1 # count from 0, so "num_layers-1" represents the last layer
    target_layers = [0]

    if args.lang != None:
        lang = args.lang
    else:
        lang = "all"

    folder_name = {
                "xnli_sample": "XNLI_sample",
                "xcopa_unseen": "XCOPA_unseen",
            }

    transplant()