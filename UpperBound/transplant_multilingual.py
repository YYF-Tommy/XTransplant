'''
Transplant from English to non-English (for multilingual tasks)

multilingual tasks: asking in different languages.
'''
import sys
sys.path.append("/XTransplant/modelwrapper")
from llama2wrapper import Llama7BHelper
from mistralwrapper import Mistral7BHelper
from qwen2wrapper import Qwen7BHelper
import torch
import os
import jsonlines
from tqdm.auto import tqdm
import argparse
import time

def get_xcopa_sample(lang):
    prompts_en = []
    with jsonlines.open(f"/XTransplant/data/XCOPA/XCOPA_sample/en/{lang}.json", 'r') as f:
        for line in f:
            prompts_en.append(line)
            # break
    prompts_non = []
    with jsonlines.open(f"/XTransplant/data/XCOPA/XCOPA_sample/split/{lang}.json", 'r') as f:
        for line in f:
            prompts_non.append(line)
    return prompts_en, prompts_non


def get_xnli_sample(lang):
    prompts_en = []
    with jsonlines.open(f"/XTransplant/data/XNLI/XNLI_sample/split/en.json", 'r') as f:
        for line in f:
            prompts_en.append(line)
            # break
    prompts_non = []
    with jsonlines.open(f"/XTransplant/data/XNLI/XNLI_sample/split/{lang}.json", 'r') as f:
        for line in f:
            prompts_non.append(line)
    return prompts_en, prompts_non


def get_xquad_sample(lang):
    prompts_en = []
    with jsonlines.open(f"/XTransplant/data/XQuAD/XQuAD_sample/split/xquad.en.json", 'r') as f:
        for line in f:
            prompts_en.append(line)
            # break
    prompts_non = []
    with jsonlines.open(f"/XTransplant/data/XQuAD/XQuAD_sample/split/xquad.{lang}.json", 'r') as f:
        for line in f:
            prompts_non.append(line)
    return prompts_en, prompts_non


def raw_generate(source_input, max_new_tokens):
    output = helper.generate_text(source_input, max_new_tokens=max_new_tokens)
    return output


def transplant_mlp(source_input, native_input, source_layer, target_layers, max_new_tokens):
    """
    source_input: original input (in English)
    native_input: translated input (in non-en language)
    """
    helper.reset_changes(None)
    output_native = helper.generate_text(native_input, max_new_tokens=1)
    mlp_activations = helper.get_activation(list(range(source_layer, source_layer+1)), 'mlp')
    
    value = mlp_activations.pop(source_layer)
    for i in target_layers:
        mlp_activations[i] = value

    helper.reset_changes(mlp_activations)
    helper.reset_now_token()
    output = helper.generate_text(source_input, max_new_tokens=max_new_tokens)
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
    outputs = []

    if dataset_name == "xquad_sample":
        print(f"******** Running Dataset: xquad_sample ********")
        prompts_en, prompts_non = get_xquad_sample(lang)
    elif dataset_name == "xnli_sample":
        print(f"******** Running Dataset: xnli_sample ********")
        prompts_en, prompts_non = get_xnli_sample(lang)
    elif dataset_name == "xcopa_sample":
        print(f"******** Running Dataset: xcopa_sample ********")
        prompts_en, prompts_non = get_xcopa_sample(lang)
    else:
        print("******** Unknown Dataset! ********")
        prompts = []

    print(len(prompts))
    batch_size = 1
    prompts_en = batch_split(prompts_en, batch_size)
    for i, prompt in tqdm(enumerate(prompts_en)):
        print(f"******** From {source_layer} to {target_layers} ********")
        output_native, output = transplant_mlp(prompts_non[i*batch_size: (i+1)*batch_size], prompt, source_layer, target_layers, max_new_tokens=20)
        outputs.extend(output)
        helper.reset_mlp()

    
    if not os.path.exists(f"/XTransplant/UpperBound/output/{folder_name[dataset_name]}/{model_folder}/{name}"):
        os.mkdir(f"/XTransplant/UpperBound/output/{folder_name[dataset_name]}/{model_folder}/{name}")
    with jsonlines.open(f"/XTransplant/UpperBound/output/{folder_name[dataset_name]}/{model_folder}/{name}/{lang}.json", 'w') as f_write:
        for line in outputs:
            f_write.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-d", "--dataset", type=str, default="")
    parser.add_argument("-s", "--source_layer", type=int, default=0)
    parser.add_argument("-t", "--target_layers", type=int, default=0)
    parser.add_argument("-l", "--lang", type=str, default=None)

    args = parser.parse_args()

    print(torch.cuda.is_available())

    dataset_name = args.dataset

    source_layer = args.source_layer
    target_layers = [args.target_layers]

    if args.lang != None:
        lang = args.lang
    else:
        lang = "all"

    folder_name = {
                    "xquad_sample": "XQuAD_sample",
                    "xnli_sample": "XNLI_sample",
                    "xcopa_sample": "XCOPA_sample"
                }


    model_name = args.model
    model_folder = model_name.split('/')[-1]

    name = f"transplant_{source_layer}to{target_layers[0]}"
    
    if os.path.exists(f"/XTransplant/UpperBound/output/{folder_name[dataset_name]}/{model_folder}/{name}/{lang}.json"):
        print(f"#SKIP# {name}")
    else:
        print(name)
        
        if not os.path.exists(f"/XTransplant/UpperBound/output/{folder_name[dataset_name]}"):
            os.mkdir(f"/XTransplant/UpperBound/output/{folder_name[dataset_name]}")
        if not os.path.exists(f"/XTransplant/UpperBound/output/{folder_name[dataset_name]}/{model_folder}"):
            os.mkdir(f"/XTransplant/UpperBound/output/{folder_name[dataset_name]}/{model_folder}")

        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=True
        )

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
        else:
            print("******** Unknown Model! ********")

        print(f"Num Layers: {num_layers}")
        
        transplant()
