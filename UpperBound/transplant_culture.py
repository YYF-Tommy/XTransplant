'''
Transplant from non-English to English (for culture-aware tasks)

culture-aware tasks: aims to evaluate the model's cultural adaptability under English context.
'''
import sys
import torch
import os
import jsonlines
from tqdm.auto import tqdm
import argparse
import time


def get_globalopinionqa_sample(dataset_name, lang):
    prompts_en = []
    with jsonlines.open(f"/XTransplant/data/GlobalOpinionQA/GlobalOpinionQA_sample/split/{lang}.json", 'r') as f:
        for line in f:
            question = line["input_text"]
            prompts_en.append(question)
            # break
    prompts_non = []
    with jsonlines.open(f"/XTransplant/data/GlobalOpinionQA/GlobalOpinionQA_sample/translated/{lang}.json", 'r') as f:
        for line in f:
            prompts_non.append(line)
    return prompts_en, prompts_non


def transplant_mlp(prompt_en, prompt_non, source_layer, target_layers, max_new_tokens):
    """
    prompt_en: input in English
    prompt_non: translated input (in non-en language)
    """
    # llama = Llama7BHelper(model_name, load_kwargs, None)
    time1 = time.time()
    helper.reset_changes(None)
    _ = helper.generate_text(prompt_non, max_new_tokens=1)
    # print(output_native)
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
    output = helper.generate_text(prompt_en, max_new_tokens=max_new_tokens)
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
    outputs = []

    if dataset_name == "globalopinionqa_sample":
        print(f"******** Running Dataset: GlobalOpinionQA_sample ********")
        prompts_en, prompts_non = get_globalopinionqa_sample(lang)
    else:
        print("******** Unknown Dataset! ********")
        prompts = []

    print(len(prompts))
    batch_size = 1
    prompts_en = batch_split(prompts_en, batch_size)
    for i, prompt in tqdm(enumerate(prompts_en)):
        print(f"******** From {source_layer} to {target_layers} ********")
        output = transplant_mlp(prompt, prompts_non[i*batch_size: (i+1)*batch_size], source_layer, target_layers, max_new_tokens=20)
        outputs.extend(output)
        helper.reset_mlp()
    
    os.makedirs(f"/XTransplant/UpperBound/output_{granularity}/{folder_name[dataset_name]}/{model_folder}/{name}", exist_ok=True)
    with jsonlines.open(f"/XTransplant/UpperBound/output_{granularity}/{folder_name[dataset_name]}/{model_folder}/{name}/{lang}.json", 'w') as f_write:
        for line in outputs:
            f_write.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-g", "--granularity", type=str, choices=['attn', 'ffn'])
    parser.add_argument("-d", "--dataset", type=str, default="")
    parser.add_argument("-s", "--source_layer", type=int, default=0)
    parser.add_argument("-t", "--target_layers", type=int, default=0)
    parser.add_argument("-l", "--lang", type=str, default=None)

    args = parser.parse_args()

    print(torch.cuda.is_available())

    granularity = args.granularity

    if granularity == "attn":
        sys.path.append("/XTransplant/modelwrapper_attn")
        from llama2wrapper import Llama7BHelper
        from mistralwrapper import Mistral7BHelper
        from qwen2wrapper import Qwen7BHelper

    elif granularity == "ffn":
        sys.path.append("/XTransplant/modelwrapper_ffn")
        from llama2wrapper import Llama7BHelper
        from mistralwrapper import Mistral7BHelper
        from qwen2wrapper import Qwen7BHelper
    else:
        print("******** Unknown Granularity! ********")

    dataset_name = args.dataset

    source_layer = args.source_layer
    target_layers = [args.target_layers]

    if args.lang != None:
        lang = args.lang
    else:
        lang = "all"

    folder_name = {"globalopinionqa_sample": "GlobalOpinionQA_sample"}

    model_name = args.model
    model_folder = model_name.split('/')[-1]

    name = f"transplant_{source_layer}to{target_layers[0]}"
    
    if os.path.exists(f"/XTransplant/UpperBound/output_{granularity}/{folder_name[dataset_name]}/{model_folder}/{name}/{lang}.json"):
        print(f"#SKIP# {name}")
    else:
        print(name)

        os.makedirs(f"/XTransplant/UpperBound/output_{granularity}/{folder_name[dataset_name]}/{model_folder}", exist_ok=True)

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
