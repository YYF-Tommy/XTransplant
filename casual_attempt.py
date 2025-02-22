import sys
sys.path.append("/XTransplant/modelwrapper")
from llama2wrapper import Llama7BHelper
from mistralwrapper import Mistral7BHelper
from qwen2wrapper import Qwen7BHelper
import torch
import argparse
import time


def transplant_mlp(input_1, input_2, source_layer, target_layers, max_new_tokens):
    # llama = Llama7BHelper(model_name, load_kwargs, None)
    time1 = time.time()
    helper.reset_changes(None)
    output_native = helper.generate_text(input_2, max_new_tokens=1)
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
    output = helper.generate_text(input_1, max_new_tokens=max_new_tokens)
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

    output = transplant_mlp(input_main, input_aux, source_layer, target_layers, max_new_tokens=20)
    helper.reset_mlp()
    helper.reset_now_token()
    helper.reset_changes(None)

    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-i1", "--input_main", type=str, default=None)
    parser.add_argument("-i2", "--input_aux", type=str, default=None)
    parser.add_argument("-s", "--source", type=int, default=None)
    parser.add_argument("-t", "--target", type=int, default=None)

    args = parser.parse_args()

    print(torch.cuda.is_available())

    model_name = args.model
    model_folder = model_name.split('/')[-1]

    load_kwargs = dict(
        device_map="auto",
        trust_remote_code=True
    )

    input_main = open(args.input_main).read()
    input_aux = open(args.input_aux).read()

    source_layer = args.source
    target_layers = [args.target]

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
