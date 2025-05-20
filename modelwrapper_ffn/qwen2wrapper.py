# encoding=utf-8

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.attn_weights = None
        self.key_value = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            # 虽然有这么一步，但好像一直没有add
            output = (output[0] + self.add_tensor,)+output[1:]
        # self.activations.shape = (batch, seq_len, 4096)

        self.activations = output[0]
        self.attn_weights = output[1]
        self.key_value = output[2]

        
        # print(output)
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, idx, config, changes=None):
        super().__init__()
        self.config = config
        self.changes = changes

        self.layer_idx = idx
        self.block = block                       # decoder

        # self.input_layernorm = self.block.input_layernorm
        self.post_attention_layernorm = self.block.post_attention_layernorm

        # self.block.self_attn = AttnWrapper(self.block.self_attn)

        self.block_output = None

        self.mlp_output_2 = None

        self.attn_mech_output = None
        self.mlp_output = []

        self.now_token = 0


    def forward(self, *args, **kwargs):
        hidden_states = self.block.input_layernorm(args[0])
        hidden_states = (hidden_states,)

        attn_output, self_attn_weights, present_key_value = self.block.self_attn(*hidden_states, **kwargs)

        attn_output += args[0]
        self.attn_mech_output = attn_output

        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))

        self.mlp_output_2 = mlp_output
        if self.changes and self.layer_idx in list(self.changes.keys()) and "mlp" in self.changes[self.layer_idx]:
            if self.now_token == 0:
                mlp_output[:, -1, :] = self.changes[self.layer_idx]["mlp"][self.now_token][:, -1, :]
                

        self.mlp_output.append(mlp_output) # 占内容

        outputs = (attn_output + mlp_output,)

        if kwargs["output_attentions"]:
            outputs += (self_attn_weights,)

        if kwargs["use_cache"]:
            outputs += (present_key_value,)

        self.now_token += 1
        
        return outputs

    def reset(self):
        self.block.self_attn.reset()


class Qwen7BHelper:
    def __init__(self, model_name, load_kwargs, changes=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        # print(self.model)
        self.config = self.model.config
        self.changes = changes
        self.embedding_layer = self.model.model.embed_tokens
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm, i, self.config, self.changes)

    def generate_text(self, prompt, max_new_tokens=500):
        """
        Direct generation
        """
        inputs = self.tokenizer.batch_encode_plus(prompt, padding=True, return_tensors="pt")
        # print(inputs)
        inputs = {k: inputs[k] for k in inputs if k in ["input_ids", "attention_mask"]}
        for t in inputs:
            if torch.is_tensor(inputs[t]):
                inputs[t] = inputs[t].to("cuda")
        # print(inputs)
        # print(inputs.input_ids.shape)
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id, do_sample=False)
        return self.tokenizer.batch_decode(generate_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def get_logits(self, prompt):
        """
        logits = last_hidden_state @ self.model.lm_head
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # print(f"Input ids: {inputs.input_ids}")
        with torch.no_grad():
          logits = self.model(inputs.input_ids.to("cuda")).logits
        #   print(self.model(inputs.input_ids.to(self.device), output_hidden_states=True).hidden_states[0])
          return logits

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def reset_changes(self, changes):
        for layer in self.model.model.layers:
            layer.changes = changes

    def reset_mlp(self):
        for layer in self.model.model.layers:
            layer.mlp_output = []

    def reset_now_token(self):
        for layer in self.model.model.layers:
            layer.now_token = 0

    def get_activation(self, layer_idx_list, loc):
        activations = {}
        for layer_idx in layer_idx_list:
            layer = self.model.model.layers[layer_idx]
            if loc == "attn":
                activations[layer_idx] = {"attn": layer.attn_mech_output}
            elif loc == "mlp":
                activations[layer_idx] = {"mlp": layer.mlp_output}
            elif loc == "mlp2":
                activations[layer_idx] = {"mlp2": layer.mlp_output_2}
            elif loc == "block":
                activations[layer_idx] = {"block": layer.block_output}
        return activations
