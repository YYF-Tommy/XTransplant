# 𝒳Transplant
The repository for **"Exploring Cross-lingual Latent Transplantation: Mutual Opportunities and Open Challenges"**

<p align="center">
  <img src="Asset/method.png" width="750px" >
</p>


## Usage
### 1. Casual Attempts
Try **𝒳Transplant** between any layers for your custom input.

```python
./casual_attempt
```

### 2. Coarse-grained Attempts

<p align="center">
  <img src="Asset/settings.png" width="750px" >
</p>

```python
# attn-level transplantation
./transplant_attn

# ffn-level transplantation
./transplant_ffn
```

### 3. UpperBound Results
```python
# for XNLI, XQuAD
./UpperBound/transplant_multilingual

# for GlobalOpinionQA
./UpperBound/transplant_culture
```

### 4. Evaluation
```python
# for vanilla evaluation
./evaluation/XXX.py

# for instance-aware upper bound evaluation
./evaluation/upper_XXX.py
```


## Citation
If you find our work useful, please cite the following paper~
```
@misc{ye2024xtransplantprobeupperbound,
      title={XTransplant: A Probe into the Upper Bound Performance of Multilingual Capability and Culture Adaptability in LLMs via Mutual Cross-lingual Feed-forward Transplantation}, 
      author={Yangfan Ye and Xiaocheng Feng and Xiachong Feng and Libo Qin and Yichong Huang and Lei Huang and Weitao Ma and Zhirui Zhang and Yunfei Lu and Xiaohui Yan and Duyu Tang and Dandan Tu and Bing Qin},
      year={2024},
      eprint={2412.12686},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.12686}, 
}
```