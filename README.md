# 𝒳Transplant
The repository for **"𝒳Transplant: A Probe into the Upper Bound Performance of Multilingual Capability and Culture Adaptability in LLMs via Mutual Cross-lingual Feed-forward Transplantation"**

<p align="center">
  <img src="Asset/method.png" width="750px" >
</p>


## Usage
### 1. Casual Attempts
Try **𝒳Transplant** between any layers for your custom input.

```python
./casual_attempt
```

### 2. UpperBound Results
```python
# for XNLI, XQuAD, XCOPA
./UpperBound/transplant_multilingual

# for GlobalOpinionQA
./UpperBound/transplant_culture
```

### 3. 𝒳Transplant-TargetFirst Strategy
```python
# Step-1: 
python ./ApplyExp/find_pairs_XXX.py

# Step-2:
./run
```
