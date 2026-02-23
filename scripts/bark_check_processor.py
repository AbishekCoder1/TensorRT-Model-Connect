#!/usr/bin/env python3
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("suno/bark-small")
inputs = proc("Hello, my dog is cute", return_tensors="pt")
print("Keys:", list(inputs.keys()))
for k, v in inputs.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    if k == "attention_mask":
        print(f"    values: {v[0].tolist()}")
        print(f"    sum (num real tokens): {v.sum().item()}")
    elif k == "input_ids":
        print(f"    first 15: {v[0, :15].tolist()}")
        nz = (v[0] > 0).sum().item()
        print(f"    non-zero count: {nz}")
        print(f"    last non-zero idx: {max(i for i in range(v.shape[1]) if v[0,i] > 0)}")
