#!/usr/bin/env python3
"""Test: flip one coarse token and see impact on audio."""
import torch, numpy as np
torch.set_grad_enabled(False)
from transformers import BarkModel, AutoProcessor
from transformers.models.bark.generation_configuration_bark import *

device = "cuda"
model = BarkModel.from_pretrained("suno/bark-small").to(device)
processor = AutoProcessor.from_pretrained("suno/bark-small")
model.eval()
sem_cfg = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
coarse_cfg = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)
fine_cfg = BarkFineGenerationConfig(**model.generation_config.fine_acoustics_config)
inputs = processor("Hello, my dog is cute", voice_preset="v2/en_speaker_6", return_tensors="pt").to(device)

torch.manual_seed(42)
hf_sem = model.semantic.generate(inputs["input_ids"], history_prompt=inputs.get("history_prompt"),
    semantic_generation_config=sem_cfg)
torch.manual_seed(42)
hf_coarse = model.coarse_acoustics.generate(hf_sem, history_prompt=inputs.get("history_prompt"),
    semantic_generation_config=sem_cfg, coarse_generation_config=coarse_cfg)

# Flip position 1866: 10085 -> 10095
fixed = hf_coarse.clone()
fixed[0, 1866] = 10095

with torch.no_grad():
    hf_fine = model.fine_acoustics.generate(hf_coarse, semantic_generation_config=sem_cfg,
        coarse_generation_config=coarse_cfg, fine_generation_config=fine_cfg)
    hf_audio = model.codec_decode(hf_fine).cpu().numpy().squeeze()

    fix_fine = model.fine_acoustics.generate(fixed, semantic_generation_config=sem_cfg,
        coarse_generation_config=coarse_cfg, fine_generation_config=fine_cfg)
    fix_audio = model.codec_decode(fix_fine).cpu().numpy().squeeze()

T = min(len(hf_audio), len(fix_audio))
corr = np.corrcoef(hf_audio[:T], fix_audio[:T])[0, 1]
print(f"Single token flip at 1866/2088: corr={corr:.6f}")
