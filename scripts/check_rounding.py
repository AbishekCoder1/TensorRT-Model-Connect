#!/usr/bin/env python3
"""Check rounding differences near divergence point."""
import numpy as np

coarse_rate_hz = 75
semantic_rate_hz = 49.9
max_coarse_history = 630
max_coarse_input_length = 256
sem_len = 748

max_semantic_history = int(np.floor(max_coarse_history * semantic_rate_hz / coarse_rate_hz))
print(f"max_semantic_history = {max_semantic_history}")

# Check semantic window for each window
sliding_window_len = 60
for win in range(20):
    total_gen = win * sliding_window_len
    semantic_idx = int(round(total_gen * semantic_rate_hz / coarse_rate_hz))
    sem_start = max(0, semantic_idx - max_semantic_history)
    sem_context = min(sem_len - sem_start, max_coarse_input_length)
    print(f"  win={win:2d}: total_gen={total_gen:4d}, sem_idx={semantic_idx:3d}, "
          f"sem[{sem_start}:{sem_start+sem_context}]={sem_context} tokens")
    if total_gen + sliding_window_len > 2248:
        break
