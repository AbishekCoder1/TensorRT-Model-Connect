#!/usr/bin/env python3
"""Trace what C++ vs HF construct as input to the coarse model for each window."""
import numpy as np
import sys

cpp_sem = np.array([int(l.strip()) for l in open("/tmp/bark_greedy_dump.sem_tokens")])
cpp_coarse = np.array([int(l.strip()) for l in open("/tmp/bark_greedy_dump.coarse_tokens")])

# Config
coarse_semantic_pad_token = 12048
coarse_infer_token = 12050
max_coarse_input_length = 256
max_coarse_history = 630
sliding_window_len = 60
n_coarse_codebooks = 2
coarse_rate_hz = 75
semantic_rate_hz = 49.9

# C++ preprocessing: replace semantic_pad_token with coarse_semantic_pad_token
x_semantic = np.where(cpp_sem == 10000, coarse_semantic_pad_token, cpp_sem)
sem_len = len(x_semantic)

semantic_to_coarse_ratio = coarse_rate_hz / semantic_rate_hz
max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))

print(f"Semantic tokens: {sem_len}")
print(f"Coarse tokens: {len(cpp_coarse)}")
print(f"semantic_to_coarse_ratio: {semantic_to_coarse_ratio:.4f}")
print(f"max_semantic_history: {max_semantic_history}")
print()

for win in range(3):
    total_gen = win * sliding_window_len  # coarse tokens generated so far
    print(f"=== Window {win} (total_gen={total_gen}) ===")

    # ---- C++ construction ----
    sliding_ndx = total_gen // n_coarse_codebooks
    sem_start_cpp = max(0, sliding_ndx - max_coarse_input_length)
    sem_end_cpp = min(sem_len, sliding_ndx + max_coarse_input_length)
    sem_context_len = sem_end_cpp - sem_start_cpp
    n_pad_left = max(0, max_coarse_input_length - sem_context_len)

    cpp_sem_input = np.concatenate([
        np.full(n_pad_left, coarse_semantic_pad_token, dtype=np.int32),
        x_semantic[sem_start_cpp:sem_end_cpp],
    ])

    print(f"  C++ : sliding_ndx={sliding_ndx}, "
          f"sem[{sem_start_cpp}:{sem_end_cpp}]={sem_context_len} tokens, "
          f"pad_left={n_pad_left}")
    print(f"       total semantic portion: {len(cpp_sem_input)} tokens")

    # ---- HF construction ----
    semantic_idx = int(round(total_gen / semantic_to_coarse_ratio))
    slice_start = max(0, semantic_idx - max_semantic_history)
    hf_sem_input = x_semantic[slice_start:][:max_coarse_input_length]
    pad_right = max_coarse_input_length - len(hf_sem_input)

    # HF right-pads
    if pad_right > 0:
        hf_sem_input = np.concatenate([
            hf_sem_input,
            np.full(pad_right, coarse_semantic_pad_token, dtype=np.int32),
        ])

    print(f"  HF  : semantic_idx={semantic_idx}, "
          f"sem[{slice_start}:{slice_start + min(len(x_semantic) - slice_start, max_coarse_input_length)}], "
          f"pad_right={pad_right}")
    print(f"       total semantic portion: {len(hf_sem_input)} tokens")

    # ---- Compare ----
    if len(cpp_sem_input) == len(hf_sem_input):
        match = int(np.sum(cpp_sem_input == hf_sem_input))
        total = len(cpp_sem_input)
        if match == total:
            print(f"  MATCH: {match}/{total} tokens identical")
        else:
            first_diff = int(np.argmax(cpp_sem_input != hf_sem_input))
            print(f"  DIFFER: {match}/{total} match, first diff at [{first_diff}]: "
                  f"C++={cpp_sem_input[first_diff]}, HF={hf_sem_input[first_diff]}")
    else:
        print(f"  LENGTH MISMATCH: C++={len(cpp_sem_input)}, HF={len(hf_sem_input)}")
        # Still compare the overlap
        mn = min(len(cpp_sem_input), len(hf_sem_input))
        match = int(np.sum(cpp_sem_input[:mn] == hf_sem_input[:mn]))
        print(f"  First {mn}: {match}/{mn} match")

    # Coarse history for this window
    hist_start = max(0, total_gen - max_coarse_history)
    coarse_hist = cpp_coarse[hist_start:total_gen]
    print(f"  Coarse history: {len(coarse_hist)} tokens")

    # Full input = semantic_portion + infer_token + coarse_history
    cpp_full = len(cpp_sem_input) + 1 + len(coarse_hist)
    hf_full = len(hf_sem_input) + 1 + len(coarse_hist)
    print(f"  Full input: C++={cpp_full}, HF={hf_full}")
    print()
