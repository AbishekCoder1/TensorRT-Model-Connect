#!/usr/bin/env python3
"""Check if torch.multinomial consumes different RNG amounts for different tensor sizes."""
import torch

# Create a fixed probability distribution with 50 non-zero entries
base_probs = torch.softmax(torch.randn(50), dim=0)

for size in [50, 100, 1000, 10001, 10048, 20000]:
    p = torch.zeros(size)
    p[:50] = base_probs

    torch.manual_seed(42)
    state_before = torch.random.get_rng_state().clone()
    tok = torch.multinomial(p.unsqueeze(0), 1)
    state_after = torch.random.get_rng_state().clone()

    # Count RNG consumption
    consumed = -1
    torch.random.set_rng_state(state_before)
    for n in range(1, 500):
        torch.empty(1).uniform_()
        if torch.equal(torch.random.get_rng_state(), state_after):
            consumed = n
            break

    print(f"size={size:6d}: token={tok.item():4d}, RNG consumed={consumed}")
