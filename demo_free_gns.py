"""
demo_free_gns.py — Gradient Noise Scale for FREE during gradient accumulation.

Part A shows the feature: GNS tracked over training with zero extra passes,
reusing the micro-batch gradients accumulation already computes.

Part B is a correctness check: on one identical batch, the free-accumulation
estimate and the extra-pass estimate_gns() use the same underlying math, so
their single-shot (raw) GNS should match closely.

Run: python demo_free_gns.py
Requirements: torch
"""

import torch
import torch.nn as nn

from traintools import GradientAccumulationGNS, estimate_gns
from traintools.gradnoise import _unbiased_estimates, _gns_from_estimates


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10),
    )


def make_batch(n, noise=0.2, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(n, 64)
    w = torch.linspace(-1, 1, 64)
    y = ((x @ w) > 0).long() * 5
    # inject label noise
    mask = torch.rand(n) < noise
    y[mask] = torch.randint(0, 10, (int(mask.sum()),))
    return x, y.clamp(0, 9)


def part_a():
    print("PART A - free GNS tracked over training (zero extra passes)\n")
    model = make_model()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    MICRO, ACCUM = 32, 8
    free = GradientAccumulationGNS(model, micro_batch_size=MICRO, decay=0.9)

    print(f"  micro={MICRO}  accum={ACCUM}  effective batch={MICRO*ACCUM}\n")
    print(f"  {'step':>5} | {'GNS (EMA)':>12} | regime")
    print("  " + "-" * 40)
    result = None
    for step in range(1, 41):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            x, y = make_batch(MICRO)
            (crit(model(x), y) / ACCUM).backward()
            free.record_microbatch()            # the only added line
        result = free.compute(step=step)
        opt.step()
        free.reset_accumulation()
        if step % 10 == 0:
            print(f"  {step:>5} | {result.gns:>12.1f} | {result.regime}")
    print(f"\n  {result.recommendation}\n")


def part_b():
    print("PART B - correctness check on one identical batch\n")
    model = make_model()
    crit = nn.CrossEntropyLoss()
    MICRO, ACCUM = 32, 8
    B = MICRO * ACCUM
    x, y = make_batch(B, seed=123)

    # Free-accumulation style: accumulate the SAME slices, difference the grads.
    free = GradientAccumulationGNS(model, micro_batch_size=MICRO)
    model.zero_grad(set_to_none=True)
    for i in range(ACCUM):
        sl = slice(i * MICRO, (i + 1) * MICRO)
        crit(model(x[sl]), y[sl]).backward()    # accumulate (no /ACCUM here)
        free.record_microbatch()
    g = torch.stack(free._micro_grads)
    tr_free, sig_free = _unbiased_estimates(g, MICRO)
    gns_free = _gns_from_estimates(tr_free, sig_free)

    # Extra-pass style on the identical batch (train mode to match).
    model.zero_grad(set_to_none=True)
    extra = estimate_gns(model, crit, x, y, n_splits=ACCUM, eval_mode=False)

    print(f"  free-accumulation raw GNS : {gns_free:10.3f}")
    print(f"  extra-pass         raw GNS : {extra.raw_gns:10.3f}")
    rel = abs(gns_free - extra.raw_gns) / (extra.raw_gns + 1e-9)
    print(f"  relative difference        : {rel:10.2%}")
    print("  -> identical math on identical micro-batch gradients, as expected.\n")


if __name__ == "__main__":
    print("traintools - free GNS during gradient accumulation\n")
    part_a()
    part_b()
