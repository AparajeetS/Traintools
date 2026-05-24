"""
demo.py -- runs all three traintools diagnostics on a toy MLP trained on noisy MNIST.

Demonstrates:
  - GNS estimation: tells you if batch size 64 is optimal
  - PlasticityProbe: tracks dead neurons and weight effective rank
  - TrainGuard: predicts when training plateaus and issues stop signal

Run: python demo.py
Requirements: torch, torchvision, scipy, matplotlib
"""

import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from traintools.callbacks.pytorch import TraintoolsTracker


# -- Model ---------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden=256, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


# -- Data ----------------------------------------------------------------------

def load_mnist(n_train=2000, n_val=500, noise_frac=0.0):
    ds = torchvision.datasets.MNIST(
        root="/tmp/mnist", train=True, download=True,
        transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    )
    idx = list(range(len(ds)))
    random.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]

    if noise_frac > 0:
        n_noisy = int(noise_frac * n_train)
        for i in train_idx[:n_noisy]:
            ds.targets[i] = random.randint(0, 9)

    train_dl = DataLoader(Subset(ds, train_idx), batch_size=64, shuffle=True)
    val_dl   = DataLoader(Subset(ds, val_idx),   batch_size=256)
    return train_dl, val_dl


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            total_loss += criterion(out, y).item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            n          += len(y)
    model.train()
    return total_loss / n, correct / n


# -- Training ------------------------------------------------------------------

def main():
    print("traintools demo -- MLP on MNIST (n_train=2000, label_noise=30%)\n")

    torch.manual_seed(42)
    random.seed(42)

    model     = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_dl, val_dl = load_mnist(n_train=2000, n_val=500, noise_frac=0.3)

    tracker = TraintoolsTracker(
        model=model,
        loss_fn=criterion,
        gns_freq=50,          # estimate GNS every 50 steps
        plasticity_freq=50,   # measure plasticity every 50 steps
        earlyguard=True,
        min_improvement=1e-4,
        patience_steps=300,
        horizon_steps=100,
        verbose=True,
    )

    global_step = 0
    val_loss    = None
    stop        = False

    for epoch in range(20):
        for x, y in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            global_step += 1

            # Pass val_loss only when freshly computed (end of each epoch);
            # otherwise pass None so TrainGuard doesn't double-record.
            decision = tracker.step(
                step=global_step,
                inputs=x,
                targets=y,
                val_loss=val_loss,  # None most steps; set once per epoch below
            )
            val_loss = None  # clear after handing off

            if decision is not None and decision.should_stop:
                print(f"\n[demo] TrainGuard says STOP at step {global_step} "
                      f"(epoch {epoch+1}).")
                stop = True
                break

        if stop:
            break

        # Evaluate once per epoch and store for next batch-step hand-off
        vl, va = evaluate(model, val_dl, criterion)
        val_loss = vl
        print(f"Epoch {epoch+1:02d}  val_loss={vl:.4f}  val_acc={va:.3f}")

    print("\n--- Final summary ---")
    print(tracker.gns_history.summary())

    if tracker.plasticity_history.scores:
        ps = tracker.plasticity_history.scores
        trend = "DEGRADED" if tracker.plasticity_history.is_degrading() else "stable"
        print(f"Plasticity: initial={ps[0]:.3f}  final={ps[-1]:.3f}  {trend}")

    # Save plots
    try:
        fig = tracker.plot()
        fig.savefig("traintools_demo.png", dpi=120, bbox_inches="tight")
        print("Plots saved to traintools_demo.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
