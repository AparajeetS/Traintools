"""PTDB-1 CIFAR-10 ResNet-18 development shard."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "benchmark_config.json"
PROTOCOL_SHA256 = "5b01d1ca47b6ebca6601a9debc80b0edfb262c3ee6feeb8e0cc998b0ee92821e"
PROTOCOL_REPOSITORY_PATH = "benchmarks/ptdb_v1/PROTOCOL.md"
FROZEN_CONFIG = {
    "protocol": "PTDB-1",
    "phase": "development_base",
    "shard": "cifar10_resnet18",
    "public_package": "traintools==0.6.2",
    "epochs": 25,
    "batch_size": 128,
    "num_workers": 2,
    "amp": True,
    "download": True,
    "output_dir": "/kaggle/working/ptdb_v1_cifar10_resnet18",
    "cells": [
        {"dataset": "cifar10", "model": model, "seed": seed, "regime": regime}
        for model in ("resnet18",)
        for seed in (11, 23, 37)
        for regime in ("clean", "symmetric", "class_conditional")
    ],
}


def ensure_public_dependencies() -> None:
    if os.environ.get("PTDB_USE_LOCAL") == "1":
        repo = HERE.parents[4]
        sys.path.insert(0, str(repo))
        return
    try:
        import traintools
        if getattr(traintools, "__version__", None) == "0.6.2":
            return
    except Exception:
        pass
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "traintools==0.6.2"]
    )


ensure_public_dependencies()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

import traintools
from traintools import AUMTracker, EL2NTracker, ExampleDynamicsTracker
from traintools.earlyguard import TrainGuard
from traintools.gradnoise import estimate_gns
from traintools.plasticity import PlasticityProbe


FROZEN_SPLIT_SEED = 20260718
NOISE_RATE = 0.20
FULL_DEVELOPMENT_RUNS = 81
FULL_EPOCHS = 25


def cuda_preflight(required: bool) -> dict[str, Any]:
    info = {
        "available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": None,
        "capability": None,
    }
    if not torch.cuda.is_available():
        if required:
            raise RuntimeError(f"PTDB-1 requires a working CUDA device; environment={info}")
        return info
    info["device"] = torch.cuda.get_device_name(0)
    info["capability"] = list(torch.cuda.get_device_capability(0))
    try:
        probe = torch.ones(16, device="cuda")
        float(probe.square().sum().item())
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"CUDA preflight failed before dataset setup; environment={info}") from exc
    return info


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_frozen_config() -> dict[str, Any]:
    if CONFIG_PATH.is_file():
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if canonical_json(config) != canonical_json(FROZEN_CONFIG):
            raise RuntimeError("adjacent benchmark_config.json does not match embedded frozen config")
    return json.loads(json.dumps(FROZEN_CONFIG))


def stable_int(value: Any, bits: int = 32) -> int:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).digest()
    return int.from_bytes(digest[: bits // 8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class IndexedNoisyDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        indices: Iterable[int],
        labels: list[int],
        n_classes: int,
        regime: str,
        noise_seed: int,
    ) -> None:
        self.base = base
        self.indices = list(indices)
        self.clean_labels = [int(labels[i]) for i in self.indices]
        self.observed_labels = list(self.clean_labels)
        self.corrupted = [False] * len(self.indices)
        if regime != "clean":
            rng = np.random.default_rng(noise_seed)
            count = int(round(NOISE_RATE * len(self.indices)))
            chosen = rng.choice(len(self.indices), size=count, replace=False)
            for pos in chosen.tolist():
                original = self.clean_labels[pos]
                if regime == "symmetric":
                    draw = int(rng.integers(0, n_classes - 1))
                    replacement = draw + int(draw >= original)
                elif regime == "class_conditional":
                    replacement = (original + 1) % n_classes
                else:
                    raise ValueError(f"unknown regime: {regime}")
                self.observed_labels[pos] = replacement
                self.corrupted[pos] = True

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        source_index = self.indices[position]
        image, _ = self.base[source_index]
        return (
            source_index,
            image,
            self.observed_labels[position],
            self.clean_labels[position],
            self.corrupted[position],
        )


class IndexedCleanDataset(Dataset):
    def __init__(self, base: Dataset, indices: Iterable[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        source_index = self.indices[position]
        image, label = self.base[source_index]
        return source_index, image, int(label), int(label), False


class SmokeDataset(Dataset):
    def __init__(self, size: int, n_classes: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(size, 3, 32, 32, generator=generator)
        self.y = torch.randint(0, n_classes, (size,), generator=generator)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        label = int(self.y[index])
        return index, self.x[index], label, label, False


def dataset_spec(name: str) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    specs = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100),
        "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970), 10),
    }
    return specs[name]


def stage_kaggle_dataset(name: str, data_root: Path) -> bool:
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return False
    if name == "cifar10":
        source = next((path for path in input_root.rglob("cifar-10-batches-py") if path.is_dir()), None)
        target = data_root / "cifar-10-batches-py"
        if (target / "data_batch_1").is_file() and (target / "test_batch").is_file():
            return True
        if source is not None:
            shutil.copytree(source, target, dirs_exist_ok=True)
            return True
    elif name == "cifar100":
        target = data_root / "cifar-100-python"
        required = ("meta", "train", "test")
        if all((target / filename).is_file() for filename in required):
            return True
        source = next(
            (path.parent for path in input_root.rglob("meta") if all((path.parent / filename).is_file() for filename in required)),
            None,
        )
        if source is not None:
            target.mkdir(parents=True, exist_ok=True)
            for filename in required:
                shutil.copy2(source / filename, target / filename)
            return True
    elif name == "svhn":
        required = ("train_32x32.mat", "test_32x32.mat")
        if all((data_root / filename).is_file() for filename in required):
            return True
        source = next((path.parent for path in input_root.rglob("train_32x32.mat") if (path.parent / "test_32x32.mat").is_file()), None)
        if source is not None:
            for filename in required:
                shutil.copy2(source / filename, data_root / filename)
            return True
    return False


def make_datasets(name: str, regime: str, seed: int, data_root: Path, smoke: bool):
    if smoke:
        return SmokeDataset(192, 10, seed), SmokeDataset(96, 10, seed + 1), SmokeDataset(96, 10, seed + 2), 10

    mean, std, n_classes = dataset_spec(name)
    staged = stage_kaggle_dataset(name, data_root)
    train_ops = [transforms.RandomCrop(32, padding=4)]
    if name != "svhn":
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_transform = transforms.Compose(train_ops)
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if name.startswith("cifar"):
        cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        train_base = cls(data_root, train=True, transform=train_transform, download=not staged)
        eval_base = cls(data_root, train=True, transform=eval_transform, download=not staged)
        test_base = cls(data_root, train=False, transform=eval_transform, download=not staged)
        labels = [int(v) for v in train_base.targets]
    elif name == "svhn":
        train_base = datasets.SVHN(data_root, split="train", transform=train_transform, download=not staged)
        eval_base = datasets.SVHN(data_root, split="train", transform=eval_transform, download=not staged)
        test_base = datasets.SVHN(data_root, split="test", transform=eval_transform, download=not staged)
        labels = [int(v) for v in train_base.labels]
    else:
        raise ValueError(name)

    split_rng = np.random.default_rng(FROZEN_SPLIT_SEED + stable_int(name))
    order = split_rng.permutation(len(labels)).tolist()
    val_size = 5000 if name.startswith("cifar") else 7000
    val_indices = order[:val_size]
    train_indices = order[val_size:]
    noise_seed = stable_int({"dataset": name, "regime": regime, "seed": seed})
    train_set = IndexedNoisyDataset(train_base, train_indices, labels, n_classes, regime, noise_seed)
    val_set = IndexedCleanDataset(eval_base, val_indices)
    test_set = IndexedCleanDataset(test_base, range(len(test_base)))
    return train_set, val_set, test_set, n_classes


class WideBasic(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.shortcut = nn.Identity() if stride == 1 and in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + self.shortcut(x)


class WideResNet28x2(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        widths = [16, 32, 64, 128]
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        self.group1 = self._group(widths[0], widths[1], 4, 1)
        self.group2 = self._group(widths[1], widths[2], 4, 2)
        self.group3 = self._group(widths[2], widths[3], 4, 2)
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=False)
        self.fc = nn.Linear(widths[3], n_classes)

    @staticmethod
    def _group(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [WideBasic(in_ch, out_ch, stride)]
        layers.extend(WideBasic(out_ch, out_ch, 1) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.group3(self.group2(self.group1(self.stem(x))))
        x = self.relu(self.bn(x)).mean(dim=(2, 3))
        return self.fc(x)


class TinyViT(nn.Module):
    def __init__(self, n_classes: int, dim: int = 192, depth: int = 6, heads: int = 3) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 65, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        return self.head(self.norm(self.encoder(x)[:, 0]))


def make_model(name: str, n_classes: int) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=None, num_classes=n_classes)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if name == "wrn28_2":
        return WideResNet28x2(n_classes)
    if name == "vit_tiny":
        return TinyViT(n_classes)
    raise ValueError(name)


def make_loaders(train_set, val_set, test_set, batch_size: int, workers: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, **common)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for _, x, y, _, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(1) == y).sum())
        count += y.numel()
    return loss_sum / max(1, count), correct / max(1, count)


def diagnostic_scores(aum, el2n, dynamics, loss_sum, loss_count, corruption, run_id: str):
    ids = sorted(corruption)
    rows = []
    for example_id in ids:
        a = aum.examples.get(example_id)
        e = el2n.examples.get(example_id)
        d = dynamics.examples.get(example_id)
        rng = np.random.default_rng(stable_int({"run_id": run_id, "id": example_id}))
        rows.append({
            "run_id": run_id,
            "example_id": example_id,
            "corrupted": int(corruption[example_id]),
            "aum": a.aum if a else math.nan,
            "el2n": e.score if e else math.nan,
            "forgetting": d.forgetting_events if d else math.nan,
            "mean_confidence": d.mean_confidence if d else math.nan,
            "mean_loss": loss_sum.get(example_id, 0.0) / max(1, loss_count.get(example_id, 0)),
            "random_score": float(rng.random()),
        })
    return rows


def ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([row["corrupted"] for row in rows], dtype=int)
    if y.min() == y.max():
        return {}
    prevalence = float(y.mean())
    k = max(1, int(round(prevalence * len(y))))
    definitions = {
        "aum": lambda r: -float(r["aum"]),
        "el2n": lambda r: float(r["el2n"]),
        "forgetting": lambda r: float(r["forgetting"]),
        "mean_confidence": lambda r: -float(r["mean_confidence"]),
        "mean_loss": lambda r: float(r["mean_loss"]),
        "random": lambda r: float(r["random_score"]),
    }
    result: dict[str, Any] = {"prevalence": prevalence}
    for name, getter in definitions.items():
        score = np.asarray([getter(row) for row in rows], dtype=float)
        order = np.argsort(-score, kind="stable")
        result[f"{name}_auroc"] = float(roc_auc_score(y, score))
        result[f"{name}_average_precision"] = float(average_precision_score(y, score))
        result[f"{name}_precision_at_noise"] = float(y[order[:k]].mean())
    return result


def measure_plasticity(model: nn.Module, batch, device: torch.device, step: int) -> dict[str, Any]:
    probe = PlasticityProbe(model, max_samples=512)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(batch[1].to(device, non_blocking=True))
    result = probe.measure(step=step)
    probe.remove_hooks()
    if was_training:
        model.train()
    dead = float(np.mean([layer.dead_fraction for layer in result.layers])) if result.layers else 0.0
    erank = float(np.mean([layer.feature_erank for layer in result.layers])) if result.layers else 1.0
    return {"plasticity_score": result.global_score, "mean_dead_fraction": dead, "mean_feature_erank": erank}


def run_cell(cell: dict[str, Any], config: dict[str, Any], instrumented: bool, smoke: bool):
    dataset_name = cell["dataset"]
    model_name = cell["model"]
    seed = int(cell["seed"])
    regime = cell["regime"]
    mode = "instrumented" if instrumented else "plain"
    run_id = f"{config['phase']}__{dataset_name}__{model_name}__s{seed}__{regime}__{mode}"
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("/tmp/ptdb_data") if Path("/kaggle").exists() else HERE / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    train_set, val_set, test_set, n_classes = make_datasets(dataset_name, regime, seed, data_root, smoke)
    workers = 0 if smoke else int(config["num_workers"])
    batch_size = 32 if smoke else int(config["batch_size"])
    train_loader, val_loader, test_loader = make_loaders(train_set, val_set, test_set, batch_size, workers, seed)

    model = make_model(model_name, n_classes).to(device)
    initial_hash = state_hash(model)
    if model_name == "vit_tiny":
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    epochs = 1 if smoke else int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    aum = AUMTracker(min_observations=1) if instrumented else None
    el2n = EL2NTracker(min_observations=1) if instrumented else None
    dynamics = ExampleDynamicsTracker(min_observations=1) if instrumented else None
    guard = TrainGuard(min_improvement=1e-3, patience_steps=10_000, horizon_steps=max(1, len(train_loader)), warmup_records=2) if instrumented else None
    loss_sum: dict[int, float] = {}
    loss_count: dict[int, int] = {}
    corruption = {int(train_set.indices[pos]) if hasattr(train_set, "indices") else pos: bool(value) for pos, value in enumerate(getattr(train_set, "corrupted", [False] * len(train_set)))}
    epoch_rows = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        model.train()
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        grad_norm_sum = 0.0
        grad_norm_count = 0
        last_batch = None
        for step, batch in enumerate(train_loader):
            ids, x, y, _, _ = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                losses = F.cross_entropy(logits, y, reduction="none")
                loss = losses.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
            grad_norm_sum += float(grad_norm)
            grad_norm_count += 1
            scaler.step(optimizer)
            scaler.update()

            batch_n = y.numel()
            train_loss_sum += float(losses.detach().sum())
            train_correct += int((logits.detach().argmax(1) == y).sum())
            train_count += batch_n
            if instrumented:
                cpu_ids = [int(v) for v in ids]
                aum.update(cpu_ids, logits, y, step=epoch * len(train_loader) + step)
                el2n.update(cpu_ids, logits, y, step=epoch * len(train_loader) + step)
                dynamics.update(cpu_ids, logits, y, step=epoch * len(train_loader) + step)
                for example_id, value in zip(cpu_ids, losses.detach().cpu().tolist()):
                    loss_sum[example_id] = loss_sum.get(example_id, 0.0) + float(value)
                    loss_count[example_id] = loss_count.get(example_id, 0) + 1
            last_batch = batch

        scheduler.step()
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        diagnostic: dict[str, Any] = {}
        if instrumented and last_batch is not None:
            np.random.seed(stable_int({"run_id": run_id, "epoch": epoch}))
            guard.record(step=(epoch + 1) * len(train_loader), val_loss=val_loss)
            decision = guard.evaluate()
            diagnostic.update({
                "guard_should_stop": int(decision.should_stop),
                "guard_predicted_improvement": decision.predicted_improvement,
                "guard_ci_low": decision.confidence_interval[0],
                "guard_ci_high": decision.confidence_interval[1],
            })
            diagnostic.update(measure_plasticity(model, last_batch, device, epoch + 1))
            gns_x = last_batch[1].to(device, non_blocking=True)
            gns_y = last_batch[2].to(device, non_blocking=True)
            gns = estimate_gns(model, F.cross_entropy, gns_x, gns_y, n_splits=2, step=epoch + 1)
            diagnostic.update({"gns": gns.gns, "critical_batch": gns.critical_batch, "gns_regime": gns.regime})

        epoch_rows.append({
            "run_id": run_id,
            "epoch": epoch + 1,
            "train_loss": train_loss_sum / max(1, train_count),
            "train_accuracy": train_correct / max(1, train_count),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "mean_grad_norm": grad_norm_sum / max(1, grad_norm_count),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
            **diagnostic,
        })

    test_loss, test_accuracy = evaluate(model, test_loader, device)
    elapsed = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    example_rows = diagnostic_scores(aum, el2n, dynamics, loss_sum, loss_count, corruption, run_id) if instrumented else []
    ranking = ranking_metrics(example_rows) if example_rows else {}
    summary = {
        "run_id": run_id,
        "phase": config["phase"],
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "regime": regime,
        "instrumented": int(instrumented),
        "epochs": epochs,
        "train_examples": len(train_set),
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_state_hash": initial_hash,
        "final_state_hash": state_hash(model),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "elapsed_seconds": elapsed,
        "seconds_per_epoch": elapsed / epochs,
        "peak_vram_bytes": int(peak_vram),
        **ranking,
    }
    return summary, epoch_rows, example_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = load_frozen_config()
    smoke = os.environ.get("PTDB_SMOKE") == "1"
    output = Path(config["output_dir"] if not smoke else HERE / "smoke_output")
    output.mkdir(parents=True, exist_ok=True)
    preflight = cuda_preflight(required=not smoke)
    manifest = {
        "protocol": config["protocol"],
        "phase": config["phase"],
        "confirmatory": True,
        "pilot_outputs_may_tune_thresholds": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_repository_path": PROTOCOL_REPOSITORY_PATH,
        "config_sha256": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
        "script_sha256": sha256_file(Path(__file__)),
        "started_at_unix": time.time(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "traintools": getattr(traintools, "__version__", "unknown"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_preflight": preflight,
            "traintools_install": "pip --no-deps (preserve platform torch)",
        },
        "config": config,
    }
    (output / "manifest.started.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "frozen_protocol_pointer.json").write_text(
        json.dumps({"sha256": PROTOCOL_SHA256, "repository_path": PROTOCOL_REPOSITORY_PATH}, indent=2),
        encoding="utf-8",
    )

    summaries: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if smoke and os.environ.get("PTDB_SMOKE_ALL") != "1":
        cells = config["cells"][:1]
    elif smoke:
        cells = [config["cells"][0], config["cells"][1], config["cells"][2]]
    else:
        cells = config["cells"]
    for cell in cells:
        paired_twin = cell["regime"] == "clean" and cell["model"] in {"resnet18", "wrn28_2"}
        modes = (False, True) if paired_twin else (True,)
        for instrumented in modes:
            run_id = f"{cell['dataset']}__{cell['model']}__s{cell['seed']}__{cell['regime']}__{'instrumented' if instrumented else 'plain'}"
            print(f"\n=== PTDB-1 {run_id} ===", flush=True)
            try:
                summary, epoch_rows, example_rows = run_cell(cell, config, instrumented, smoke)
                summaries.append(summary)
                epochs.extend(epoch_rows)
                examples.extend(example_rows)
            except Exception as exc:
                errors.append({"run_id": run_id, "error": repr(exc), "traceback": traceback.format_exc()})
                print(errors[-1]["traceback"], flush=True)
            write_csv(output / "run_summary.partial.csv", summaries)
            write_csv(output / "epoch_metrics.partial.csv", epochs)
            (output / "errors.partial.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    write_csv(output / "run_summary.csv", summaries)
    write_csv(output / "epoch_metrics.csv", epochs)
    write_gzip_csv(output / "example_scores.csv.gz", examples)
    (output / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    instrumented = [row for row in summaries if row["instrumented"]]
    plain_by_cell = {(r["dataset"], r["model"], r["seed"], r["regime"]): r for r in summaries if not r["instrumented"]}
    pairs = []
    for row in instrumented:
        key = (row["dataset"], row["model"], row["seed"], row["regime"])
        plain = plain_by_cell.get(key)
        if plain:
            pairs.append({
                "cell": "__".join(map(str, key)),
                "exact_final_hash": int(plain["final_state_hash"] == row["final_state_hash"]),
                "accuracy_difference": row["test_accuracy"] - plain["test_accuracy"],
                "loss_difference": row["test_loss"] - plain["test_loss"],
                "runtime_ratio": row["elapsed_seconds"] / max(1e-9, plain["elapsed_seconds"]),
            })
    write_csv(output / "paired_noninterference.csv", pairs)

    seconds_per_epoch = [row["seconds_per_epoch"] for row in instrumented]
    projected_core_hours = (float(np.mean(seconds_per_epoch)) * FULL_DEVELOPMENT_RUNS * FULL_EPOCHS / 3600.0) if seconds_per_epoch else None
    projection = {
        "completed_executions": len(summaries),
        "error_rows": len(errors),
        "instrumented_seconds_per_epoch_mean": float(np.mean(seconds_per_epoch)) if seconds_per_epoch else None,
        "instrumented_seconds_per_epoch_max": float(np.max(seconds_per_epoch)) if seconds_per_epoch else None,
        "projected_81_run_core_gpu_hours": projected_core_hours,
        "projection_is_timing_only": True,
        "exact_hash_pairs": sum(row["exact_final_hash"] for row in pairs),
        "paired_cells": len(pairs),
    }
    (output / "execution_summary.json").write_text(json.dumps(projection, indent=2), encoding="utf-8")
    manifest["finished_at_unix"] = time.time()
    manifest["outputs"] = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(projection, indent=2), flush=True)
    if errors:
        raise SystemExit(f"PTDB-1 development shard completed with {len(errors)} error rows")


if __name__ == "__main__":
    main()
