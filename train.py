"""FashionMNIST training with PyTorch DDP, CSV tracking, and checkpoints.

Usage (single process):
    python train.py --epochs 5 --batch-size 128 --lr 1e-3 --tracker csv
Usage (two GPUs on Linux/Kaggle):
    torchrun --standalone --nproc_per_node=2 train.py --epochs 5 --batch-size 128 --lr 1e-3 --tracker csv
"""

import argparse
import csv
import os
from pathlib import Path
import random
import re

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    """Basic fully connected FashionMNIST classifier (28x28 -> 10 classes)."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """Return unnormalised class scores (logits) for each image."""
        return self.linear_relu_stack(self.flatten(x))


def parse_args():
    parser = argparse.ArgumentParser(description="FashionMNIST PyTorch distributed training (CSV tracker)")
    parser.add_argument("--epochs", type=int, default=5, help="Total epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-process batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--tracker", choices=["csv"], default="csv", help="Experiment tracking (CSV only)")
    parser.add_argument("--optimiser", choices=["SGD", "Adam"], default="SGD", help="Optimiser (default: SGD)")
    parser.add_argument("--run-name", default="fashionmnist_ddp", help="Output filename prefix")
    parser.add_argument("--data-dir", default="data", help="FashionMNIST dataset root")
    parser.add_argument("--resume", default=None, help="Checkpoint path for resuming")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional quick smoke test limit")
    parser.add_argument("--max-test-batches", type=int, default=None, help="Optional quick smoke test limit")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not (0 < args.lr < float("inf")):
        parser.error("--lr must be a finite positive number")
    for key in ("max_train_batches", "max_test_batches"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be at least 1")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name) or args.run_name in (".", ".."):
        parser.error("--run-name may contain letters, digits, _, - and . only")
    return args


def setup_distributed():
    """Initialise torchrun workers; choose CUDA/NCCL or CPU/Gloo."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if local_rank >= gpu_count:
            raise RuntimeError(f"LOCAL_RANK={local_rank} but only {gpu_count} CUDA GPU(s) visible")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl" if os.name != "nt" else "gloo"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1:
        dist.init_process_group(backend=backend, init_method="env://")

    print(f"WORKER rank={rank} local_rank={local_rank} world_size={world_size} device={device} backend={backend}", flush=True)
    return device, rank, world_size, local_rank



def get_data(args, rank, world_size, device):
    """Load FashionMNIST and distribute training data across workers."""

    transform = transforms.ToTensor()
    data_dir = args.data_dir

    # Rank 0 downloads the dataset
    if rank == 0:
        datasets.FashionMNIST(
            data_dir, train=True, download=True,
            transform=transform
        )
        datasets.FashionMNIST(
            data_dir, train=False, download=True,
            transform=transform
        )

    if world_size > 1:
        dist.barrier()

    train_data = datasets.FashionMNIST(
        data_dir, train=True, download=False,
        transform=transform
    )

    # Divide training data between GPUs
    train_sampler = (
        DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        if world_size > 1 else None
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Evaluate on rank 0 only
    test_loader = None

    if rank == 0:
        test_data = datasets.FashionMNIST(
            data_dir, train=False, download=False,
            transform=transform
        )

        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda")
        )

    return train_loader, train_sampler, test_loader

 

def train_one_epoch(
    loader, model, loss_fn, optimizer,
    device, world_size, max_batches=None
):
    """Train one epoch and aggregate metrics across GPU workers."""

    model.train()

    totals = torch.zeros(3, dtype=torch.float64, device=device)

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        totals[0] += loss.detach().item() * labels.size(0)
        totals[1] += (logits.argmax(dim=1) == labels).sum().item()
        totals[2] += labels.size(0)

    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    if totals[2].item() == 0:
        raise RuntimeError("No training batches were processed.")

    avg_loss = totals[0].item() / totals[2].item()
    accuracy = 100 * totals[1].item() / totals[2].item()

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(loader, raw_model, loss_fn, device, max_batches=None):
    """Evaluate the original model on rank 0."""
    raw_model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = raw_model(images)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise RuntimeError("No evaluation batches were processed.")

    return total_loss / total, 100 * correct / total


def main():
    args = parse_args()
    torch.manual_seed(42)
    random.seed(42)
    device, rank, world_size, local_rank = setup_distributed()
    try:
        if rank == 0:
            print(f"CONFIG epochs={args.epochs} batch_size={args.batch_size} "
                  f"lr={args.lr} optimiser={args.optimiser} tracker={args.tracker} "
                  f"world_size={world_size} run_name={args.run_name}", flush=True)
        train_loader, sampler, test_loader = get_data(args, rank, world_size, device)
        raw_model = NeuralNetwork().to(device)
        model = DDP(raw_model, device_ids=[local_rank] if device.type == "cuda" else None) if world_size > 1 else raw_model
        loss_fn = nn.CrossEntropyLoss()
        optimizer_class = torch.optim.Adam if args.optimiser == "Adam" else torch.optim.SGD
        optimizer = optimizer_class(model.parameters(), lr=args.lr)
        start_epoch = 0

        if args.resume:
            # Loading a user-controlled pickle checkpoint should only be done from trusted files.
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            raw_model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"])
            if rank == 0:
                print(f"RESUME completed_epochs={start_epoch}", flush=True)

        results_dir = Path("results")
        checkpoints_dir = Path("checkpoints")
        csv_path = results_dir / f"{args.run_name}.csv"
        if rank == 0:
            results_dir.mkdir(exist_ok=True)
            checkpoints_dir.mkdir(exist_ok=True)
            if start_epoch == 0:
                with csv_path.open("w", newline="", encoding="utf-8") as fp:
                    csv.writer(fp).writerow(["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy", "lr", "optimiser", "world_size"])
            elif not csv_path.exists():
                raise FileNotFoundError(f"Resume CSV missing: {csv_path}")

        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            train_loss, train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer,
                                                    device, world_size, args.max_train_batches)
            # All ranks finish training / reduction before rank 0 evaluates.
            if rank == 0:
                test_loss, test_acc = evaluate(test_loader, raw_model, loss_fn, device, args.max_test_batches)
                with csv_path.open("a", newline="", encoding="utf-8") as fp:
                    csv.writer(fp).writerow([epoch + 1, f"{train_loss:.6f}", f"{train_acc:.3f}",
                                             f"{test_loss:.6f}", f"{test_acc:.3f}", args.lr,
                                             args.optimiser, world_size])
                torch.save({"epoch": epoch + 1, "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(), "world_size": world_size},
                           checkpoints_dir / f"{args.run_name}_latest.pt")
                print(f"EPOCH {epoch+1}/{args.epochs} train_loss={train_loss:.4f} "
                      f"train_accuracy={train_acc:.2f}% test_loss={test_loss:.4f} "
                      f"test_accuracy={test_acc:.2f}%", flush=True)
            if world_size > 1:
                dist.barrier()
        if rank == 0:
            print(f"COMPLETED results={csv_path} checkpoint={checkpoints_dir / (args.run_name + '_latest.pt')}", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"ERROR rank={rank}: CUDA out of memory; reduce --batch-size (e.g., 32), then retry.", flush=True)
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
          