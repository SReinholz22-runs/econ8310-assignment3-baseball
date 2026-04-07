"""
train.py — Train Faster R-CNN for baseball detection
Uses a ResNet-50 FPN backbone pretrained on COCO, fine-tuned for 1 class (baseball).

Usage:
    python train.py --data_dir data/ --epochs 10 --batch_size 2

Saves weights to: weights/model.pth
"""

import argparse
import os
import time

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import get_dataloader


# ── Model ───────────────────────────────────────────────────────────────────────

def build_model(num_classes=2):
    """
    Load Faster R-CNN pretrained on COCO and replace the classification head
    for our task.

    num_classes = 2 because torchvision detection models use:
        class 0 = background (required)
        class 1 = baseball

    Args:
        num_classes (int): Total classes including background.

    Returns:
        model (nn.Module): Ready-to-train Faster R-CNN.
    """
    # Load pretrained backbone — this downloads weights on first run (~160 MB)
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace the box predictor head with one sized for our classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ── Training loop ───────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Run one full pass over the dataset and return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = len(data_loader)

    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass — Faster R-CNN returns a dict of losses during training
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Skip batch if loss is nan
        if torch.isnan(losses):
            print(f"  [!] Skipping batch {batch_idx} — nan loss detected")
            optimizer.zero_grad()
            continue

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            print(
                f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{n_batches}] "
                f"Loss: {losses.item():.4f}"
            )

    return total_loss / n_batches


def train(data_dir, epochs, batch_size, lr, weight_decay, save_path):
    # ── Device ──────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────────
    print(f"\nLoading data from: {data_dir}")
    loader, dataset = get_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # increase if on Linux with many CPUs
    )
    print(f"Training on {len(dataset)} annotated frames across {len(loader)} batches per epoch.\n")

    # ── Model ───────────────────────────────────────────────────────────────────
    model = build_model(num_classes=2)
    model.to(device)

    # ── Optimizer ───────────────────────────────────────────────────────────────
    # Only update parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.0001,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler — reduces LR by 0.1 every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # ── Training ─────────────────────────────────────────────────────────────────
    print("Starting training...\n")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, optimizer, loader, device, epoch)
        lr_scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch}/{epochs}] — Avg Loss: {avg_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {elapsed:.1f}s\n"
        )

        # Save checkpoint if this is the best epoch so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_weights(model, save_path)
            print(f"  ✓ New best model saved to '{save_path}'\n")

    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Final weights saved to: {save_path}")


def _save_weights(model, save_path):
    """Save model state dict to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


# ── Entry point ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train baseball detector")
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Path to folder with paired .mov + .xml files (default: data/)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Frames per batch (default: 2)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.005,
        help="Learning rate (default: 0.005)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005,
        help="Weight decay for SGD (default: 0.0005)"
    )
    parser.add_argument(
        "--save_path", type=str, default="weights/model.pth",
        help="Where to save model weights (default: weights/model.pth)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_path=args.save_path,
    )
