"""
Training Script for Brain Tumor Classification using Vision Mamba.
==================================================================

Usage:
    python train.py                          # Train with default config
    python train.py --epochs 30              # Override epochs
    python train.py --variant vim_small      # Use Vim-Small instead of Tiny
    python train.py --batch_size 16          # Smaller batch for limited GPU

On Google Colab:
    !python train.py --epochs 25 --variant vim_tiny
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

import config
from models.classifier import BrainTumorVim
from utils.data_loader import get_data_loaders
from utils.metrics import compute_metrics, plot_confusion_matrix
from utils.visualization import plot_training_curves, visualize_predictions


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="  Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_amp):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc="  Validating", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        if use_amp and device.type == "cuda":
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """Cosine annealing scheduler with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults["lr"],
                   0.5 * (1.0 + np.cos(np.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(args):
    """Main training function."""
    print("=" * 70)
    print("  Brain Tumor Classification using Vision Mamba (Vim)")
    print("=" * 70)

    # Seed
    set_seed(config.SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Data
    print("\n" + "-" * 70)
    print("  Loading Dataset...")
    print("-" * 70)
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    print("-" * 70)
    print("  Building Model...")
    print("-" * 70)
    model = BrainTumorVim(
        num_classes=config.NUM_CLASSES,
        variant=args.variant,
        img_size=config.IMG_SIZE,
        drop_rate=0.1,
    )
    model.summary()
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=args.epochs,
        min_lr=config.MIN_LR,
    )
    scaler = GradScaler() if args.use_amp else None

    # Training history
    history = {
        "train_acc": [], "val_acc": [],
        "train_loss": [], "val_loss": [],
        "lr": [],
    }

    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    print("\n" + "=" * 70)
    print("  Starting Training...")
    print("=" * 70)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n  Epoch [{epoch}/{args.epochs}]  LR: {current_lr:.6f}")
        print("  " + "-" * 50)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.use_amp
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, args.use_amp)

        # Step scheduler
        scheduler.step()

        # Record history
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Time: {epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc + config.MIN_DELTA:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "variant": args.variant,
                "history": history,
            }, best_model_path)
            print(f"  ★ Best model saved! (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n  Early stopping triggered after {epoch} epochs.")
                break

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  Training Complete!")
    print(f"  Total Time:       {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Best Model:        {best_model_path}")
    print("=" * 70)

    # ---- Plot training curves ----
    plot_training_curves(history)

    # ---- Evaluate on test set ----
    if test_loader is not None:
        print("\n" + "=" * 70)
        print("  Evaluating on Test Set...")
        print("=" * 70)

        # Load best model
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="  Testing"):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Metrics
        metrics = compute_metrics(all_labels, all_preds, class_names)

        # Confusion matrix
        plot_confusion_matrix(all_labels, all_preds, class_names)

        # Prediction visualization
        visualize_predictions(model, test_loader, class_names, device)

        # Save metrics to JSON
        metrics_save = {k: v for k, v in metrics.items() if k != "classification_report"}
        metrics_save["best_val_acc"] = best_val_acc
        metrics_save["total_training_time_seconds"] = total_time
        metrics_save["variant"] = args.variant
        metrics_path = os.path.join(config.RESULTS_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_save, f, indent=2)
        print(f"\n  Metrics saved to: {metrics_path}")

    # Save training history
    history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved to: {history_path}")

    print("\n  All results saved in:", config.RESULTS_DIR)
    print("  Done! ✓\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain Tumor Classifier with Vision Mamba")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--variant", type=str, default=config.MODEL_VARIANT,
                        choices=["vim_tiny", "vim_small"], help="Model variant")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--use_amp", action="store_true", default=config.USE_AMP, help="Use mixed precision")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")

    args = parser.parse_args()
    if args.no_amp:
        args.use_amp = False

    main(args)
