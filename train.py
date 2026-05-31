"""
模型训练脚本
- ImageNet 预训练权重初始化
- 前 5 轮冻结骨干 → 后 15 轮微调解冻
- AdamW + 余弦退火学习率 + 标签平滑
"""
import os
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import AMobileNetGesture
from config import (
    NUM_CLASSES, INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, NUM_EPOCHS, FREEZE_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING,
    GESTURE_LABELS, GESTURE_LABELS_CN,
)


def build_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()


def set_trainable(model: AMobileNetGesture, trainable: bool):
    for param in model.features.parameters():
        param.requires_grad = trainable
    for module in [model.eca, model.spatial, model.classifier]:
        for param in module.parameters():
            param.requires_grad = True


def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    losses, preds, targets = [], [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(outputs.argmax(1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch}/{total_epochs} | Batch {batch_idx}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    return np.mean(losses), accuracy_score(targets, preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        losses.append(criterion(outputs, labels).item())
        preds.extend(outputs.argmax(1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    return np.mean(losses), accuracy_score(targets, preds), targets, preds


def plot_curves(history: dict, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_acc"], "b-", label="Train", linewidth=1.5)
    ax1.plot(epochs, history["val_acc"], "r-", label="Val", linewidth=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Training & Validation Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_loss"], "b-", label="Train", linewidth=1.5)
    ax2.plot(epochs, history["val_loss"], "r-", label="Val", linewidth=1.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Training & Validation Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[OK] Training curves saved: {save_path}")


def plot_confusion_matrix(targets, preds, save_path: str):
    cm = confusion_matrix(targets, preds)
    # Use English class labels to avoid font issues
    labels_en = [GESTURE_LABELS[i] for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    fontsize=9, color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(labels_en, rotation=45, ha="right")
    ax.set_yticklabels(labels_en)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("A-MobileNet-HGR Confusion Matrix")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"[OK] Confusion matrix saved: {save_path}")


def train(train_dir: str, val_dir: str, test_dir: str,
          output_dir: str = "./output", device: str = None):
    os.makedirs(output_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. 数据加载 & 标签对齐 ----
    print("\n=== Loading Dataset ===")
    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(train=True))
    val_ds   = datasets.ImageFolder(val_dir,   transform=build_transforms(train=False))
    test_ds  = datasets.ImageFolder(test_dir,  transform=build_transforms(train=False))

    name_to_idx = {v: k for k, v in GESTURE_LABELS.items()}
    for ds in [train_ds, val_ds, test_ds]:
        remap = [name_to_idx[name] for name in ds.classes]
        ds.samples = [(path, remap[label]) for path, label in ds.samples]
        ds.targets = [remap[t] for t in ds.targets]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Classes (alphabetical): {train_ds.classes}")
    print(f"Classes (model order):  {[GESTURE_LABELS[i] for i in range(NUM_CLASSES)]}")

    # ---- 2. 模型 ----
    print("\n=== Building A-MobileNet-HGR ===")
    model = AMobileNetGesture(num_classes=NUM_CLASSES).to(device)
    print(f"Parameters: {model.param_count:.2f}M")

    # ---- 3. 损失函数 & 优化器 ----
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - FREEZE_EPOCHS)

    # ---- 4. 训练循环 ----
    print("\n=== Training Started ===")
    history = defaultdict(list)
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pth")
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        frozen = epoch <= FREEZE_EPOCHS
        set_trainable(model, trainable=not frozen)
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} "
              f"({'backbone frozen' if frozen else 'full fine-tuning'}) ---")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, NUM_EPOCHS)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        if not frozen:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2%} | LR: {lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  [OK] Best model saved (Val Acc: {best_val_acc:.2%})")

    print(f"\n=== Training Complete ({(time.time() - t_start):.1f}s) ===")
    print(f"Best val accuracy: {best_val_acc:.2%}")

    # 保存训练历史（后续可重新出图）
    import json
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(dict(history), f)

    # ---- 5. 测试 & 图表 ----
    test_and_plot(model, best_model_path, test_loader, criterion, device, output_dir, history)


def test_and_plot(model, model_path, test_loader, criterion, device, output_dir,
                  history: dict = None):
    """从已有模型权重评估测试集并生成图表"""
    import json as _json
    print("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_acc, test_targets, test_preds = evaluate(
        model, test_loader, criterion, device)

    print(f"Test Accuracy:  {test_acc:.2%}")
    print(f"Macro Precision: {precision_score(test_targets, test_preds, average='macro'):.2%}")
    print(f"Macro Recall:    {recall_score(test_targets, test_preds, average='macro'):.2%}")
    print(f"Macro F1:        {f1_score(test_targets, test_preds, average='macro'):.2%}")
    print(f"\n{classification_report(
        test_targets, test_preds,
        target_names=[GESTURE_LABELS[i] for i in range(NUM_CLASSES)]
    )}")

    # 训练曲线：优先用传入的 history，其次从文件加载
    if history is None:
        hist_path = os.path.join(output_dir, "history.json")
        history = _json.load(open(hist_path)) if os.path.exists(hist_path) else defaultdict(list)

    plot_curves(history, os.path.join(output_dir, "training_curves.png"))
    plot_confusion_matrix(test_targets, test_preds,
                          os.path.join(output_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 / 评估 A-MobileNet-HGR 模型")
    parser.add_argument("--train_dir", type=str, default="./dataset_zwq/hagrid_dataset/organized/train")
    parser.add_argument("--val_dir", type=str, default="./dataset_zwq/hagrid_dataset/organized/val")
    parser.add_argument("--test_dir", type=str, default="./dataset_zwq/hagrid_dataset/organized/test")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-only", type=str, default=None,
                        help="仅评估已有模型并重新生成图表（指定模型路径）")
    args = parser.parse_args()

    if args.eval_only:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        test_ds = datasets.ImageFolder(args.test_dir, transform=build_transforms(train=False))
        name_to_idx = {v: k for k, v in GESTURE_LABELS.items()}
        remap = [name_to_idx[name] for name in test_ds.classes]
        test_ds.samples = [(path, remap[label]) for path, label in test_ds.samples]
        test_ds.targets = [remap[t] for t in test_ds.targets]

        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=2, pin_memory=True)

        model = AMobileNetGesture(num_classes=NUM_CLASSES).to(device)
        criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
        test_and_plot(model, args.eval_only, test_loader, criterion, device, args.output_dir)
    else:
        train(args.train_dir, args.val_dir, args.test_dir, args.output_dir, args.device)
