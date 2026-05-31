"""
模型训练脚本
实现论文3.5节描述的迁移学习训练策略：
- ImageNet 预训练权重初始化
- 前5轮冻结骨干 → 后15轮微调解冻
- AdamW + 余弦退火学习率
- 标签平滑交叉熵损失
"""
import os
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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
    """构建数据增强和预处理变换"""
    if train:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def set_trainable(model: AMobileNetGesture, trainable: bool):
    """设置骨干网络是否可训练"""
    for param in model.features.parameters():
        param.requires_grad = trainable
    # 注意力模块和分类头始终可训练
    for param in model.eca.parameters():
        param.requires_grad = True
    for param in model.spatial.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    """训练一个 epoch"""
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

    acc = accuracy_score(targets, preds)
    avg_loss = np.mean(losses)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    losses, preds, targets = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        losses.append(loss.item())
        preds.extend(outputs.argmax(1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    avg_loss = np.mean(losses)
    return avg_loss, acc, targets, preds


def plot_curves(history: dict, save_path: str):
    """绘制训练曲线（论文图4风格）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # 准确率曲线
    ax1.plot(epochs, history["train_acc"], "b-", label="训练准确率", linewidth=1.5)
    ax1.plot(epochs, history["val_acc"], "r-", label="验证准确率", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("准确率")
    ax1.set_title("训练与验证准确率变化")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 损失曲线
    ax2.plot(epochs, history["train_loss"], "b-", label="训练损失", linewidth=1.5)
    ax2.plot(epochs, history["val_loss"], "r-", label="验证损失", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("训练与验证损失变化")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Training curves saved: {save_path}")


def plot_confusion_matrix(targets, preds, save_path: str):
    """绘制混淆矩阵"""
    cm = confusion_matrix(targets, preds)
    labels_cn = [GESTURE_LABELS_CN[i] for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")

    # 标注数值
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    fontsize=9, color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(labels_cn, rotation=45, ha="right")
    ax.set_yticklabels(labels_cn)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("A-MobileNet-HGR 混淆矩阵")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix saved: {save_path}")


def train(data_dir: str, output_dir: str = "./output", device: str = None):
    """
    完整训练流程

    Args:
        data_dir: 数据集目录（ImageFolder 格式，每类一个子文件夹）
        output_dir: 输出目录
        device: 训练设备
    """
    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- 1. 数据加载 ----
    print("\n=== Loading Dataset ===")
    train_transform = build_transforms(train=True)
    val_transform = build_transforms(train=False)

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    # 8:1:1 划分
    total = len(full_dataset)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    # 验证集和测试集使用不含数据增强的变换
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"Total: {total} | Train: {train_size} | Val: {val_size} | Test: {test_size}")

    # ---- 2. 模型初始化 ----
    print("\n=== Building A-MobileNet-HGR Model ===")
    model = AMobileNetGesture(num_classes=NUM_CLASSES).to(device)
    print(f"Parameters: {model.param_count:.2f}M")

    # ---- 3. 损失函数 & 优化器 ----
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - FREEZE_EPOCHS
    )

    # ---- 4. 训练循环 ----
    print("\n=== Training Started ===")
    history = defaultdict(list)
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # 前 FREEZE_EPOCHS 轮冻结骨干网络
        if epoch <= FREEZE_EPOCHS:
            set_trainable(model, trainable=False)
            print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} (backbone frozen) ---")
        else:
            set_trainable(model, trainable=True)
            print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} (full model fine-tuning) ---")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, NUM_EPOCHS
        )

        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # 调整学习率（仅在微调阶段）
        if epoch > FREEZE_EPOCHS:
            scheduler.step()

        # 记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2%} | LR: {lr_now:.2e}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  [OK] Best model saved (Val Acc: {best_val_acc:.2%})")

    t_total = time.time() - t_start
    print(f"\n=== Training Complete ({t_total:.1f}s) ===")
    print(f"Best val accuracy: {best_val_acc:.2%}")

    # ---- 5. 最终测试 ----
    print("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_targets, test_preds = evaluate(
        model, test_loader, criterion, device
    )

    precision = precision_score(test_targets, test_preds, average="macro")
    recall = recall_score(test_targets, test_preds, average="macro")
    f1 = f1_score(test_targets, test_preds, average="macro")

    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Macro Precision: {precision:.2%}")
    print(f"Macro Recall: {recall:.2%}")
    print(f"Macro F1: {f1:.2%}")
    print(f"\nClassification Report:\n{classification_report(
        test_targets, test_preds,
        target_names=[GESTURE_LABELS[i] for i in range(NUM_CLASSES)]
    )}")

    # ---- 6. 保存图表 ----
    plot_curves(history, os.path.join(output_dir, "training_curves.png"))
    plot_confusion_matrix(test_targets, test_preds, os.path.join(output_dir, "confusion_matrix.png"))

    # ---- 7. 导出最终模型 ----
    final_path = os.path.join(output_dir, "amobilenet_hgr_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[OK] Final model saved: {final_path}")

    return model, history


def create_dummy_dataset(data_dir: str):
    """
    创建一个演示用的小型数据集结构（供测试流程用）
    实际使用时请替换为真实数据集（如 HaGRID）
    """
    os.makedirs(data_dir, exist_ok=True)
    labels = list(GESTURE_LABELS.values())

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        # 为每个类别生成一些纯色占位图（仅用于验证代码流程）
        for i in range(50):
            color = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            from PIL import Image
            img = Image.fromarray(color)
            img.save(os.path.join(label_dir, f"{i:04d}.jpg"))

    print(f"[OK] Dummy dataset created: {data_dir}")
    print("  Note: This is for testing the training pipeline only. Use real gesture data for actual training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 A-MobileNet-HGR 模型")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="数据集目录 (ImageFolder 格式)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="输出目录")
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备 (cuda/cpu)")
    parser.add_argument("--create_dummy", action="store_true",
                        help="创建演示数据集（仅用于测试）")
    args = parser.parse_args()

    if args.create_dummy:
        create_dummy_dataset(args.data_dir)

    train(args.data_dir, args.output_dir, args.device)
