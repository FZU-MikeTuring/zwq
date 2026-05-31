"""
准备论文所需数据集（152201226_张雯倩2）

流程:
  1. 下载 hagridv2_512.zip (119GB，已存在则跳过)
  2. 下载 annotations.zip
  3. 仅解压 7 个手势类
  4. 根据标注按 train/val/test 分集
  5. 可选抽样（复现论文每类 350 张）

用法:
  python prepare_dataset.py --data-dir ./hagrid_dataset
  python prepare_dataset.py --data-dir ./hagrid_dataset --samples 350
  python prepare_dataset.py --data-dir ./hagrid_dataset --skip-download
"""

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

# ── 论文 7 个手势类 ────────────────────────────────────
TARGET_CLASSES = ["palm", "fist", "ok", "like", "dislike", "one", "no_gesture"]

# ── 下载地址 ──────────────────────────────────────────
V2 = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/"
URL_512 = f"{V2}hagridv2_512.zip"
URL_ANN = f"{V2}annotations_with_landmarks/annotations.zip"

# ── ImageNet 标准归一化参数 ────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def download(url: str, dest: str, label: str) -> bool:
    """下载文件，支持 wget 断点续传，已存在则跳过"""
    dest = Path(dest)
    if dest.exists():
        gb = dest.stat().st_size / (1024**3)
        print(f"  [OK] {label} 已存在 ({gb:.1f} GB)，跳过")
        return True
    print(f"  [↓] 下载 {label} ...")
    print(f"      {url}")
    ret = os.system(f'wget -c "{url}" -O "{dest}"')
    if ret != 0:
        ret = os.system(f'curl -C - -L "{url}" -o "{dest}"')
    return ret == 0


def extract_zip_classes(zip_path: str, classes: list[str],
                         out_dir: str) -> dict[str, int]:
    """仅解压 ZIP 中指定类的图片，返回 {类名: 数量}"""
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  扫描 ZIP ({zip_path.stat().st_size / 1024**3:.1f} GB)...")

    counts = {c: 0 for c in classes}

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        # 探测每个目标类在 ZIP 中的目录前缀
        class_prefix = {}
        for cls in classes:
            candidates = set()
            for name in all_names:
                parts = Path(name).parts
                if cls in parts and name.lower().endswith((".jpg", ".jpeg", ".png")):
                    idx = parts.index(cls)
                    candidates.add("/".join(parts[:idx + 1]))
            if candidates:
                class_prefix[cls] = sorted(candidates, key=len)[0]

        if not class_prefix:
            # 扁平结构退化为文件名包含匹配
            print("  [!] 未检测到类子目录，使用文件名匹配...")
            for cls in classes:
                class_prefix[cls] = cls

        # 逐类解压
        for cls, prefix in class_prefix.items():
            cls_out = out_dir / cls
            cls_out.mkdir(parents=True, exist_ok=True)

            for name in all_names:
                if prefix not in name:
                    continue
                if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue

                filename = Path(name).name
                dest = cls_out / filename
                if dest.exists():
                    counts[cls] += 1
                    continue

                zf.extract(name, out_dir)
                extracted = out_dir / name
                if extracted.exists() and extracted != dest:
                    shutil.move(str(extracted), str(dest))
                counts[cls] += 1

                if counts[cls] % 5000 == 0:
                    print(f"    {cls}: {counts[cls]} 张...")

            print(f"    [OK] {cls}: {counts[cls]} 张")

    # 清理空目录
    for d in sorted(out_dir.rglob("*"), key=lambda p: len(str(p)), reverse=True):
        if d.is_dir() and d != out_dir and not any(d.iterdir()):
            d.rmdir()

    return counts


def split_dataset(raw_dir: str, ann_dir: str, out_dir: str,
                  classes: list[str], max_per_class: int | None = None,
                  seed: int = 42) -> dict:
    """根据标注 JSON 将图片按 train/val/test 分集（硬链接省空间）"""
    random.seed(seed)
    raw_dir, ann_dir, out_dir = Path(raw_dir), Path(ann_dir), Path(out_dir)

    stats = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        for cls in classes:
            dest_dir = out_dir / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)

            ann_file = ann_dir / split / f"{cls}.json"
            if not ann_file.exists():
                stats[split][cls] = 0
                continue

            with open(ann_file) as f:
                valid_ids = set(json.load(f).keys())

            raw_cls = raw_dir / cls
            if not raw_cls.exists():
                stats[split][cls] = 0
                continue

            matched = [img for img in raw_cls.iterdir()
                       if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                       and img.stem in valid_ids]

            if max_per_class and len(matched) > max_per_class:
                matched = random.sample(matched, max_per_class)

            n = 0
            for img in matched:
                dest = dest_dir / img.name
                if not dest.exists():
                    try:
                        os.link(img, dest)
                    except OSError:
                        shutil.copy2(img, dest)
                n += 1

            stats[split][cls] = n
            print(f"    [{split}/{cls}] {n} 张")

    return stats


def print_summary(stats: dict, max_per_class: int | None):
    print("\n" + "=" * 56)
    print("  数据集准备完成！")
    print("=" * 56)
    grand = 0
    for split in ["train", "val", "test"]:
        t = sum(stats[split].values())
        grand += t
        print(f"\n  [{split.upper()}] 共 {t} 张")
        for cls, cnt in stats[split].items():
            print(f"    {cls:<15} {cnt:>6} 张")
    print(f"\n  总计: {grand} 张 | 类别数: {len(TARGET_CLASSES)}")
    if max_per_class:
        print(f"  抽样上限: 每类每集 {max_per_class} 张")
    print(f"  ImageNet 归一化: mean={IMAGENET_MEAN}  std={IMAGENET_STD}")


def main():
    parser = argparse.ArgumentParser(description="准备论文手势识别数据集")
    parser.add_argument("--data-dir", default="./hagrid_dataset", help="数据根目录")
    parser.add_argument("--samples", type=int, default=None,
                        help="每类每集最多 N 张（论文用 350）")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_512 = data_dir / "hagridv2_512.zip"
    ann_zip = data_dir / "annotations.zip"
    raw_dir = data_dir / "raw_images"
    ann_dir = data_dir / "annotations"
    out_dir = data_dir / "organized"

    print("=" * 56)
    print("  论文数据集准备 — HaGRIDv2 512px")
    print(f"  目标类: {', '.join(TARGET_CLASSES)}")
    print("=" * 56)

    # ── 1. 下载 ──────────────────────────────────────
    if args.skip_download:
        print("\n[1/4] 跳过下载")
    else:
        print("\n[1/4] 下载文件")
        ok = download(URL_512, zip_512, "HaGRIDv2 512px") and \
             download(URL_ANN, ann_zip, "标注文件")
        if not ok:
            print("❌ 下载失败，检查网络后重试")
            sys.exit(1)

    # ── 2. 解压标注 ──────────────────────────────────
    print("\n[2/4] 解压标注")
    if not ann_zip.exists():
        print(f"❌ 缺少 {ann_zip}")
        sys.exit(1)
    with zipfile.ZipFile(ann_zip) as zf:
        zf.extractall(ann_dir)
    print(f"  [OK] → {ann_dir}")

    # ── 3. 选择性解压 7 类 ────────────────────────────
    print("\n[3/4] 解压目标手势类")
    if not zip_512.exists():
        print(f"❌ 缺少 {zip_512}")
        sys.exit(1)
    counts = extract_zip_classes(str(zip_512), TARGET_CLASSES, str(raw_dir))
    print(f"\n  共解压 {sum(counts.values())} 张")

    # ── 4. 按 train/val/test 分集 ─────────────────────
    print("\n[4/4] 按 train/val/test 分集")
    stats = split_dataset(str(raw_dir), str(ann_dir), str(out_dir),
                          TARGET_CLASSES, args.samples, args.seed)

    print_summary(stats, args.samples)

    # 提示
    print(f"\n  💡 最终数据: {out_dir}/")
    print(f"  💡 可删除: {raw_dir} (已硬链接到 organized)")
    print(f"  💡 可删除: {zip_512} (119GB 压缩包)")


if __name__ == "__main__":
    main()
