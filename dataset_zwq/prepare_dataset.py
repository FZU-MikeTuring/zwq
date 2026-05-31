"""
准备论文所需数据集（152201226_张雯倩2）

流程:
  1. 下载 hagridv2_512.zip (119GB，支持多线程并行 + 断点续传)
  2. 下载 annotations.zip
  3. 仅解压 7 个手势类
  4. 根据标注按 train/val/test 分集
  5. 可选抽样（复现论文每类 350 张）

用法:
  python prepare_dataset.py --data-dir ./hagrid_dataset
  python prepare_dataset.py --data-dir ./hagrid_dataset --samples 350
  python prepare_dataset.py --data-dir ./hagrid_dataset --skip-download
  python prepare_dataset.py --data-dir ./hagrid_dataset --workers 16  # 16线程并行下载
"""

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import sys
import time
import urllib.request
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


def download_parallel(url: str, dest: str, label: str, workers: int = 8) -> bool:
    """多线程分块并行下载，支持断点续传。

    使用 HTTP Range 请求将文件分成 workers 块同时下载，
    每个块独立续传（已完成的分块自动跳过）。
    """
    dest = Path(dest)

    if dest.exists():
        gb = dest.stat().st_size / (1024**3)
        print(f"  [OK] {label} 已存在 ({gb:.1f} GB)，跳过")
        return True

    print(f"  [↓] 并行下载 {label}（{workers} 线程）...")
    print(f"      {url}")

    # ── 获取文件总大小 ──
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_size = int(resp.headers["Content-Length"])

    gb = total_size / (1024**3)
    print(f"      总大小: {gb:.1f} GB，每块约 {gb/workers:.1f} GB")

    # ── 分块临时目录 ──
    temp_dir = Path(str(dest) + ".parts")
    temp_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = total_size // workers
    downloaded_bytes = [0] * workers
    lock = None  # 不用锁，每个线程写自己的 chunk

    def download_chunk(i: int) -> tuple[int, bool]:
        """下载第 i 个分块，返回 (chunk_index, success)"""
        start = i * chunk_size
        end = total_size - 1 if i == workers - 1 else (i + 1) * chunk_size - 1
        part_file = temp_dir / f"part_{i:04d}"
        expected_size = end - start + 1

        # 检查已完成的 chunk
        if part_file.exists() and part_file.stat().st_size == expected_size:
            downloaded_bytes[i] = expected_size
            return (i, True)

        # 下载（最多重试 3 次）
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with open(part_file, "wb") as f:
                        while True:
                            buf = resp.read(4 * 1024 * 1024)  # 4MB buffer
                            if not buf:
                                break
                            f.write(buf)
                            downloaded_bytes[i] = part_file.stat().st_size

                # 校验大小
                if part_file.stat().st_size == expected_size:
                    return (i, True)
                else:
                    print(f"    [!] chunk {i+1} 大小不匹配，重试...")
                    part_file.unlink(missing_ok=True)

            except Exception as e:
                if attempt < 2:
                    print(f"    [!] chunk {i+1} 出错，{3-attempt-1}次重试: {e}")
                    time.sleep(2)
                else:
                    print(f"    [✗] chunk {i+1} 失败: {e}")
                    return (i, False)

        return (i, False)

    # ── 进度报告线程 ──
    def progress_reporter():
        last_total = 0
        while True:
            time.sleep(3)
            total_dl = sum(downloaded_bytes)
            if total_dl > 0:
                pct = total_dl / total_size * 100
                speed = (total_dl - last_total) / 3 / (1024**2)
                eta = (total_size - total_dl) / max((total_dl - last_total) / 3, 1)
                eta_h = eta / 3600
                print(f"      {pct:.1f}% | {total_dl/1024**3:.1f}/{gb:.1f} GB"
                      f" | {speed:.1f} MB/s | 剩余 ~{eta_h:.1f}h")
                last_total = total_dl
            if total_dl >= total_size:
                break

    # ── 启动下载 ──
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        progress_future = executor.submit(progress_reporter)
        chunk_futures = {executor.submit(download_chunk, i): i
                         for i in range(workers)}

        all_ok = True
        for future in concurrent.futures.as_completed(chunk_futures):
            i, ok = future.result()
            if not ok:
                all_ok = False
                print(f"    [✗] 分块 {i+1}/{workers} 下载失败")

    # 等待进度线程结束
    progress_future.cancel()

    if not all_ok:
        print(f"  ✗ 部分分块下载失败，请重试（已完成的分块会自动复用）")
        return False

    # ── 合并分块 ──
    print(f"  [↻] 合并 {workers} 个分块...")
    with open(dest, "wb") as out:
        for i in range(workers):
            part_file = temp_dir / f"part_{i:04d}"
            with open(part_file, "rb") as f:
                while True:
                    buf = f.read(8 * 1024 * 1024)  # 8MB read buffer
                    if not buf:
                        break
                    out.write(buf)

    # ── 校验 & 清理 ──
    final_size = dest.stat().st_size
    if final_size == total_size:
        shutil.rmtree(temp_dir)
        print(f"  [✓] {label} 下载完成 ({final_size/1024**3:.1f} GB)")
        return True
    else:
        print(f"  ✗ 合并后大小不匹配 ({final_size} vs {total_size})")
        return False


def download_simple(url: str, dest: str, label: str) -> bool:
    """下载小文件（wget/curl），已存在则跳过"""
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
    parser.add_argument("--split-only", action="store_true",
                        help="仅重新分集（需先完成过完整流程）")
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="并行下载线程数（默认 8）")
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
    print(f"  下载线程: {args.workers}")
    print("=" * 56)

    if args.split_only:
        # 仅重新分集，跳过下载和解压
        print("\n[跳过 1-3] 仅重新分集")
        if not raw_dir.exists() or not ann_dir.exists():
            print("❌ 缺少 raw_images/ 或 annotations/，请先完成完整流程")
            sys.exit(1)
    else:
        # ── 1. 下载 ──────────────────────────────────
        if args.skip_download:
            print("\n[1/4] 跳过下载")
        else:
            print("\n[1/4] 下载文件")
            ok = download_parallel(URL_512, zip_512, "HaGRIDv2 512px",
                                   workers=args.workers)
            ok = ok and download_simple(URL_ANN, ann_zip, "标注文件")
            if not ok:
                print("❌ 下载失败，检查网络后重试（已完成的分块会自动复用）")
                sys.exit(1)

        # ── 2. 解压标注 ──────────────────────────────
        print("\n[2/4] 解压标注")
        if not ann_zip.exists():
            print(f"❌ 缺少 {ann_zip}")
            sys.exit(1)
        with zipfile.ZipFile(ann_zip) as zf:
            zf.extractall(ann_dir)
        # 处理 zip 内可能多套的一层 annotations/ 目录
        nested = ann_dir / "annotations"
        if nested.is_dir() and not (ann_dir / "train").is_dir():
            for item in list(nested.iterdir()):
                shutil.move(str(item), str(ann_dir / item.name))
            nested.rmdir()
        print(f"  [OK] → {ann_dir}")

        # ── 3. 选择性解压 7 类 ────────────────────────
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
