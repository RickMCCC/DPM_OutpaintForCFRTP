"""
将 PNG 切片数据集预加载并缓存为本地 .pt 文件。

每个 block 的 768 个 PNG 文件（init/mask/target 各 256 张切片）
合并为单个 .pt 文件，训练时直接 torch.load()，消除网络 I/O 瓶颈。

用法:
  python scripts/cache_dataset.py \
    --source "Z:/CFRTP XCT/processed/size256_mask64_sourcel2-1.485" \
    --cache "data/cache/size256_mask64_sourcel2-1.485"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from scripts.outpainting_3d_dataset import sanitize_triplet


def parse_args():
    parser = argparse.ArgumentParser(description="缓存 PNG 切片数据集为 .pt 文件")
    parser.add_argument("--source", type=str, required=True,
                        help="源数据集目录（包含 init/full_mask/target 子目录）")
    parser.add_argument("--cache", type=str, default="data/cache/outpaint_3d",
                        help="缓存输出目录，默认 data/cache/outpaint_3d")
    return parser.parse_args()


def load_slice_minus1_1(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr)


def load_slice_0_1(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def load_volume(source_dir: Path, base_name: str, load_fn) -> torch.Tensor:
    """加载某个 block 的所有 Z 切片并堆叠为 (D, H, W) tensor"""
    pattern = f"{base_name}_z*.png"
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到切片: {source_dir / pattern}")
    slices = [load_fn(f) for f in files]
    return torch.stack(slices, dim=0)


def main():
    args = parse_args()
    source_root = Path(args.source)
    cache_root = Path(args.cache)

    init_dir = source_root / "init"
    mask_dir = source_root / "full_mask"
    target_dir = source_root / "target"

    for d in [init_dir, mask_dir, target_dir]:
        if not d.is_dir():
            raise FileNotFoundError(f"目录不存在: {d}")

    # 收集所有 base_name
    init_files = list(init_dir.glob("*_z*.png"))
    base_names = sorted(set(
        "_".join(f.stem.split("_")[:-1]) for f in init_files
    ))

    # 验证三元组完整性
    valid_names = []
    for name in base_names:
        n_init = len(list(init_dir.glob(f"{name}_z*.png")))
        n_mask = len(list(mask_dir.glob(f"{name}_z*.png")))
        n_target = len(list(target_dir.glob(f"{name}_z*.png")))
        if n_init == n_mask == n_target:
            valid_names.append(name)
        else:
            print(f"跳过不完整的 block: {name} (init={n_init}, mask={n_mask}, target={n_target})")

    print(f"找到 {len(valid_names)} 个完整的 block，开始缓存...")
    print(f"  源目录: {source_root.resolve()}")
    print(f"  缓存目录: {cache_root.resolve()}")

    cache_root.mkdir(parents=True, exist_ok=True)

    for name in tqdm(valid_names, desc="缓存 block"):
        init = load_volume(init_dir, name, load_slice_minus1_1)   # (D,H,W) in [-1,1]
        mask = load_volume(mask_dir, name, load_slice_0_1)         # (D,H,W) in [0,1]
        target = load_volume(target_dir, name, load_slice_minus1_1)  # (D,H,W) in [-1,1]

        # 添加 channel 维度: (1, D, H, W)
        triplet = sanitize_triplet(
            init.unsqueeze(0),
            mask.unsqueeze(0),
            target.unsqueeze(0),
        )
        # Preserve the source PNG precision while using one quarter of the
        # previous float32 cache size. Unknown init voxels become exact zero
        # again when sanitize_triplet is applied during loading.
        data = {
            "init": ((triplet["init"] + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8),
            "mask": triplet["mask"].to(torch.uint8),
            "target": ((triplet["target"] + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8),
        }
        torch.save(data, cache_root / f"{name}.pt")

    # 保存 manifest
    manifest = {
        "schema_version": 2,
        "storage_dtype": "uint8",
        "source": str(source_root.resolve()),
        "num_blocks": len(valid_names),
        "depth": init.shape[0] if valid_names else 0,
        "names": valid_names,
    }
    with open(cache_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 计算缓存大小
    total_size = sum(
        (cache_root / f"{n}.pt").stat().st_size for n in valid_names
    )
    print(f"\n缓存完成: {len(valid_names)} 个 block, 总大小 {total_size / 1024**3:.1f} GB")
    print(f"缓存目录: {cache_root.resolve()}")


if __name__ == "__main__":
    main()
