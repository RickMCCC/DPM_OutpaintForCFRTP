"""
Prepare outpainting dataset from 3D grayscale volumes stacked from TIFF slices.

For each cubic block (volume_size^3) extracted from the volume, generate triplets
for each of the 8 corner sub-cubes (sub_size^3) as known context.

Output directory structure:
  samples_outpaint_3d/
    size{volume_size}_mask{sub_size}_source{source_tag}/
      init/        -> initial images (black background with corner sub-cube) for each slice
      full_mask/   -> mask images (unknown=255, known=0) for each slice
      target/      -> target images (full canvas) for each slice

Each 3D block is saved as separate PNG files (one per Z slice), following the
naming convention similar to prepare_outpainting_data.py.

Example:
  python scripts/prepare_outpainting_3d_data.py \
    --input_dir "C:/Terong/CFRTP XCT/1.576" \
    --output_dir "samples_outpaint_3d" \
    --volume_size 128 --sub_size 32
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 3D outpainting dataset")
    parser.add_argument("--input_dir", type=str, help="顺序 .tif 切片所在目录")
    parser.add_argument("--output_dir", type=str, help="输出数据根目录，默认samples_outpaint_3d")
    parser.add_argument("--volume_size", type=int, help="立方体画布边长，默认128")
    parser.add_argument("--stride_z", type=int, help="沿Z的步长，默认与volume_size相同")
    parser.add_argument("--stride_y", type=int, help="沿Y的步长，默认与volume_size相同")
    parser.add_argument("--stride_x", type=int, help="沿X的步长，默认与volume_size相同")
    parser.add_argument("--sub_size", type=int, help="角落子块边长（已知区域），默认32")
    parser.add_argument("--feather", type=int, help="mask边缘羽化半径(体素)，默认0")
    parser.add_argument("--limit_blocks", type=int, help="调试用：最多处理多少个立方体，默认不限制")
    parser.set_defaults(
        input_dir="C:/Terong/CFRTP XCT/nonwoven",
        output_dir="samples_outpaint_3d",
        volume_size=96,
        stride_z=None,
        stride_y=None,
        stride_x=None,
        sub_size=32,
        feather=0,
        limit_blocks=None,
    )
    return parser.parse_args()


def load_grayscale_8bit(img: Image.Image) -> np.ndarray:
    """Convert PIL image to uint8 grayscale array, handling 16-bit inputs."""

    if img.mode == "I;16":
        arr16 = np.array(img, dtype=np.int32)
        min_val = int(arr16.min())
        max_val = int(arr16.max())
        if max_val == min_val:
            return np.full(arr16.shape, 128, dtype=np.uint8)
        scaled = (arr16 - min_val) / max(1, (max_val - min_val))
        arr8 = (scaled * 255.0).astype(np.uint8)
        return arr8
    arr = np.array(img.convert("L"), dtype=np.uint8)
    return arr


def iter_blocks(
    tensor: np.ndarray,
    block_size: int,
    stride_z: int,
    stride_y: int,
    stride_x: int,
) -> Iterable[Tuple[Tuple[int, int, int], np.ndarray]]:
    depth, height, width = tensor.shape
    for z0 in range(0, depth - block_size + 1, stride_z):
        for y0 in range(0, height - block_size + 1, stride_y):
            for x0 in range(0, width - block_size + 1, stride_x):
                block = tensor[z0 : z0 + block_size, y0 : y0 + block_size, x0 : x0 + block_size]
                yield (z0, y0, x0), block


def corner_boxes_3d(size: int, sub_size: int) -> List[Tuple[int, int, int, int, int, int]]:
    assert 0 < sub_size <= size
    offsets = [0, size - sub_size]
    boxes: List[Tuple[int, int, int, int, int, int]] = []
    for z in offsets:
        for y in offsets:
            for x in offsets:
                boxes.append((z, y, x, z + sub_size, y + sub_size, x + sub_size))
    return boxes


def ensure_scipy() -> None:
    try:
        import scipy.ndimage  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("feather>0 需要安装 SciPy (pip install scipy)") from exc


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    ensure_scipy()
    from scipy.ndimage import gaussian_filter

    blurred = gaussian_filter(mask.astype(np.float32), sigma=radius)
    blurred = np.clip(blurred, 0.0, 1.0)
    return blurred.astype(np.float32)


def save_slice_png(path: Path, slice_arr: np.ndarray) -> None:
    """Save a single 2D slice as PNG"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if slice_arr.dtype != np.uint8:
        arr = np.clip(slice_arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = slice_arr
    image = Image.fromarray(arr, mode="L")
    image.save(path, "PNG")


def save_volume_slices(base_path: Path, volume: np.ndarray, slice_prefix: str = "") -> int:
    """
    Save 3D volume as separate PNG files, one per Z slice.
    
    Args:
        base_path: Base path without extension (will add _z{idx:03d}.png)
        volume: 3D array (depth, height, width)
        slice_prefix: Optional prefix for slice index (e.g., "slice")
    
    Returns:
        Number of slices saved
    """
    depth = volume.shape[0]
    base_stem = base_path.stem
    parent_dir = base_path.parent
    
    for z_idx in range(depth):
        slice_name = f"{base_stem}_{slice_prefix}{z_idx:03d}.png" if slice_prefix else f"{base_stem}_z{z_idx:03d}.png"
        slice_path = parent_dir / slice_name
        save_slice_png(slice_path, volume[z_idx])
    
    return depth


def load_volume_from_tiffs(tif_paths: Sequence[Path]) -> np.ndarray:
    slices = []
    for path in tqdm(tif_paths, desc="加载切片", leave=False):
        with Image.open(path) as img:
            arr = load_grayscale_8bit(img)
        slices.append(arr)
    volume = np.stack(slices, axis=0)
    return volume


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir_base = Path(args.output_dir)
    volume_size = int(args.volume_size)
    sub_size = int(args.sub_size)
    feather = int(args.feather)
    limit_blocks = None if args.limit_blocks is None else int(args.limit_blocks)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    tif_files = sorted(input_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"在 {input_dir} 未找到 .tif 文件")

    print(f"输入目录: {input_dir.resolve()}")
    print(f"读取 {len(tif_files)} 张切片，开始装载体数据...")

    volume = load_volume_from_tiffs(tif_files)

    depth, height, width = volume.shape
    print(f"体数据尺寸: (Z={depth}, Y={height}, X={width})")

    if any(size < volume_size for size in (depth, height, width)):
        raise ValueError(
            f"体尺寸 {volume.shape} 小于 volume_size {volume_size}，无法裁剪立方体"
        )

    if sub_size > volume_size:
        raise ValueError("sub_size 必须不大于 volume_size")

    stride_z = int(args.stride_z or volume_size)
    stride_y = int(args.stride_y or volume_size)
    stride_x = int(args.stride_x or volume_size)

    if stride_z <= 0 or stride_y <= 0 or stride_x <= 0:
        raise ValueError("stride_z/stride_y/stride_x 必须大于0")

    # 创建类似2D版本的目录结构: samples_outpaint_3d/size{volume_size}_mask{sub_size}_source{source_tag}/
    source_tag = input_dir.name
    scale_root = output_dir_base / f"size{volume_size}_mask{sub_size}_source{source_tag}"
    init_dir = scale_root / "init"
    full_mask_dir = scale_root / "full_mask"
    target_dir = scale_root / "target"

    scale_root.mkdir(parents=True, exist_ok=True)
    init_dir.mkdir(parents=True, exist_ok=True)
    full_mask_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {scale_root.resolve()}")

    corner_boxes = corner_boxes_3d(volume_size, sub_size)

    total_blocks = 0
    total_triplets = 0

    for (z0, y0, x0), block in tqdm(
        iter_blocks(volume, volume_size, stride_z, stride_y, stride_x),
        total=math.prod(
            [
                1 + (depth - volume_size) // stride_z,
                1 + (height - volume_size) // stride_y,
                1 + (width - volume_size) // stride_x,
            ]
        ),
        desc="裁剪立方体",
    ):
        total_blocks += 1
        block_id = f"z{z0:04d}_y{y0:04d}_x{x0:04d}"

        # 为每个立方体生成一个三元组，包含所有8个顶角子块（类似2D版本的outpaint_sample.py）
        # init: 黑色背景，粘贴所有8个角落的子块
        init_tensor = np.zeros_like(block, dtype=np.uint8)
        
        # mask: 未知区域=255，已知区域（所有8个角落子块）=0
        mask_tensor = np.ones(block.shape, dtype=np.float32)
        
        # 将所有8个角落的子块都添加到init和mask中
        for z1, y1, x1, z2, y2, x2 in corner_boxes:
            sub_patch = block[z1:z2, y1:y2, x1:x2]
            init_tensor[z1:z2, y1:y2, x1:x2] = sub_patch
            mask_tensor[z1:z2, y1:y2, x1:x2] = 0.0
        
        # 应用羽化（如果启用）
        if feather > 0:
            mask_tensor = feather_mask(mask_tensor, feather)
        # 转换为uint8: 未知区域=255，已知区域=0
        mask_tensor = (mask_tensor * 255.0).astype(np.uint8)

        # target: 完整的立方体
        target_tensor = block

        # 命名格式：{block_id}_c{volume_size}_s{sub_size}（不再包含corner索引）
        triplet_stem = f"{block_id}_c{volume_size}_s{sub_size}"

        # 为每个Z切片单独保存（不包含扩展名，函数会自动添加_z{idx:03d}.png）
        base_path_init = init_dir / triplet_stem
        base_path_mask = full_mask_dir / triplet_stem
        base_path_target = target_dir / triplet_stem

        save_volume_slices(base_path_init, init_tensor)
        save_volume_slices(base_path_mask, mask_tensor)
        save_volume_slices(base_path_target, target_tensor)
        
        total_triplets += 1

        if limit_blocks is not None and total_blocks >= limit_blocks:
            print(f"达到 limit_blocks={limit_blocks}，提前结束")
            break

    total_files = total_triplets * 3 * volume_size  # 每个三元组有3个类型，每个类型有volume_size个切片

    print(f"完成：处理了 {total_blocks} 个立方体块")
    print(f"  - 生成了 {total_triplets} 个三元组（每个立方体包含所有8个角落子块）")
    print(f"  - 共保存 {total_files} 个PNG文件（init/full_mask/target，每个立方体×{volume_size}个切片）")


if __name__ == "__main__":
    main()


