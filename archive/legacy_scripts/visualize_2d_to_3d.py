"""
将2D截面图转化为3D立体图的可视化脚本

功能：
- 从init/target/example文件夹加载2D切片，组装成3D体积
- 支持训练数据格式和生成样本格式
- 如果是init，将空白部分（黑色）和mask(size32)之外的部分变透明
- 如果是init，用虚线连接体积块外部相邻的端点（12条边）

使用方法:

1. 训练数据格式（prepare_outpainting_3d_data.py生成的）:
   python scripts/visualize_2d_to_3d.py \
     --input_folder "samples_outpaint_3d/size96_mask32_source1.576/init" \
     --sample_name "z0000_y0000_x0000_c96_s32" \
     --output_path "visualizations/init_3d.png"

2. 生成样本格式（outpaint_3d_train.py生成的）:
   python scripts/visualize_2d_to_3d.py \
     --input_folder "samples_outpaint_3d_train/step_5000" \
     --sample_index 0 \
     --data_type "sample" \
     --output_path "visualizations/sample_0_3d.png"
"""

import os
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from typing import Tuple, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))

# 默认参数
DEFAULT_INPUT_FOLDER = r"C:\Terong\improved-diffusion\samples_outpaint_3d\size96_mask32_source1.576\target"
DEFAULT_SAMPLE_NAME = "z0000_y0000_x0000_c96_s32"
DEFAULT_OUTPUT_DIR = r"C:\Terong\improved-diffusion\scripts\visualizations"


def corner_boxes_3d(size: int, sub_size: int) -> List[Tuple[int, int, int, int, int, int]]:
    """
    计算3D立方体中8个角落子块的位置
    返回: [(z1, y1, x1, z2, y2, x2), ...] 共8个
    """
    assert 0 < sub_size <= size
    offsets = [0, size - sub_size]
    boxes: List[Tuple[int, int, int, int, int, int]] = []
    for z in offsets:
        for y in offsets:
            for x in offsets:
                boxes.append((z, y, x, z + sub_size, y + sub_size, x + sub_size))
    return boxes


def get_cube_edges(size: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """
    获取立方体的12条边（连接外部相邻的端点）
    
    Args:
        size: (D, H, W) 立方体的尺寸
    
    Returns:
        12条边的列表，每条边是((z1, y1, x1), (z2, y2, x2))
    """
    D, H, W = size
    
    # 立方体的8个顶点坐标
    vertices = [
        (0, 0, 0),      # 0: (z_min, y_min, x_min)
        (0, 0, W),      # 1: (z_min, y_min, x_max)
        (0, H, 0),      # 2: (z_min, y_max, x_min)
        (0, H, W),      # 3: (z_min, y_max, x_max)
        (D, 0, 0),      # 4: (z_max, y_min, x_min)
        (D, 0, W),      # 5: (z_max, y_min, x_max)
        (D, H, 0),      # 6: (z_max, y_max, x_min)
        (D, H, W),      # 7: (z_max, y_max, x_max)
    ]
    
    # 立方体的12条边（连接相邻的顶点）
    edges = [
        (vertices[0], vertices[1]),  # 底面前边
        (vertices[0], vertices[2]),  # 底面左边
        (vertices[0], vertices[4]),  # 前面左边
        (vertices[1], vertices[3]),  # 底面右边
        (vertices[1], vertices[5]),  # 前面右边
        (vertices[2], vertices[3]),  # 底面后边
        (vertices[2], vertices[6]),  # 后面左边
        (vertices[3], vertices[7]),  # 后面右边
        (vertices[4], vertices[5]),  # 顶面前边
        (vertices[4], vertices[6]),  # 顶面左边
        (vertices[5], vertices[7]),  # 顶面右边
        (vertices[6], vertices[7]),  # 顶面后边
    ]
    
    return edges


def load_volume_from_slices(
    input_folder: str, 
    sample_name: Optional[str] = None, 
    sample_index: Optional[int] = None, 
    data_type: str = "sample"
) -> np.ndarray:
    """
    从文件夹加载所有Z切片，组装成3D体积
    
    支持两种格式：
    1. 训练数据格式：{sample_name}_z*.png（如 z0000_y0000_x0000_c96_s32_z000.png）
    2. 生成样本格式：{data_type}_{sample_index}_z*.png（如 sample_0_z000.png）
    
    Args:
        input_folder: 包含切片的文件夹路径（如 init/, target/, 或 step_5000/）
        sample_name: 样本基础名称（训练数据格式，不含_z{idx}.png后缀）
        sample_index: 样本索引（生成样本格式，如 0, 1, 2...）
        data_type: 数据类型（生成样本格式，如 "sample", "target", "init", "mask"）
    
    Returns:
        3D numpy数组，形状为 (D, H, W)，dtype=uint8
    """
    folder = Path(input_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在: {folder}")
    
    # 确定使用哪种格式
    if sample_index is not None:
        # 生成样本格式：{data_type}_{sample_index}_z*.png
        pattern = f"{data_type}_{sample_index}_z*.png"
        slice_files = sorted(folder.glob(pattern))
    elif sample_name:
        # 训练数据格式：{sample_name}_z*.png
        pattern = f"{sample_name}_z*.png"
        slice_files = sorted(folder.glob(pattern))
    else:
        # 尝试自动检测：查找所有匹配的z*.png文件
        pattern = f"sample_*_z*.png"
        slice_files = sorted(folder.glob(pattern))
        if not slice_files:
            pattern = f"*_z*.png"
            slice_files = sorted(folder.glob(pattern))
    
    if not slice_files:
        raise FileNotFoundError(f"未找到匹配的切片文件。尝试的模式: {pattern} in {folder}")
    
    # 只加载前96张切片
    slice_files = slice_files[:96]
    
    # 加载切片
    slices = []
    for slice_file in slice_files:
        img = Image.open(slice_file).convert("L")
        arr = np.array(img, dtype=np.uint8)
        slices.append(arr)
    
    # 堆叠成3D体积: (D, H, W)
    volume = np.stack(slices, axis=0)
    return volume


def parse_volume_size_from_name(sample_name: str) -> Tuple[int, int]:
    """
    从样本名称解析volume_size和sub_size
    格式: z0000_y0000_x0000_c{volume_size}_s{sub_size}
    
    Returns:
        (volume_size, sub_size)
    """
    parts = sample_name.split('_')
    volume_size = None
    sub_size = None
    
    for part in parts:
        if part.startswith('c'):
            volume_size = int(part[1:])
        elif part.startswith('s'):
            sub_size = int(part[1:])
    
    if volume_size is None or sub_size is None:
        raise ValueError(f"无法从样本名称解析尺寸: {sample_name}")
    
    return volume_size, sub_size


def create_mask_region_mask(volume_size: int, sub_size: int, volume_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    创建mask区域的掩码（8个角落子块的并集）
    
    Args:
        volume_size: 立方体边长
        sub_size: 角落子块边长
        volume_shape: 体积的实际形状 (D, H, W)
    
    Returns:
        布尔掩码，True表示在mask区域内
    """
    D, H, W = volume_shape
    mask = np.zeros((D, H, W), dtype=bool)
    
    # 获取8个角落子块的位置
    corner_boxes = corner_boxes_3d(volume_size, sub_size)
    
    # 标记所有mask区域
    for z1, y1, x1, z2, y2, x2 in corner_boxes:
        # 确保索引在范围内
        z1 = max(0, min(z1, D))
        z2 = max(0, min(z2, D))
        y1 = max(0, min(y1, H))
        y2 = max(0, min(y2, H))
        x1 = max(0, min(x1, W))
        x2 = max(0, min(x2, W))
        mask[z1:z2, y1:y2, x1:x2] = True
    
    return mask


def visualize_3d_volume(
    volume: np.ndarray,
    output_path: Optional[str] = None,
    is_init: bool = False,
    volume_size: Optional[int] = None,
    sub_size: Optional[int] = None,
    threshold: int = 5,  # 低于此值的像素视为空白
    alpha: float = 1.0,  # 不透明度（默认1.0，不透明）
):
    """
    可视化3D体积
    
    Args:
        volume: 3D数组，形状 (D, H, W)
        output_path: 输出图片路径
        is_init: 是否为init数据（需要处理透明度）
        volume_size: 立方体边长（用于计算mask区域）
        sub_size: 角落子块边长（用于计算mask区域）
        threshold: 空白阈值（低于此值的像素视为空白，设为透明）
        alpha: 不透明度（0-1，默认1.0，不透明）
    """
    D, H, W = volume.shape
    
    # 创建3D图形，设置背景透明
    fig = plt.figure(figsize=(14, 12), facecolor='black')
    fig.patch.set_alpha(0.0)  # 背景透明
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_alpha(0.0)  # 坐标轴背景透明
    
    # 如果是init数据，需要特殊处理
    if is_init:
        # 创建透明度掩码
        # 1. 空白区域（值 < threshold）设为透明
        non_blank_mask = volume > threshold
        
        # 2. 如果提供了volume_size和sub_size，mask区域之外的部分也设为透明
        if volume_size is not None and sub_size is not None:
            mask_region = create_mask_region_mask(volume_size, sub_size, volume.shape)
            # 只显示mask区域内的非空白部分
            visible_mask = non_blank_mask & mask_region
        else:
            visible_mask = non_blank_mask
        
        # 创建RGBA颜色数组
        colors = np.zeros((D, H, W, 4), dtype=np.float32)
        
        # 将灰度值归一化到[0, 1]
        normalized = volume.astype(np.float32) / 255.0
        
        # 使用灰度colormap
        cmap = plt.cm.gray
        
        # 为可见区域分配颜色（不透明）
        visible_indices = np.where(visible_mask)
        if len(visible_indices[0]) > 0:
            z_indices, y_indices, x_indices = visible_indices
            rgb_values = cmap(normalized[visible_mask])[:, :3]
            colors[z_indices, y_indices, x_indices, :3] = rgb_values
            colors[z_indices, y_indices, x_indices, 3] = alpha  # 不透明
        
        # 不可见区域设为透明
        colors[~visible_mask, 3] = 0.0
        
        # 使用voxels绘制，只显示可见区域
        ax.voxels(visible_mask, facecolors=colors, edgecolors=None, linewidth=0)
    else:
        # 非init数据：使用简单的voxel可视化
        mask = volume > threshold
        
        # 创建颜色数组
        colors = np.zeros((D, H, W, 4), dtype=np.float32)
        normalized = volume.astype(np.float32) / 255.0
        cmap = plt.cm.gray
        
        # 为mask区域分配颜色（不透明）
        mask_indices = np.where(mask)
        if len(mask_indices[0]) > 0:
            z_indices, y_indices, x_indices = mask_indices
            rgb_values = cmap(normalized[mask])[:, :3]
            colors[z_indices, y_indices, x_indices, :3] = rgb_values
            colors[z_indices, y_indices, x_indices, 3] = alpha  # 不透明
        
        # 使用voxels绘制
        ax.voxels(mask, facecolors=colors, edgecolors=None, linewidth=0)
    
    # 设置坐标范围
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(0, D)
    
    # 隐藏所有坐标轴、网格线和标签
    ax.grid(False)
    ax.set_axis_off()  # 隐藏所有坐标轴和标签
    
    # 设置视角（不对称的角度）
    ax.view_init(elev=20, azim=50)
    
    plt.tight_layout()
    
    if output_path:
        # 保存时使用透明背景
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1, 
                   transparent=True)
        print(f"3D可视化已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="将2D截面图转化为3D立体图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 训练数据格式 - 可视化init数据:
   python scripts/visualize_2d_to_3d.py \\
     --input_folder "samples_outpaint_3d/size96_mask32_source1.576/init" \\
     --sample_name "z0000_y0000_x0000_c96_s32"

2. 训练数据格式 - 可视化target数据:
   python scripts/visualize_2d_to_3d.py \\
     --input_folder "samples_outpaint_3d/size96_mask32_source1.576/target" \\
     --sample_name "z0000_y0000_x0000_c96_s32" \\
     --is_init False

3. 生成样本格式 - 可视化生成的sample:
   python scripts/visualize_2d_to_3d.py \\
     --input_folder "samples_outpaint_3d_train/step_5000" \\
     --sample_index 0 \\
     --data_type "sample"

4. 生成样本格式 - 可视化target:
   python scripts/visualize_2d_to_3d.py \\
     --input_folder "samples_outpaint_3d_train/step_5000" \\
     --sample_index 0 \\
     --data_type "target" \\
     --is_init False

5. 生成样本格式 - 可视化init:
   python scripts/visualize_2d_to_3d.py \\
     --input_folder "samples_outpaint_3d_train/step_5000" \\
     --sample_index 0 \\
     --data_type "init"
        """
    )
    
    parser.add_argument(
        "--input_folder",
        type=str,
        default=DEFAULT_INPUT_FOLDER,
        help=f"输入文件夹路径（包含切片的文件夹，如 init/, target/，默认: {DEFAULT_INPUT_FOLDER}）"
    )
    
    parser.add_argument(
        "--sample_name",
        type=str,
        default=None,
        help=f"样本基础名称（训练数据格式，不含_z{{idx}}.png后缀，如 z0000_y0000_x0000_c96_s32）"
    )
    
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="样本索引（生成样本格式，如 0, 1, 2...，与--data_type一起使用）"
    )
    
    parser.add_argument(
        "--data_type",
        type=str,
        default="sample",
        choices=["sample", "target", "init", "mask"],
        help="数据类型（生成样本格式，默认: sample）"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=f"输出图片路径（默认: 自动生成在 {DEFAULT_OUTPUT_DIR}）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）"
    )
    
    parser.add_argument(
        "--is_init",
        type=lambda x: x.lower() == 'true',
        default=None,
        help="是否为init数据（默认: 自动检测，如果input_folder包含'init'则为True）"
    )
    
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="空白阈值（低于此值的像素视为空白，设为透明，默认: 5）"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="不透明度（0-1，默认: 1.0，不透明）"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 自动检测是否为init数据
    if args.is_init is None:
        args.is_init = 'init' in args.input_folder.lower() or args.data_type == 'init'
    
    # 加载3D体积
    print(f"正在加载数据: {args.input_folder}")
    if args.sample_index is not None:
        print(f"样本索引: {args.sample_index}, 数据类型: {args.data_type}")
    elif args.sample_name:
        print(f"样本名称: {args.sample_name}")
    else:
        print("自动检测文件格式...")
    
    volume = load_volume_from_slices(
        args.input_folder, 
        sample_name=args.sample_name,
        sample_index=args.sample_index,
        data_type=args.data_type
    )
    print(f"体积形状: {volume.shape}")
    
    # 解析尺寸（用于init数据的mask区域和连接线）
    volume_size = None
    sub_size = None
    if args.is_init and args.sample_name:
        try:
            volume_size, sub_size = parse_volume_size_from_name(args.sample_name)
            print(f"解析得到: volume_size={volume_size}, sub_size={sub_size}")
        except ValueError as e:
            print(f"警告: {e}，将不应用mask区域限制和连接线")
    elif args.is_init and args.sample_index is not None:
        # 对于生成的样本，可以从体积形状推断volume_size
        # 但sub_size无法从文件名推断，需要用户提供或跳过
        volume_size = volume.shape[0]  # 假设是立方体
        print(f"从体积形状推断: volume_size={volume_size}（无法推断sub_size，将不应用mask区域限制和连接线）")
        sub_size = None
    
    # 生成输出路径
    if args.output_path is None:
        if args.sample_index is not None:
            output_filename = f"{args.data_type}_{args.sample_index}_3d.png"
        elif args.sample_name:
            data_type = "init" if args.is_init else "target"
            output_filename = f"{args.sample_name}_{data_type}_3d.png"
        else:
            data_type = "init" if args.is_init else "target"
            output_filename = f"unknown_{data_type}_3d.png"
        args.output_path = os.path.join(args.output_dir, output_filename)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 可视化
    visualize_3d_volume(
        volume=volume,
        output_path=args.output_path,
        is_init=args.is_init,
        volume_size=volume_size,
        sub_size=sub_size,
        threshold=args.threshold,
        alpha=args.alpha,
    )
    
    print("完成！")


if __name__ == "__main__":
    main()
