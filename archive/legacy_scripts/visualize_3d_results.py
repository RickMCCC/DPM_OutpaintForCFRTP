"""
3D结果可视化脚本
用于可视化复合材料XCT扫描结果，区分纤维和树脂部分

使用方法:
  python scripts/visualize_3d_results.py \\
    --input_folder "samples_outpaint_3d_train/step_5000" \\
    --output_dir "visualizations" \\
    --fiber_threshold 128 \\
    --sample_index 0
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy import ndimage
from skimage.filters import threshold_otsu, threshold_local
from matplotlib.colors import Normalize, LinearSegmentedColormap

current_dir = os.path.dirname(os.path.abspath(__file__))

def create_yellow_green_cyan_colormap():
    """
    创建自定义的黄色-绿色-青色光谱colormap
    从黄色(1,1,0) -> 绿色(0,1,0) -> 青色(0,1,1)
    """
    colors = [
        (1.0, 1.0, 0.0),  # 黄色
        (0.5, 1.0, 0.25), # 黄绿色
        (0.0, 1.0, 0.5),  # 绿色
        (0.0, 0.75, 0.75), # 青绿色
        (0.0, 0.5, 1.0),  # 青色
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('yellow_green_cyan', colors, N=n_bins)
    return cmap


def get_low_saturation_colors():
    """
    生成5种低饱和度颜色，用于z轴分段
    返回5个RGB颜色元组的列表，饱和度较低
    """
    # 使用低饱和度的颜色：通过混合白色来降低饱和度
    base_colors = [
        (0.8, 0.3, 0.3),  # 低饱和度红色
        (0.3, 0.6, 0.3),  # 低饱和度绿色
        (0.3, 0.3, 0.8),  # 低饱和度蓝色
        (0.7, 0.5, 0.2),  # 低饱和度橙色
        (0.5, 0.3, 0.7),  # 低饱和度紫色
    ]
    
    # 进一步降低饱和度：与白色混合
    saturation_factor = 0.4  # 饱和度因子，越小越接近白色
    low_sat_colors = []
    for r, g, b in base_colors:
        # 与白色混合：new_color = old_color * factor + white * (1 - factor)
        new_r = r * saturation_factor + (1 - saturation_factor)
        new_g = g * saturation_factor + (1 - saturation_factor)
        new_b = b * saturation_factor + (1 - saturation_factor)
        low_sat_colors.append((new_r, new_g, new_b))
    
    return low_sat_colors

DEFAULT_INPUT_FOLDER = r"C:\Terong\improved-diffusion\scripts\outpaint_3d_checkpoint\mask32_size96_source1.576\step_25000"
DEFAULT_OUTPUT_DIR = r"C:\Terong\improved-diffusion\scripts\visualizations"

DEFAULT_FIBER_THRESHOLD = 128  # 灰度阈值（仅在threshold_method='absolute'时使用）
DEFAULT_SAMPLE_INDEX = 0  # 要可视化的样本索引
DEFAULT_DOWNSAMPLE = 1  # 降采样因子，用于减少计算量（1=不降采样，2=每2个像素取1个）
DEFAULT_THRESHOLD_METHOD = 'adaptive_percentile'  # 阈值方法：'absolute', 'otsu', 'adaptive_percentile', 'adaptive_local'
DEFAULT_PERCENTILE = 75  # 百分位数阈值（仅在adaptive_percentile时使用）


def load_volume_from_sample_folder(sample_folder, sample_index=0):
    """
    从训练时保存的样本文件夹加载体积
    :param sample_folder: 样本文件夹路径（如 samples_outpaint_3d_train/step_5000）
    :param sample_index: 样本索引（0, 1, ...）
    :return: 3D numpy数组 (D, H, W)，值范围[0, 255]
    """
    volumes = []
    
    # 查找所有sample_*.png文件
    all_files = sorted([f for f in os.listdir(sample_folder) if f.endswith('.png')])
    
    # 按样本ID分组
    sample_dict = {}
    for f in all_files:
        if f.startswith('sample_'):
            # 例如: sample_0_z000.png -> sample_0
            parts = f.rsplit('_z', 1)
            if len(parts) == 2:
                sample_id = parts[0]  # sample_0
                if sample_id not in sample_dict:
                    sample_dict[sample_id] = []
                sample_dict[sample_id].append(f)
    
    # 加载指定样本的所有Z切片
    sample_id = f"sample_{sample_index}"
    if sample_id not in sample_dict:
        raise ValueError(f"样本 {sample_index} 不存在！可用样本: {sorted([int(k.split('_')[1]) for k in sample_dict.keys()])}")
    
    slices = []
    for slice_file in sorted(sample_dict[sample_id]):
        filepath = os.path.join(sample_folder, slice_file)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            slices.append(img)
    
    if len(slices) == 0:
        raise ValueError(f"未找到样本 {sample_index} 的切片文件！")
    
    volume = np.stack(slices, axis=0)  # (D, H, W)
    return volume


def threshold_segmentation(volume, fiber_threshold=128, method='adaptive_percentile', percentile=75):
    """
    阈值分割：区分纤维和树脂（支持多种方法）
    :param volume: 3D体积数组，值范围[0, 255]
    :param fiber_threshold: 纤维阈值（仅在method='absolute'时使用）
    :param method: 阈值方法
        - 'absolute': 绝对阈值
        - 'otsu': Otsu自适应阈值（基于直方图）
        - 'adaptive_percentile': 基于百分位数的自适应阈值（考虑相对灰度）
        - 'adaptive_local': 局部自适应阈值（每个切片独立计算）
    :param percentile: 百分位数（仅在adaptive_percentile时使用，默认75表示75%分位数）
    :return: 二值化数组，1=纤维，0=树脂
    """
    if method == 'absolute':
        fiber_mask = (volume > fiber_threshold).astype(np.uint8)
    
    elif method == 'otsu':
        # 使用Otsu方法计算全局阈值
        threshold = threshold_otsu(volume)
        fiber_mask = (volume > threshold).astype(np.uint8)
        print(f"  Otsu阈值: {threshold:.2f}")
    
    elif method == 'adaptive_percentile':
        # 基于百分位数的自适应阈值（考虑整张图的相对灰度分布）
        threshold = np.percentile(volume, percentile)
        fiber_mask = (volume > threshold).astype(np.uint8)
        print(f"  百分位数阈值 ({percentile}%): {threshold:.2f}")
    
    elif method == 'adaptive_local':
        # 局部自适应阈值（每个切片独立计算Otsu阈值）
        fiber_mask = np.zeros_like(volume, dtype=np.uint8)
        D = volume.shape[0]
        thresholds = []
        for d in range(D):
            slice_2d = volume[d]
            if slice_2d.max() > slice_2d.min():  # 避免全相同值
                threshold = threshold_otsu(slice_2d)
                fiber_mask[d] = (slice_2d > threshold).astype(np.uint8)
                thresholds.append(threshold)
            else:
                fiber_mask[d] = np.zeros_like(slice_2d, dtype=np.uint8)
        if thresholds:
            print(f"  局部自适应阈值范围: [{min(thresholds):.2f}, {max(thresholds):.2f}], 平均: {np.mean(thresholds):.2f}")
    
    else:
        raise ValueError(f"未知的阈值方法: {method}")
    
    return fiber_mask


def downsample_volume(volume, factor=2):
    """
    降采样体积以减少计算量
    :param volume: 3D体积数组
    :param factor: 降采样因子
    :return: 降采样后的体积
    """
    if factor <= 1:
        return volume
    
    # 使用平均池化进行降采样
    D, H, W = volume.shape
    new_D = D // factor
    new_H = H // factor
    new_W = W // factor
    
    downsampled = np.zeros((new_D, new_H, new_W), dtype=volume.dtype)
    
    for d in range(new_D):
        for h in range(new_H):
            for w in range(new_W):
                d_start = d * factor
                d_end = min((d + 1) * factor, D)
                h_start = h * factor
                h_end = min((h + 1) * factor, H)
                w_start = w * factor
                w_end = min((w + 1) * factor, W)
                
                downsampled[d, h, w] = np.mean(
                    volume[d_start:d_end, h_start:h_end, w_start:w_end]
                )
    
    return downsampled


def visualize_3d_voxels(fiber_mask, volume=None, output_path=None, title="3D Fiber Visualization", use_grayscale_colormap=True):
    """
    使用voxels方法可视化3D体积
    :param fiber_mask: 二值化掩码，1=纤维，0=树脂
    :param volume: 原始体积（用于灰度颜色映射），如果为None则使用简单颜色
    :param output_path: 输出图片路径，如果为None则显示
    :param title: 图表标题
    :param use_grayscale_colormap: 是否根据灰度值使用颜色映射
    """
    # 不进行降采样，使用原始尺寸
    D, H, W = fiber_mask.shape
    
    fig = plt.figure(figsize=(14, 10))  # 增加宽度以显示z轴标签
    ax = fig.add_subplot(111, projection='3d')
    
    if use_grayscale_colormap and volume is not None:
        # 根据灰度值使用颜色映射（只对纤维部分）
        # 使用自定义的黄-绿-青光谱colormap，将灰度值映射到颜色
        fiber_volume = volume * fiber_mask.astype(float)  # 只保留纤维部分的灰度值
        
        # 归一化灰度值到[0, 1]，扩展范围以增加颜色变化范围
        if fiber_volume.max() > fiber_volume.min():
            # 扩展范围：使用更宽的归一化范围，让颜色变化更明显
            range_expansion = 1.0  # 扩展100%的范围，大幅增加颜色变化力度
            v_min = fiber_volume.min()
            v_max = fiber_volume.max()
            v_range = v_max - v_min
            normalized = (fiber_volume - (v_min - v_range * range_expansion)) / (v_range * (1 + 2 * range_expansion))
            normalized = np.clip(normalized, 0, 1)  # 确保在[0, 1]范围内
        else:
            normalized = fiber_volume / 255.0
        
        # 使用自定义的黄-绿-青光谱colormap
        colormap = create_yellow_green_cyan_colormap()
        colors_rgb = colormap(normalized)[..., :3]  # 只取RGB，不要alpha
        
        # 创建RGBA颜色数组
        colors = np.zeros(fiber_mask.shape + (4,))
        D, H, W = fiber_mask.shape
        
        # 按z轴分成5个水平部分，每部分使用不同的低饱和度颜色
        z_coords = np.arange(D).reshape(D, 1, 1)  # (D, 1, 1)
        z_step = D / 5.0  # 每部分的高度
        
        # 获取5种低饱和度颜色
        low_sat_colors = get_low_saturation_colors()
        
        # 为每个z层分配颜色
        for i in range(5):
            z_start = i * z_step
            z_end = (i + 1) * z_step
            z_mask_layer = (z_coords >= z_start) & (z_coords < z_end)  # (D, 1, 1)
            z_mask_3d = np.broadcast_to(z_mask_layer, (D, H, W))  # (D, H, W)
            
            # 设置该层的颜色（低饱和度）
            layer_color = np.array(low_sat_colors[i])
            colors[z_mask_3d, :3] = layer_color
        
        # 设置透明度：所有部分都使用较低的不透明度
        low_alpha = 0.08  # 非常低的不透明度
        colors[..., 3] = fiber_mask.astype(float) * low_alpha
        
        # 树脂部分设为透明
        colors[fiber_mask == 0, 3] = 0.0
    else:
        # 简单颜色映射：按z轴分成5个水平部分
        colors = np.empty(fiber_mask.shape + (4,))
        D, H, W = fiber_mask.shape
        
        # 按z轴分成5个水平部分，每部分使用不同的低饱和度颜色
        z_coords = np.arange(D).reshape(D, 1, 1)  # (D, 1, 1)
        z_step = D / 5.0  # 每部分的高度
        
        # 获取5种低饱和度颜色
        low_sat_colors = get_low_saturation_colors()
        
        # 为每个z层分配颜色
        for i in range(5):
            z_start = i * z_step
            z_end = (i + 1) * z_step
            z_mask_layer = (z_coords >= z_start) & (z_coords < z_end)  # (D, 1, 1)
            z_mask_3d = np.broadcast_to(z_mask_layer, (D, H, W))  # (D, H, W)
            
            # 设置该层的颜色（低饱和度）
            layer_color = np.array(low_sat_colors[i])
            colors[z_mask_3d, :3] = layer_color
        
        # 设置透明度：所有部分都使用较低的不透明度
        low_alpha = 0.08  # 非常低的不透明度
        colors[..., 3] = fiber_mask.astype(float) * low_alpha
    
    # 绘制voxels（无格线）
    ax.voxels(fiber_mask, facecolors=colors, edgecolors=None, linewidth=0)
    
    ax.set_xlabel('X (Width)', fontsize=12)
    ax.set_ylabel('Y (Height)', fontsize=12)
    ax.set_zlabel('Z (Depth)', fontsize=12, labelpad=30)  # 大幅增加labelpad以显示z轴标签
    ax.set_title(title, fontsize=14, pad=20)
    
    # 调整布局，为z轴标签留出更多空间
    plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.1)
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        print(f"3D可视化已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def visualize_3d_marching_cubes(fiber_mask, volume=None, output_path=None, title="3D Fiber Surface", use_grayscale_colormap=True):
    """
    使用marching cubes算法提取等值面并可视化
    :param fiber_mask: 二值化掩码，1=纤维，0=树脂
    :param volume: 原始体积（用于灰度颜色映射），如果为None则使用简单颜色
    :param output_path: 输出图片路径，如果为None则显示
    :param title: 图表标题
    :param use_grayscale_colormap: 是否根据灰度值使用颜色映射
    """
    # 使用marching cubes提取等值面
    try:
        verts, faces, normals, values = measure.marching_cubes(
            fiber_mask.astype(float), level=0.5, spacing=(1, 1, 1)
        )
    except Exception as e:
        print(f"Marching cubes失败: {e}")
        print("回退到voxels方法...")
        visualize_3d_voxels(fiber_mask, volume, output_path, title, use_grayscale_colormap)
        return
    
    fig = plt.figure(figsize=(14, 10))  # 增加宽度以显示z轴标签
    ax = fig.add_subplot(111, projection='3d')
    
    if use_grayscale_colormap and volume is not None:
        # 根据顶点位置的灰度值设置颜色
        # 获取顶点对应的灰度值
        verts_int = verts.astype(int)
        # 确保索引在范围内
        verts_int[:, 0] = np.clip(verts_int[:, 0], 0, fiber_mask.shape[0] - 1)
        verts_int[:, 1] = np.clip(verts_int[:, 1], 0, fiber_mask.shape[1] - 1)
        verts_int[:, 2] = np.clip(verts_int[:, 2], 0, fiber_mask.shape[2] - 1)
        
        # 获取每个顶点的灰度值
        gray_values = volume[verts_int[:, 0], verts_int[:, 1], verts_int[:, 2]]
        
        # 归一化到[0, 1]，扩展范围以增加颜色变化范围
        if gray_values.max() > gray_values.min():
            range_expansion = 1.0  # 扩展100%的范围，大幅增加颜色变化力度
            v_min = gray_values.min()
            v_max = gray_values.max()
            v_range = v_max - v_min
            normalized = (gray_values - (v_min - v_range * range_expansion)) / (v_range * (1 + 2 * range_expansion))
            normalized = np.clip(normalized, 0, 1)  # 确保在[0, 1]范围内
        else:
            normalized = gray_values / 255.0
        
        # 按z轴分成5个水平部分，每部分使用不同的低饱和度颜色
        # 获取面的z坐标（使用面的第一个顶点）
        face_z_coords = verts[faces[:, 0], 0]  # z坐标在verts的第0列
        D = fiber_mask.shape[0]
        z_step = D / 5.0
        
        # 获取5种低饱和度颜色
        low_sat_colors = get_low_saturation_colors()
        
        # 为每个z层创建mesh集合（无格线）
        for i in range(5):
            z_start = i * z_step
            z_end = (i + 1) * z_step
            z_mask = (face_z_coords >= z_start) & (face_z_coords < z_end)
            
            if np.any(z_mask):
                mesh_layer = Poly3DCollection(verts[faces[z_mask]], edgecolor='none', alpha=0.08)
                mesh_layer.set_facecolor(low_sat_colors[i])
                ax.add_collection3d(mesh_layer)
    else:
        # 简单颜色映射：按z轴分成5个水平部分
        # 获取面的z坐标（使用面的第一个顶点）
        face_z_coords = verts[faces[:, 0], 0]  # z坐标在verts的第0列
        D = fiber_mask.shape[0]
        z_step = D / 5.0
        
        # 获取5种低饱和度颜色
        low_sat_colors = get_low_saturation_colors()
        
        # 为每个z层创建mesh集合（无格线）
        for i in range(5):
            z_start = i * z_step
            z_end = (i + 1) * z_step
            z_mask = (face_z_coords >= z_start) & (face_z_coords < z_end)
            
            if np.any(z_mask):
                mesh_layer = Poly3DCollection(verts[faces[z_mask]], edgecolor='none', alpha=0.08)
                mesh_layer.set_facecolor(low_sat_colors[i])
                ax.add_collection3d(mesh_layer)
    
    # 设置坐标范围
    ax.set_xlim(0, fiber_mask.shape[2])
    ax.set_ylim(0, fiber_mask.shape[1])
    ax.set_zlim(0, fiber_mask.shape[0])
    
    ax.set_xlabel('X (Width)', fontsize=12)
    ax.set_ylabel('Y (Height)', fontsize=12)
    ax.set_zlabel('Z (Depth)', fontsize=12, labelpad=30)  # 大幅增加labelpad以显示z轴标签
    ax.set_title(title, fontsize=14, pad=20)
    
    # 调整布局，为z轴标签留出更多空间
    plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.1)
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        print(f"3D表面可视化已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def visualize_2d_slices(volume, fiber_mask, output_path=None, num_slices=5, use_grayscale_colormap=True):
    """
    可视化2D切片（原始图像和分割结果）
    :param volume: 原始3D体积
    :param fiber_mask: 二值化掩码
    :param output_path: 输出图片路径
    :param num_slices: 要显示的切片数量
    :param use_grayscale_colormap: 是否根据灰度值使用颜色映射
    """
    D, H, W = volume.shape
    
    # 选择均匀分布的切片
    slice_indices = np.linspace(0, D - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(2, num_slices, figsize=(4 * num_slices, 8))
    
    for i, z_idx in enumerate(slice_indices):
        # 原始图像
        axes[0, i].imshow(volume[z_idx], cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f'Original Slice Z={z_idx}', fontsize=10)
        axes[0, i].axis('off')
        
        # 分割结果（中间版本：原始图像+纤维部分用光谱颜色标记）
        if use_grayscale_colormap:
            # 创建RGB图像：原始图像为灰度，纤维部分用光谱颜色
            slice_volume = volume[z_idx].copy()
            slice_mask = fiber_mask[z_idx].astype(bool)
            
            # 创建RGB图像
            rgb_slice = np.zeros((H, W, 3))
            
            # 树脂部分：保持灰度
            resin_mask = ~slice_mask
            gray_normalized = slice_volume[resin_mask] / 255.0
            rgb_slice[resin_mask] = np.stack([gray_normalized, gray_normalized, gray_normalized], axis=-1)
            
            # 纤维部分：根据灰度值映射到黄-绿-青光谱colormap
            if np.any(slice_mask):
                fiber_gray = slice_volume[slice_mask]
                if fiber_gray.max() > fiber_gray.min():
                    # 扩展范围以增加颜色变化范围
                    range_expansion = 0.6  # 扩展60%的范围
                    v_min = fiber_gray.min()
                    v_max = fiber_gray.max()
                    v_range = v_max - v_min
                    normalized = (fiber_gray - (v_min - v_range * range_expansion)) / (v_range * (1 + 2 * range_expansion))
                    normalized = np.clip(normalized, 0, 1)  # 确保在[0, 1]范围内
                else:
                    normalized = fiber_gray / 255.0
                
                colormap = create_yellow_green_cyan_colormap()
                rgb_slice[slice_mask] = colormap(normalized)[:, :3]  # RGB颜色
            
            axes[1, i].imshow(rgb_slice)
        else:
            # 简单叠加显示
            axes[1, i].imshow(volume[z_idx], cmap='gray', vmin=0, vmax=255)
            # 叠加纤维区域（红色半透明）
            fiber_overlay = fiber_mask[z_idx].astype(float)
            axes[1, i].imshow(fiber_overlay, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        
        axes[1, i].set_title(f'Segmented Slice Z={z_idx}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"2D切片可视化已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def calculate_statistics(volume, fiber_mask):
    """
    计算统计信息
    :param volume: 原始3D体积
    :param fiber_mask: 二值化掩码
    :return: 统计信息字典
    """
    total_voxels = fiber_mask.size
    fiber_voxels = np.sum(fiber_mask)
    resin_voxels = total_voxels - fiber_voxels
    
    fiber_volume_fraction = fiber_voxels / total_voxels * 100
    
    # 计算连通区域
    from scipy.ndimage import label
    labeled, num_features = label(fiber_mask)
    
    stats = {
        'total_voxels': total_voxels,
        'fiber_voxels': fiber_voxels,
        'resin_voxels': resin_voxels,
        'fiber_volume_fraction': fiber_volume_fraction,
        'num_fiber_regions': num_features,
        'volume_shape': volume.shape,
    }
    
    return stats


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="3D结果可视化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 可视化默认样本:
   python scripts/visualize_3d_results.py

2. 指定输入文件夹和阈值:
   python scripts/visualize_3d_results.py \\
     --input_folder "samples_outpaint_3d_train/step_5000" \\
     --fiber_threshold 150 \\
     --sample_index 0

3. 生成所有可视化:
   python scripts/visualize_3d_results.py \\
     --input_folder "samples_outpaint_3d_train/step_5000" \\
     --output_dir "visualizations" \\
     --fiber_threshold 128 \\
     --visualize_3d_voxels \\
     --visualize_3d_surface \\
     --visualize_2d_slices
        """
    )
    
    parser.add_argument(
        "--input_folder",
        type=str,
        default=DEFAULT_INPUT_FOLDER,
        help=f"输入文件夹路径（默认: {DEFAULT_INPUT_FOLDER}）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）"
    )
    
    parser.add_argument(
        "--fiber_threshold",
        type=int,
        default=DEFAULT_FIBER_THRESHOLD,
        help=f"纤维阈值（0-255，仅在threshold_method='absolute'时使用，默认: {DEFAULT_FIBER_THRESHOLD}）"
    )
    
    parser.add_argument(
        "--threshold_method",
        type=str,
        default=DEFAULT_THRESHOLD_METHOD,
        choices=['absolute', 'otsu', 'adaptive_percentile', 'adaptive_local'],
        help=f"阈值方法（默认: {DEFAULT_THRESHOLD_METHOD}）"
    )
    
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help=f"百分位数阈值（仅在adaptive_percentile时使用，默认: {DEFAULT_PERCENTILE}）"
    )
    
    parser.add_argument(
        "--use_grayscale_colormap",
        action="store_true",
        help="根据灰度值使用颜色映射（默认启用）"
    )
    
    parser.add_argument(
        "--no_grayscale_colormap",
        action="store_false",
        dest="use_grayscale_colormap",
        help="禁用灰度颜色映射，使用简单颜色"
    )
    
    # 设置默认值：默认启用灰度颜色映射
    parser.set_defaults(use_grayscale_colormap=True)
    
    parser.add_argument(
        "--sample_index",
        type=int,
        default=DEFAULT_SAMPLE_INDEX,
        help=f"要可视化的样本索引（默认: {DEFAULT_SAMPLE_INDEX}）"
    )
    
    parser.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_DOWNSAMPLE,
        help=f"降采样因子（用于减少计算量，默认: {DEFAULT_DOWNSAMPLE}）"
    )
    
    parser.add_argument(
        "--visualize_3d_voxels",
        action="store_true",
        help="生成3D voxels可视化"
    )
    
    parser.add_argument(
        "--visualize_3d_surface",
        action="store_true",
        help="生成3D表面可视化（marching cubes）"
    )
    
    parser.add_argument(
        "--visualize_2d_slices",
        action="store_true",
        help="生成2D切片可视化"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="生成所有可视化"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("3D结果可视化")
    print("=" * 60)
    
    # 加载体积
    print(f"\n从 {args.input_folder} 加载样本 {args.sample_index}...")
    try:
        volume = load_volume_from_sample_folder(args.input_folder, args.sample_index)
        print(f"体积尺寸: {volume.shape} (D, H, W)")
        print(f"灰度值范围: [{volume.min()}, {volume.max()}]")
    except Exception as e:
        print(f"错误: {e}")
        return
    
    # 阈值分割
    print(f"\n进行阈值分割（方法={args.threshold_method}）...")
    fiber_mask = threshold_segmentation(
        volume, 
        fiber_threshold=args.fiber_threshold,
        method=args.threshold_method,
        percentile=args.percentile
    )
    
    # 不进行降采样，使用原始尺寸
    
    # 计算统计信息
    stats = calculate_statistics(volume, fiber_mask)
    print(f"\n统计信息:")
    print(f"  总体积: {stats['volume_shape']}")
    print(f"  纤维体积分数: {stats['fiber_volume_fraction']:.2f}%")
    print(f"  纤维体素数: {stats['fiber_voxels']}")
    print(f"  树脂体素数: {stats['resin_voxels']}")
    print(f"  纤维连通区域数: {stats['num_fiber_regions']}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成可视化
    method_suffix = args.threshold_method
    if args.threshold_method == 'adaptive_percentile':
        method_suffix = f"{method_suffix}_{args.percentile}"
    base_name = f"sample_{args.sample_index}_{method_suffix}"
    
    if args.all or args.visualize_3d_voxels:
        print("\n生成3D voxels可视化...")
        output_path = os.path.join(args.output_dir, f"{base_name}_3d_voxels.png")
        visualize_3d_voxels(fiber_mask, volume, output_path, 
                           title=f"3D Fiber Visualization ({args.threshold_method})",
                           use_grayscale_colormap=args.use_grayscale_colormap)
    
    if args.all or args.visualize_3d_surface:
        print("\n生成3D表面可视化...")
        output_path = os.path.join(args.output_dir, f"{base_name}_3d_surface.png")
        visualize_3d_marching_cubes(fiber_mask, volume, output_path,
                                  title=f"3D Fiber Surface ({args.threshold_method})",
                                  use_grayscale_colormap=args.use_grayscale_colormap)
    
    if args.all or args.visualize_2d_slices:
        print("\n生成2D切片可视化...")
        output_path = os.path.join(args.output_dir, f"{base_name}_2d_slices.png")
        visualize_2d_slices(volume, fiber_mask, output_path, use_grayscale_colormap=args.use_grayscale_colormap)
    
    # 如果没有指定任何可视化选项，默认生成所有
    if not (args.all or args.visualize_3d_voxels or args.visualize_3d_surface or args.visualize_2d_slices):
        print("\n未指定可视化选项，生成所有可视化...")
        output_path = os.path.join(args.output_dir, f"{base_name}_3d_voxels.png")
        visualize_3d_voxels(fiber_mask, volume, output_path,
                           title=f"3D Fiber Visualization ({args.threshold_method})",
                           use_grayscale_colormap=args.use_grayscale_colormap)
        
        output_path = os.path.join(args.output_dir, f"{base_name}_2d_slices.png")
        visualize_2d_slices(volume, fiber_mask, output_path, use_grayscale_colormap=args.use_grayscale_colormap)
    
    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
