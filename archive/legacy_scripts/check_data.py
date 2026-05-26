import sys
import os
import torch
import numpy as np

# ==================== 修改点 1 ====================
# 在导入pyplot之前，强制matplotlib使用Agg后端（用于写入文件）
import matplotlib

matplotlib.use('Agg')
# ===============================================

import matplotlib.pyplot as plt

# 这是一个小技巧，确保可以从scripts目录导入上层的improved_diffusion库
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from improved_diffusion.image_datasets import load_data
from scripts.image_train import model_and_diffusion_defaults, create_argparser


def main():
    """
    加载一个批次的数据并进行可视化检查。
    """
    args = create_argparser().parse_args([])

    args.data_dir = "E:/MyPython/improved-diffusion/improved_diffusion/training_data"
    args.batch_size = 4
    args.image_size = 64
    args.class_cond = False

    print("--- 开始数据加载检查 ---")
    print(f"数据目录: {args.data_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"图像尺寸: {args.image_size}")

    data_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    batch, cond = next(iter(data_loader))

    print("\n--- 检查点 ---")
    print(f"批次张量的形状 (Batch Shape): {batch.shape}")
    print(f"批次张量的数据类型 (dtype): {batch.dtype}")
    print(f"批次张量的数值范围 (Min): {batch.min()}")
    print(f"批次张量的数值范围 (Max): {batch.max()}")

    batch = ((batch + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    batch = batch.permute(0, 2, 3, 1)

    if batch.shape[-1] == 1:
        batch = batch.squeeze(-1)

    images_to_show = batch.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle("Data Batch Visualization Check")

    for i, ax in enumerate(axes.flat):
        if i < len(images_to_show):
            ax.imshow(images_to_show[i], cmap='gray')
            ax.set_title(f"Sample {i + 1}")
        ax.axis("off")

    plt.tight_layout()

    # ==================== 修改点 2 ====================
    # 将 plt.show() 替换为 plt.savefig()
    output_filename = "data_check_output.png"
    plt.savefig(output_filename)
    print(f"\n检查图像已保存为: {output_filename}")
    # ===============================================


if __name__ == "__main__":
    main()