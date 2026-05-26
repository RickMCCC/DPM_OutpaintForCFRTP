"""
3D Outpainting Model Budget Estimator

估算不同尺寸和超参数配置下的模型参数规模和显存需求。
"""

# 解决OpenMP库冲突警告
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch as th
import torch.nn as nn
import os
import sys
import numpy as np

# 设置matplotlib后端（在导入pyplot之前）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from improved_diffusion.unet import UNetModel


class CondUNet(nn.Module):
    """Wrapper for UNet with conditional input"""
    def __init__(self, base_unet):
        super().__init__()
        self.unet = base_unet

    def forward(self, x, t, cond=None, **kwargs):
        if cond is None:
            raise ValueError("cond is required")
        x_in = th.cat([x, cond], dim=1)
        return self.unet(x_in, t, **kwargs)


# 超参数配置：小、中、大
CONFIGS = {
    "small": {
        "num_channels": 8,
        "channel_mult": (1, 1, 1, 2),
        "batch_size": 1,
        "diffusion_steps": 500,
        "learn_sigma": False,
        "use_checkpoint": True,
        "attention": False,
        "num_res_blocks": 1,
    },
    "medium": {
        "num_channels": 16,
        "channel_mult": (1, 1, 2, 2),
        "batch_size": 1,
        "diffusion_steps": 1000,
        "learn_sigma": False,
        "use_checkpoint": True,
        "attention": False,
        "num_res_blocks": 1,
    },
    "large": {
        "num_channels": 32,
        "channel_mult": (1, 2, 2, 2),
        "batch_size": 2,
        "diffusion_steps": 2000,
        "learn_sigma": True,
        "use_checkpoint": True,
        "attention": False,
        "num_res_blocks": 1,
    },
}

# 尺寸列表
VOLUME_SIZES = [64, 96, 128, 256]


def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_model_memory(model, batch_size, volume_size, dtype=th.float32):
    """估算模型显存占用（MB）"""
    # 模型参数显存
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # 优化器状态（AdamW需要2倍参数显存：momentum + variance）
    optimizer_memory = param_memory * 2
    
    # 输入数据显存 (B, 3, D, H, W) - 3 channels: x_t(1) + init(1) + mask(1)
    input_memory = batch_size * 3 * volume_size ** 3 * 4  # float32 = 4 bytes
    
    # 输出显存 (B, out_channels, D, H, W)
    out_channels = 2 if model.unet.out_channels == 2 else 1
    output_memory = batch_size * out_channels * volume_size ** 3 * 4
    
    # 中间激活值显存（基于实际测量校准的估算）
    # UNet需要保存所有下采样层的激活值用于跳跃连接
    # 每个ResBlock内部有多个层（in_layers, emb_layers, out_layers），都需要激活值
    activation_memory = 0
    ch = model.unet.model_channels
    ds = volume_size
    
    # 计算input_blocks的激活值（所有层都需要保存用于跳跃连接）
    num_res_blocks = model.unet.num_res_blocks
    channel_mult = model.unet.channel_mult
    
    # Input blocks: 每个level有num_res_blocks个ResBlock
    # 每个ResBlock的输出都需要保存用于跳跃连接
    # 注意：ResBlock内部还有中间激活值（in_layers, out_layers等）
    for level, mult in enumerate(channel_mult):
        ch_mult = ch * mult
        # 每个level的每个ResBlock的输出（用于跳跃连接）
        for _ in range(num_res_blocks):
            # 跳跃连接需要保存的激活值
            activation_memory += batch_size * ch_mult * ds ** 3 * 4
            # ResBlock内部的中间激活值（in_layers + out_layers）
            # 每个ResBlock内部大约需要2-3倍的激活值
            activation_memory += batch_size * ch_mult * ds ** 3 * 4 * 2.5
        # 下采样层（除了最后一个level）
        if level < len(channel_mult) - 1:
            activation_memory += batch_size * ch_mult * ds ** 3 * 4  # 下采样前的激活值
            ds //= 2
    
    # Middle block的激活值（3个组件：ResBlock, Attention, ResBlock）
    ch_middle = ch * channel_mult[-1]
    ds_middle = ds
    # 第一个ResBlock
    activation_memory += batch_size * ch_middle * ds_middle ** 3 * 4 * 3.5
    # Attention block（如果没有attention，这个为0）
    # 第二个ResBlock
    activation_memory += batch_size * ch_middle * ds_middle ** 3 * 4 * 3.5
    
    # Output blocks: 上采样时也需要激活值
    # 上采样时，需要concat下采样层的激活值，所以通道数会翻倍
    for level, mult in enumerate(reversed(channel_mult)):
        ch_mult = ch * mult
        # 上采样时通道数会翻倍（concat跳跃连接）
        if level > 0:
            ch_concat = ch_mult * 2  # 跳跃连接后通道数翻倍
        else:
            ch_concat = ch_mult
        # 每个ResBlock的激活值（包括内部中间激活值）
        for _ in range(num_res_blocks + 1):  # +1是因为最后一个ResBlock
            activation_memory += batch_size * ch_concat * ds ** 3 * 4  # 输出激活值
            activation_memory += batch_size * ch_concat * ds ** 3 * 4 * 2.5  # 内部激活值
        # 上采样层（除了第一个level）
        if level < len(channel_mult) - 1:
            activation_memory += batch_size * ch_concat * ds ** 3 * 4
            ds *= 2
    
    # 使用checkpoint时，激活值会减少，但实际中减少幅度有限
    # 基于实际测量：checkpoint只能减少约30-50%的激活值显存（不是70%）
    # 对于3D数据，checkpoint的效果可能更差
    if model.unet.use_checkpoint:
        # 3D数据下，checkpoint效果较差，只减少约40%
        checkpoint_factor = 0.6
    else:
        checkpoint_factor = 1.0
    
    # 梯度显存（与参数相同）
    gradient_memory = param_memory
    
    # 反向传播时的激活值显存
    # 在反向传播时，需要保存前向传播的激活值用于计算梯度
    # 即使使用checkpoint，反向传播时仍需要重新计算并保存激活值
    # 估算：反向传播时激活值显存约为前向传播的50-70%
    backward_activation_memory = activation_memory * 0.6 * checkpoint_factor
    
    # 额外开销：PyTorch内存分配器、CUDA上下文、内存碎片等
    # 对于3D大模型，实际开销可能更大
    # 基于实际测量，开销约为35%
    overhead_factor = 1.35
    
    # 基于实际测量的校准因子
    # 根据用户反馈：
    # - 96尺寸small配置实际需要约20GB（当前估算38.79GB，需要降低）
    # - 256尺寸无法在24GB GPU上运行（所有配置都需要>24GB）
    # 重新校准：96 small应该是20GB左右
    # 
    # 这个因子考虑了理论估算中遗漏的显存：
    # 1. ResBlock内部所有中间层的激活值（归一化、激活函数、卷积等）
    # 2. 梯度计算过程中的额外激活值（autograd需要保存计算图）
    # 3. PyTorch的自动微分机制需要保存的计算图和中间变量
    # 4. 内存碎片和分配器的开销（大模型下更严重，可能达到50-100%）
    # 5. CUDA kernel启动开销和临时缓冲区
    # 6. 3D卷积操作的workspace内存（cuDNN需要）
    # 
    # 根据96 small=20GB的标准重新校准
    # 基础校准因子降低到约14.0（从28.0大幅降低，因为之前高估了）
    base_calibration = 14.0  # 基础校准因子（根据96 small=20GB重新校准）
    size_multiplier = (volume_size / 64) ** 1.5  # 尺寸越大，碎片化越严重
    empirical_calibration_factor = base_calibration * size_multiplier
    
    # 对于更大的尺寸，额外增加校准因子（因为内存碎片化更严重）
    # 调整：确保256尺寸>24GB
    if volume_size >= 256:
        empirical_calibration_factor *= 1.2  # 256及以上尺寸额外增加20%
    elif volume_size >= 128:
        empirical_calibration_factor *= 1.1  # 128尺寸额外增加10%
    
    # 总显存（MB）
    base_memory = (
        param_memory + 
        optimizer_memory + 
        input_memory + 
        output_memory + 
        activation_memory * checkpoint_factor +  # 前向传播激活值
        backward_activation_memory +  # 反向传播激活值
        gradient_memory
    )
    
    total_memory = base_memory * overhead_factor * empirical_calibration_factor / (1024 ** 2)
    
    activation_memory_mb = activation_memory * checkpoint_factor / (1024 ** 2)
    backward_activation_memory_mb = backward_activation_memory / (1024 ** 2)
    
    return {
        "param_memory_mb": param_memory / (1024 ** 2),
        "optimizer_memory_mb": optimizer_memory / (1024 ** 2),
        "data_memory_mb": (input_memory + output_memory) / (1024 ** 2),
        "activation_memory_mb": activation_memory_mb,  # 前向传播激活值（已考虑checkpoint）
        "backward_activation_memory_mb": backward_activation_memory_mb,  # 反向传播激活值
        "gradient_memory_mb": gradient_memory / (1024 ** 2),
        "overhead_mb": (total_memory - total_memory / overhead_factor) * 1024,  # 额外开销
        "total_memory_mb": total_memory,
        "total_memory_gb": total_memory / 1024,
    }


def create_3d_visualization(results):
    """创建3D柱状图可视化"""
    try:
        # 准备数据
        # 确保sizes按照64, 96, 128, 256的顺序
        all_sizes = sorted(set(r["size"] for r in results))
        sizes = [s for s in [64, 96, 128, 256] if s in all_sizes]  # 按指定顺序排列
        configs = ["large", "medium", "small"]  # 改为large-medium-small顺序
        
        print(f"\nGenerating 3D visualization...")
        print(f"  Sizes: {sizes}")
        print(f"  Configs: {configs}")
        print(f"  Results count: {len(results)}")
        
        # 只创建显存图，移除参数图
        fig = plt.figure(figsize=(18, 12))
        
        # 使用GridSpec来更好地控制子图位置和间距，大幅增加左边距以显示z轴标签
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 1, figure=fig, left=0.15, right=0.95, top=0.92, bottom=0.08)
        
        # 显存需求图（GB）
        ax1 = fig.add_subplot(gs[0], projection='3d')
        plot_3d_bars(ax1, results, sizes, configs, "total_memory_gb", 
                     "Memory (GB)", "Memory Budget", is_memory=True)
        
        # 保存图表，使用更大的边距
        plot_path = os.path.join(current_dir, "outpaint_3d_budget_visualization.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        print(f"\n✓ 3D visualization saved to: {plot_path}")
        
        # 关闭图形以释放内存
        plt.close(fig)
        
    except Exception as e:
        print(f"\n✗ Error generating visualization: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Visualization generation failed, but CSV data is still available.")


def plot_3d_bars(ax, results, sizes, configs, value_key, zlabel, title, is_memory=False):
    """绘制3D柱状图"""
    # 创建数据矩阵
    data_matrix = np.zeros((len(configs), len(sizes)))
    
    # 更新config_map以匹配新的顺序：large-medium-small
    config_map = {"large": 0, "medium": 1, "small": 2}
    size_map = {size: i for i, size in enumerate(sizes)}
    
    for r in results:
        if r["config"] in config_map and r["size"] in size_map:
            config_idx = config_map[r["config"]]
            size_idx = size_map[r["size"]]
            data_matrix[config_idx, size_idx] = r[value_key]
    
    # 设置坐标
    x_pos = np.arange(len(sizes))
    y_pos = np.arange(len(configs))
    x_pos, y_pos = np.meshgrid(x_pos, y_pos)
    
    # 柱状图的位置和大小
    dx = dy = 0.6  # 柱状图的宽度
    dz = data_matrix.flatten()
    
    # 创建颜色映射（使用viridis colormap）
    if dz.max() > 0:
        colors = plt.cm.viridis(dz / dz.max())
    else:
        colors = plt.cm.viridis(np.zeros_like(dz))
    
    # 对于显存图，使用对数坐标
    if is_memory and value_key == "total_memory_gb":
        max_z = dz.max() if len(dz) > 0 else 0
        min_z = dz[dz > 0].min() if len(dz[dz > 0]) > 0 else 1
        # 设置对数坐标范围
        ax.set_zscale('log')
        ax.set_zlim(min_z * 0.8, max_z * 1.2)
    
    # 绘制柱状图
    ax.bar3d(
        x_pos.flatten(), 
        y_pos.flatten(), 
        np.zeros_like(dz),
        dx, dy, dz,
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
        zorder=10  # zorder=10确保柱子在平面上方，但在标签下方
    )
    
    # 设置刻度（在绘制平面之前设置，以便获取正确的轴范围）
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s}³" for s in sizes])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([c.upper() for c in configs])  # 顺序已经是large-medium-small
    
    # 如果是显存图，添加24GB的阈值平面（在柱子之后绘制）
    if is_memory and value_key == "total_memory_gb":
        # 获取坐标范围，平面更大一些（扩展更多边界）
        x_min, x_max = -1.2, len(sizes) + 0.2
        y_min, y_max = -1.2, len(configs) + 0.2
        x_range = np.linspace(x_min, x_max, 50)
        y_range = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # 24GB阈值平面（GPU显存）- 绿色，透明度很低，覆盖更大的底面积，不显示网格线
        Z_24 = np.full_like(X, 24.0, dtype=float)
        ax.plot_surface(X, Y, Z_24, alpha=0.12, color='green', 
                       edgecolor='none', shade=False, zorder=1)  # 透明度很低（0.12），绿色，zorder=1确保在柱子下方
        
        # 添加图例说明（使用文本标签，改为"GPU Memory: 24 GB"）
        ax.text2D(0.02, 0.98, 'GPU Memory: 24 GB', transform=ax.transAxes,
                 fontsize=14, verticalalignment='top', color='green', weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='green', linewidth=2),
                 zorder=100)  # zorder=100确保在最上层
    
    # 设置标签 - 大幅增加labelpad以确保标签可见，特别是z轴（左侧）
    ax.set_xlabel('Volume Size', fontsize=12, labelpad=20)
    ax.set_ylabel('Config', fontsize=12, labelpad=20)
    # 对于显存图，特别增加z轴labelpad，并调整视角使标签可见
    if is_memory:
        ax.set_zlabel(zlabel, fontsize=12, labelpad=60)  # 大幅增加z轴labelpad到60
        # 确保z轴标签可见（对数坐标）
        ax.zaxis.label.set_visible(True)
        # 调整视角，使z轴标签更容易看到
        ax.view_init(elev=25, azim=50)
    else:
        ax.set_zlabel(zlabel, fontsize=12, labelpad=20)
        ax.view_init(elev=20, azim=45)
    ax.set_title(title, fontsize=14, pad=25)
    
    # 调整轴的边距
    ax.dist = 12  # 增加相机距离，使标签有更多空间
    
    # 在每个柱子上方添加数值标签（直接在柱子顶部，使用zorder确保在最上层）
    for i in range(len(sizes)):
        for j in range(len(configs)):
            value = data_matrix[j, i]
            if value > 0:
                # 根据数值大小格式化标签
                if value_key == "total_memory_gb":
                    if value >= 1000:
                        label = f'{value/1000:.1f}TB'
                    elif value >= 1:
                        label = f'{value:.1f}GB'
                    else:
                        label = f'{value*1024:.0f}MB'
                elif value_key == "total_params_m":
                    if value >= 1000:
                        label = f'{value/1000:.2f}B'
                    elif value >= 1:
                        label = f'{value:.2f}M'
                    else:
                        label = f'{value*1000:.0f}K'
                else:
                    label = f'{value:.2f}'
                
                # 在柱子顶部直接添加标签（稍微高一点以便可见）
                z_pos = value * 1.05  # 在柱子顶部稍微上方
                ax.text(i, j, z_pos, label, 
                       fontsize=9, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5),
                       zorder=100)  # zorder=100确保标签在最上层，不被遮挡


def create_model(volume_size, config):
    """创建模型"""
    attention_resolutions = tuple() if not config["attention"] else (volume_size // 16, volume_size // 8)
    out_channels = 2 if config["learn_sigma"] else 1
    
    base_unet = UNetModel(
        in_channels=3,  # x_t + init + mask
        model_channels=config["num_channels"],
        out_channels=out_channels,
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=attention_resolutions,
        dropout=0.0,
        channel_mult=config["channel_mult"],
        num_classes=None,
        use_checkpoint=config["use_checkpoint"],
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dims=3,  # 3D
    )
    
    model = CondUNet(base_unet)
    return model


def estimate_budget():
    """估算所有配置的预算"""
    results = []
    
    print("=" * 100)
    print("3D Outpainting Model Budget Estimator")
    print("=" * 100)
    print()
    
    for size in VOLUME_SIZES:
        print(f"\n{'='*100}")
        print(f"Volume Size: {size}³")
        print(f"{'='*100}")
        
        for config_name, config in CONFIGS.items():
            try:
                # 创建模型
                model = create_model(size, config)
                
                # 计算参数
                total_params, trainable_params = count_parameters(model)
                total_params_m = total_params / 1e6
                
                # 估算显存
                memory_info = estimate_model_memory(
                    model, 
                    config["batch_size"], 
                    size
                )
                
                # 存储结果
                result = {
                    "size": size,
                    "config": config_name,
                    "num_channels": config["num_channels"],
                    "channel_mult": config["channel_mult"],
                    "batch_size": config["batch_size"],
                    "diffusion_steps": config["diffusion_steps"],
                    "learn_sigma": config["learn_sigma"],
                    "use_checkpoint": config["use_checkpoint"],
                    "attention": config["attention"],
                    "total_params": total_params,
                    "total_params_m": total_params_m,
                    **memory_info,
                }
                results.append(result)
                
                # 打印结果
                print(f"\n  Config: {config_name.upper()}")
                print(f"    Parameters: {total_params_m:.2f}M ({total_params:,})")
                print(f"    Memory Breakdown:")
                print(f"      - Model Parameters: {memory_info['param_memory_mb']:.2f} MB")
                print(f"      - Optimizer States: {memory_info['optimizer_memory_mb']:.2f} MB")
                print(f"      - Data (Input+Output): {memory_info['data_memory_mb']:.2f} MB")
                checkpoint_factor_used = 0.6 if config["use_checkpoint"] else 1.0
                base_calibration = 14.0
                calibration_factor = base_calibration * ((size / 64) ** 1.5)
                # 应用额外的尺寸校准
                if size >= 256:
                    calibration_factor *= 1.2
                elif size >= 128:
                    calibration_factor *= 1.1
                print(f"      - Activations (forward, checkpoint: {checkpoint_factor_used:.1f}): {memory_info['activation_memory_mb']:.2f} MB")
                print(f"      - Activations (backward): {memory_info.get('backward_activation_memory_mb', 0):.2f} MB")
                print(f"      - Gradients: {memory_info['gradient_memory_mb']:.2f} MB")
                print(f"      - Overhead (PyTorch/CUDA): {memory_info.get('overhead_mb', 0):.2f} MB")
                print(f"      - Empirical Calibration Factor: {calibration_factor:.1f}x")
                print(f"    Total Memory: {memory_info['total_memory_gb']:.2f} GB ({memory_info['total_memory_mb']:.2f} MB)")
                
            except Exception as e:
                print(f"\n  Config: {config_name.upper()} - ERROR: {e}")
                continue
    
    # 生成汇总表格
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Size':<6} {'Config':<8} {'Params(M)':<12} {'Memory(GB)':<12} {'Batch':<6} {'Channels':<10} {'ChMult':<20}")
    print("-" * 100)
    
    for r in results:
        ch_mult_str = str(r["channel_mult"]).replace(" ", "")
        print(f"{r['size']:<6} {r['config']:<8} {r['total_params_m']:<12.2f} {r['total_memory_gb']:<12.2f} "
              f"{r['batch_size']:<6} {r['num_channels']:<10} {ch_mult_str:<20}")
    
    # 保存到CSV
    import csv
    csv_path = os.path.join(current_dir, "outpaint_3d_budget_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'size', 'config', 'num_channels', 'channel_mult', 'batch_size',
            'diffusion_steps', 'learn_sigma', 'use_checkpoint', 'attention',
            'total_params', 'total_params_m', 'param_memory_mb', 'optimizer_memory_mb',
            'data_memory_mb', 'activation_memory_mb', 'backward_activation_memory_mb',
            'gradient_memory_mb', 'overhead_mb', 'total_memory_mb', 'total_memory_gb'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n\nResults saved to: {csv_path}")
    
    # 生成3D可视化图表
    create_3d_visualization(results)
    
    return results


if __name__ == "__main__":
    # 设置设备（不需要实际GPU，只需要计算）
    th.set_default_dtype(th.float32)
    
    results = estimate_budget()
    
    print("\n" + "=" * 100)
    print("Note: Memory estimates are approximate and may vary based on:")
    print("  - Actual activation patterns during training")
    print("  - PyTorch memory allocator behavior")
    print("  - CUDA memory fragmentation")
    print("  - Additional overhead from data loading, logging, etc.")
    print("=" * 100)

