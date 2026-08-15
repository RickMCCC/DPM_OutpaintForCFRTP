# 模型结构与训练指南

> 最后更新: 2026-07-10 | 汇总所有改动后的完整文档

---

## 一、任务定义

**3D Outpainting**：给定体积中 8 个角落的已知子块（各 64³），生成中央未知区域（体积的 ~87.5%），完成完整的 256³ XCT 碳纤维增强热塑性塑料体积。

```
┌─────────────────────┐
│ ■ 已知   未知   ■  │   已知: 8 个角落, 各 64³
│                     │   未知: 中央区域, 需生成
│     ( 生成 )       │   总体积: 256³
│                     │
│ ■ 已知   未知   ■  │
└─────────────────────┘
```

---

## 二、两阶段训练流程

### Stage 1: 3D VAE — 学习压缩/重建

**目的**: 将 256³ 体积压缩到潜空间，降低 Stage 2 的计算量。

**检查点**: `vae_3d_ckpt_45000.pt`

| 参数 | 值 | 说明 |
|------|:---:|------|
| `in_channels` | 1 | 灰度 XCT 体积 |
| `latent_channels` | 8 | 潜空间通道数 |
| `hidden_channels` | 64 | Encoder/Decoder 隐藏层通道 |
| `num_res_blocks` | 2 | 残差块数量 |
| `num_downsamples` | **2** | 下采样次数: 256→128→64, 4× 压缩 |
| 潜空间尺寸 | **64³ × 8ch** | 262K 体素 (原体积 16.8M, 压缩比 ~16:1) |
| Batch size | 4 | |
| Learning rate | 1e-4 | |
| Crop size | 128 | 随机裁剪 128³ 训练 |
| KL weight | 0.001 | |
| Perceptual weight | 0.1 | LPIPS (VGG) |

**训练 Loss**: `L1 + MSE + 0.001×KL + 0.1×LPIPS`

**最终表现** (step ~48K): Total=0.038, Recon=0.038, KL=7.58, LPIPS=0.000

**⚠️ 关键改动**: Decoder 输出层**没有 tanh**。改为线性输出，避免扩散采样时潜变量偏移导致 tanh 饱和（见附录 A）。

### Stage 2: 3D Latent Diffusion — 学习生成

**目的**: 在 VAE 潜空间中训练扩散模型，从已知潜变量条件出发生成未知区域。

**配置**: `configs/outpaint_ldm_3d.yaml`

| 参数 | 值 | 说明 |
|------|:---:|------|
| `num_channels` | 64 | UNet 基础通道数 |
| `num_res_blocks` | 2 | 每层残差块 |
| `channel_mult` | (1, 2, 4, 4) | 4 层 UNet, 通道: 64→128→256→256 |
| `attention_resolutions` | "64,32" | 最深 2 层启用 attention (16³ 和 8³) |
| `cond_channels` | 9 | Cross-attn 条件: z_init(8ch)+mask(1ch) |
| `in_channels` | 17 | x_t(8ch) + cond(9ch), 通道拼接做局部条件 |
| `latent_channels` | 8 | 与 VAE 一致 |
| `vae_num_downsamples` | **2** | 与 VAE 一致, 潜空间 64³ |
| Diffusion steps | 1000 | Cosine noise schedule |
| Batch size | 2 | |
| Learning rate | 1e-4 | |
| `crop_size` / `eval_crop_size` | 256 | 完整体积训练 |
| `loss_unknown_weight` | 4.0 | 未知区域 loss 权重 ×4 |
| `vae_recon_weight` | 0.1 | VAE decoder 同步权重 |

**条件注入: 双向机制 (P0 修复)**

```
x_t (8ch)
    │
    ├── 通道拼接 [x_t, z_init, mask] = 17ch
    │       → 逐层 3D 卷积提供局部空间信息
    │
    └── Cross-Attention: z_init + mask → Context Encoder → K/V
            → Bottleneck 处的全局空间推理
```

---

## 三、UNet 架构详解 (Stage 2)

### 3.1 与 Stable Diffusion 的关键差异

| | **Stable Diffusion** | **本模型** |
|---|---|---|
| 维度 | 2D | **3D** |
| 下采样 stride | (2,2) | **(2,2,2)** — 全维度多尺度 |
| 条件注入 | Cross-Attention (text) | **通道拼接 + Cross-Attention** (spatial) |
| 通道倍数 | (1,2,4,8) | (1,2,4,4) |
| VAE | 冻结 | **可训练** (encoder 通过扩散 loss, decoder 通过重建 loss) |

### 3.2 UNet 层级结构 (潜空间 64³)

```
层级    ds    分辨率(D×H×W)     通道       Attention?
──────────────────────────────────────────────────
L0       1    64×64×64           64          ✗ (token数过大)
L1       2    32×32×32          128          ✗
L2       4    16×16×16          256          ✓ ds=4: 4096 tokens
L3       8     8×8×8            256          ✓ ds=8: 512 tokens
Middle       (bottleneck)       256          ✓ Self-Attn + Cross-Attn
```

每层有 2 个 ResBlock (`num_res_blocks=2`)。每层之间用 Conv3d stride=(2,2,2) 降采样。

### 3.3 Cross-Attention 实现

```python
CondUNet.forward(x_t, t, cond):
    x_in = cat([x_t, cond])       # 局部: 通道拼接
    return unet(x_in, t, cond=cond)  # 全局: cross-attention

UNetModel.forward(x, t, cond):
    context = context_encoder(cond)  # 9ch → 编码到 bottleneck 尺寸
    # ... 通过所有 input/middle/output blocks, context 传递给 CrossAttentionBlock
```

`ContextEncoder`: 逐层降采样 9→64→128→128ch (3 层 Downsample)，与 UNet bottleneck 空间匹配。

### 3.4 CrossAttentionBlock

```python
Q = Linear(LayerNorm(UNet_features))    # 来自 UNet
K, V = Linear(Context)                   # 来自 encoded condition
Attention(Q, K, V) → add residual
```

仅在 middle block 中使用（成本最高且全局信息最丰富的位置）。

### 3.5 总参数量: 72.3M

---

## 四、数据

### 4.1 数据来源

- 原始数据: `Z:/CFRTP XCT/1.576/` (TIFF 切片)
- 处理后数据集: `Z:/CFRTP XCT/processed/size256_mask64_sourcel2-1.485/`
  - `init/` — 已知角落子块 PNG
  - `full_mask/` — mask (未知=255, 已知=0)
  - `target/` — 完整画布 PNG
- 本地缓存 (.pt): `data/cache/size256_mask64_sourcel2-1.485/` (13GB, 64 个 .pt 文件)
- 验证集: 与训练集相同 (仅用于评估, 同一分布)

### 4.2 数据格式

每个 .pt 文件包含:
```python
{
    "name": "z0000_y0000_x0000_c256_s64",
    "init":  torch.float32  [1, 256, 256, 256],  # 已知角落子块 [-1,1]
    "mask":  torch.float32  [1, 256, 256, 256],  # 0=已知, 1=未知
    "target": torch.float32 [1, 256, 256, 256],  # 完整画布 [-1,1]
}
```

### 4.3 训练用缓存 (消除网络 I/O)

VAE 和 LDM 训练均使用 `CachedOutpainting3DDataset` (读取 .pt)，**不使用** PNG 网络盘读取。

---

## 五、训练命令

### Stage 1: VAE

```powershell
python scripts/train_vae_3d.py `
  --cache_dir "data/cache/size256_mask64_sourcel2-1.485" `
  --val_root "Z:/CFRTP XCT/processed/size256_mask64_sourcel2-1.485" `
  --latent_channels 8 --vae_hidden_channels 64 `
  --vae_num_res_blocks 2 --vae_num_downsamples 2 `
  --batch_size 4 --lr 1e-4 --crop_size 128 `
  --save_interval 5000 --epochs 3000 `
  --remote_output_dir "Z:/Outpaint_Checkpoint/DPM_OutpaintForCFRTP/outputs/checkpoints/vae_3d"
```

### Stage 2: LDM

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python scripts/outpaint_3d_train_ldm.py `
  --config configs/outpaint_ldm_3d.yaml `
  --val_root "Z:/CFRTP XCT/processed/size256_mask64_sourcel2-1.485" `
  --remote_output_dir "Z:/Outpaint_Checkpoint/DPM_OutpaintForCFRTP/outputs/checkpoints/outpaint_3d" `
  --vae_recon_weight 0.1 --batch_size 2
```

### 数据准备 (首次/扩展数据时)

```bash
# Step 1: 从 TIFF 生成三元组
python scripts/prepare_outpainting_3d_data.py \
  --input_dir "Z:/CFRTP XCT/1.576" \
  --output_dir "Z:/CFRTP XCT/processed/outpaint_3d" \
  --volume_size 256 --sub_size 64 \
  --stride_z 100 --stride_y 100 --stride_x 100

# Step 2: 缓存为本地 .pt
python scripts/cache_dataset.py \
  --source "Z:/CFRTP XCT/processed/size256_mask64_sourcel2-1.485" \
  --cache "data/cache/size256_mask64_sourcel2-1.485"
```

---

## 六、Checkpoint 与输出

### 本地 (快速 I/O)
- VAE: `outputs/checkpoints/vae_3d/<tag>/ckpt/vae_3d_ckpt_<step>.pt`
- LDM: `outputs/checkpoints/outpaint_3d/<tag>/ckpt/ldm_ckpt_<step>.pt`

### 远程 (Z 盘, 每次 save 后同步拷贝)
- VAE: `Z:/Outpaint_Checkpoint/.../vae_3d/<tag>/ckpt/`
- LDM: `Z:/Outpaint_Checkpoint/.../outpaint_3d/<tag>/ckpt/`

### 验证样本
- `.pt` 文件: `val_sample_{idx}_step_{step}.pt`
  - 包含: `{"init", "mask", "pred"}`
- PNG 切片: `val_sample_{idx}_step_{step}_pngs/slice_{ddd}.png`

### Checkpoint 内容
```python
{
    "model": ...,           # UNet 权重
    "vae": ...,             # VAE 权重
    "optimizer": ...,       # 优化器状态
    "step": N,              # 当前步数
}
```

---

## 七、采样过程

1. 加载 val sample 的 `init` + `mask`
2. VAE encode: `z_init = encode(init)`
3. 潜空间 mask 降采样: `mask_latent = interpolate(mask)`
4. 条件: `cond = [z_init, mask_latent]`
5. 1000 步 DDPM 反向采样
   - 每步 `denoised_fn`: `mask * x0 + (1-mask) * z_init`
   - 已知区域潜变量冻结, 未知区域自由去噪
6. VAE decode: `out = decode(z_final)`
7. 像素空间覆盖: `out = mask * out + (1-mask) * init` (已知区域用原始像素)
8. 保存 .pt + PNG

---

## 八、Loss 设计

### VAE Loss
```
Total = L1(recon, target) + MSE(recon, target)
        + 0.001 × KL(mu, logvar)
        + 0.1 × LPIPS(recon, target)
```

### LDM Loss
```
diffusion_mse = weighted_MSE(epsilon_pred, epsilon_true)
  权重: mask * 4.0 + (1-mask) * 1.0  (未知区域 4×)

vae_recon = 0.1 × MSE(vae.decode(z_target), target)

Total = diffusion_mse + vae_recon
```

---

## 九、已知问题与未来方向

### ✅ 已修复
| 问题 | 修复 |
|------|------|
| VAE decoder tanh 饱和 | 去掉 tanh, 线性输出 |
| UNet Z 轴不下采样 | stride 改为 (2,2,2) |
| Attention 不生效 | resolution 改为 "64,32" |
| 条件注入仅通道拼接 | 增加 cross-attention |
| 网络盘 I/O 瓶颈 | .pt 缓存 + 本地 checkpoint |

### 🔮 待优化
| 项目 | 方向 |
|------|------|
| VAE 可训练 vs 冻结 | 小数据集下建议冻结, 当前联合训练 |
| CFG (Classifier-Free Guidance) | 实现条件 dropout 训练 + CFG 采样 |
| channel_mult | 可尝试 (1,2,4,8) 增加最深一层容量 |
| 数据集 | 当前 64 个 volume, 可扩充或减小 stride 增加块数 |
| 位置编码 | 当前 mask 不区分空间位置, 加坐标编码可能有帮助 |
| 采样方式 | 考虑像素空间 denoised_fn 替代潜空间替换 |

---

## 附录 A: tanh 饱和问题的完整记录

### 现象
输出 PNG 全部为纯黑白 (-1 或 +1), 无任何纤维结构。

### 根因
1. VAE decoder 用 `torch.tanh` 钳制输出
2. 扩散采样产出的潜变量分布与 VAE encoder 不同
3. 稍大的潜变量通过 decoder 后中间层激活值 >3 或 <-3
4. tanh 饱和为 ±1 → 丢失所有细节

### 修复
`improved_diffusion/vae.py:134` — `torch.tanh(h)` → `h` (线性输出)

### 备注
主流 LDM (Stable Diffusion 等) 的 decoder 都不使用硬钳位激活函数。

---

## 附录 B: 文件索引

| 文件 | 用途 |
|------|------|
| `improved_diffusion/vae.py` | 3D VAE (encoder + decoder) |
| `improved_diffusion/unet.py` | 3D UNet + CrossAttention + Down/Upsample |
| `improved_diffusion/gaussian_diffusion.py` | DDPM 扩散过程 + weighted loss |
| `improved_diffusion/nn.py` | 基础层 (conv_nd, avg_pool_nd, normalization, checkpoint) |
| `scripts/train_vae_3d.py` | Stage 1 VAE 训练 |
| `scripts/outpaint_3d_train_ldm.py` | Stage 2 LDM 训练 (含 CondUNet) |
| `scripts/outpainting_3d_dataset.py` | CachedOutpainting3DDataset + Outpainting3DTripletDataset |
| `scripts/cache_dataset.py` | PNG → .pt 缓存脚本 |
| `scripts/prepare_outpainting_3d_data.py` | TIFF → PNG 三元组 |
| `configs/vae_3d.yaml` | VAE 参考配置 |
| `configs/outpaint_ldm_3d.yaml` | LDM 主配置 (运行时加载) |
| `docs/architecture-analysis.md` | 结构对比与潜在问题分析 |
| `docs/model-and-training-guide.md` | 本文档 |
