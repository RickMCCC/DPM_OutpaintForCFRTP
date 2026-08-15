# DPM Outpaint for CFRTP

3D outpainting (extrapolation) for CFRTP (Carbon Fiber Reinforced Thermoplastic) XCT volumes using diffusion probabilistic models.

This project trains a latent diffusion model (LDM) to perform 3D outpainting on XCT volumetric data. Given a known corner sub-volume, the model generates the surrounding structure to produce a larger complete volume.

## Environment

- Python 3.12 (PyTorch CUDA 不支持 Python 3.13)
- CUDA 12.6 + PyTorch 2.12.0+cu126
- NVIDIA RTX 4090 24GB

```bash
# 创建 venv 并从 PyTorch 官方源安装 CUDA 版本
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -r requirements.txt -r requirements-eval.txt
```

## Project Structure

```
DPM_OutpaintForCFRTP/
  README.md
  CLAUDE.md
  setup.py
  requirements.txt / requirements-eval.txt
  improved_diffusion/             # Core library (UNet, VAE, diffusion, utilities)
  scripts/                        # Entry-point scripts
    prepare_outpainting_3d_data.py   # PNG → triplet dataset
    cache_dataset.py                 # PNG → .pt cache (消除网络 I/O)
    train_vae_3d.py                  # Stage 1: VAE training
    outpaint_3d_train_ldm.py         # Stage 2: LDM training
    evaluate_3d_samples.py
    visualize_3d_results.py
  configs/                        # Reference configs (not loaded at runtime)
  data/                           # Local datasets and cache (git-ignored)
    raw/
    processed/
    cache/
  outputs/                        # Checkpoints, samples, metrics, logs (git-ignored)
    checkpoints/
    experiments/
  archive/                        # Legacy scripts and upstream reference
```

## Data Format

每个 3D 样本由 PNG 切片组成，目录结构如下：

```
dataset/size256_mask64_sourcel2-1.485/
  init/         # 已知角落子块（黑色背景 + 子块内容），每 Z 层一张 PNG
  full_mask/    # mask（未知区域=255，已知区域=0）
  target/       # 完整画布
```

命名格式：`z{起始坐标}_y{起始坐标}_x{起始坐标}_c{画布尺寸}_s{子块尺寸}_z{切片编号}.png`

## Data Preparation

### 1. 从原始 TIFF 切片生成三元组数据集

```bash
python scripts/prepare_outpainting_3d_data.py \
  --input_dir "Z:/CFRTP XCT/1.576" \
  --output_dir "Z:/CFRTP XCT/processed/outpaint_3d" \
  --volume_size 256 --sub_size 64 \
  --stride_z 100 --stride_y 100 --stride_x 100
```

Key options:
- `--volume_size`: 立方体画布边长 (256)
- `--sub_size`: 角落子块边长 / 已知区域 (64)
- `--feather`: mask 边缘羽化半径 (0)

### 2. 缓存为本地 .pt 文件（消除网络 I/O 瓶颈）

```bash
python scripts/cache_dataset.py \
  --source "Z:/CFRTP XCT/processed/size256_mask64_sourcel2-1.485" \
  --cache "data/cache/size256_mask64_sourcel2-1.485"
```

将每个 block 的 768 个 PNG 文件合并为单个 .pt 文件。训练时从本地 SSD 直接 `torch.load()`，比从网络驱动器逐文件读取快 50 倍以上。

## Training

### Stage 1: Train 3D VAE

```bash
python scripts/train_vae_3d.py \
  --data_dir "Z:/CFRTP XCT/processed/size256_mask64_sourcel2-1.485" \
  --val_root "Z:/CFRTP XCT/processed/size256_mask64_sourcel2-1.485" \
  --output_dir outputs/checkpoints/vae_3d \
  --latent_channels 8 \
  --vae_hidden_channels 32 \
  --vae_num_res_blocks 2 \
  --vae_num_downsamples 3 \
  --batch_size 4 --lr 1e-4
```

已训练好的 VAE checkpoint: `outputs/checkpoints/vae_3d/mask64_size256_sourcel2-1.485/ckpt/vae_3d_ckpt_92000.pt`

VAE 参数说明：
- `num_downsamples=3`: 8× 空间压缩 (256 → 32)
- `num_res_blocks=2`: 2 个残差块
- `hidden_channels=32`: 隐层通道数
- `latent_channels=8`: 潜空间通道数

### Stage 2: Train LDM (当前工作参数)

```powershell
# 减少显存碎片
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python scripts/outpaint_3d_train_ldm.py \
  --data_dir "Z:/CFRTP XCT/processed/size256_mask64_sourcel2-1.485" \
  --cache_dir "data/cache/size256_mask64_sourcel2-1.485" \
  --val_root "Z:/CFRTP XCT/processed/size256_mask64_sourcel2-1.485" \
  --vae_checkpoint "outputs/checkpoints/vae_3d/mask64_size256_sourcel2-1.485/ckpt/vae_3d_ckpt_92000.pt" \
  --num_channels 192 --num_res_blocks 4 \
  --lr 1e-4 \
  --vae_num_downsamples 3 --vae_num_res_blocks 2 \
  --use_fp16 True
```

LDM 参数说明：
| 参数 | 值 | 说明 |
|------|-----|------|
| `num_channels` | 192 | UNet 基础通道数（320 会爆 24GB 显存）|
| `num_res_blocks` | 4 | 残差块数量 |
| `vae_num_downsamples` | 3 | 必须与 VAE checkpoint 一致 |
| `vae_num_res_blocks` | 2 | 必须与 VAE checkpoint 一致 |
| `use_fp16` | True | 半精度训练，显存减半 |
| `crop_size` | 128 | 随机裁剪尺寸（代码默认值）|
| `batch_size` | 1 | 单样本训练 |

Checkpoint 保存到 `Z:\Outpaint_Checkpoint\mask64_size256_sourcel2-1.485_ds3_ch8_res2\ckpt\`。

### 训练上限

`max_steps = lr_anneal_steps if lr_anneal_steps > 0 else 100000`（默认 100,000 步）

## Evaluation

```bash
python scripts/evaluate_3d_samples.py \
  --recon_folder outputs/samples/my_experiment \
  --train_data_root data/processed/l2-1.485_val \
  --output_file outputs/metrics/results.csv
```

## Current Recommended LDM Run

Use the YAML-driven entry point for new LDM experiments:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python scripts/outpaint_3d_train_ldm.py `
  --config configs/outpaint_ldm_3d.yaml
```

The script reads `configs/outpaint_ldm_3d.yaml` and writes the resolved runtime
settings to `config_used.json` beside the checkpoint. Command-line flags still
override YAML values.

Important changes in the current LDM training path:

| Parameter | Recommended value | Reason |
|---|---:|---|
| `num_channels` | 32 | Reduces 3D activation and parameter memory |
| `num_res_blocks` | 2 | Adds local capacity for boundary continuation (19.11M parameters with guidance) |
| `attention_resolutions` | `""` | Uses attention only in the bottleneck |
| `learn_sigma` | true | Learns the reverse-process variance |
| `crop_size` | 256 | Train on complete outpainting geometry |
| `eval_crop_size` | 256 | Validate at the same volume scale as training |
| `loss_unknown_weight` | 4.0 | Region-normalized unknown loss weight |
| `loss_boundary_weight` | 0.2 | Extra supervision at the generated boundary |
| `voxel_boundary_width` | 12 | Decoded generated-side seam band in voxels |
| `voxel_boundary_max_timestep` | 250 | Applies decoded losses only at low-noise timesteps |
| `voxel_intensity_weight` | 0.5 | Matches decoded intensity near the seam |
| `voxel_normal_gradient_weight` | 0.25 | Matches derivatives normal to the init boundary |
| `voxel_structure_weight` | 0.1 | Matches normalized local 3D structure tensors |
| `eval_num_samples` | 2 | Avoid judging checkpoints from only one validation volume |
| `augment_flip` | true | Simple data augmentation for the small dataset |
| `use_multiscale_context` | true | Injects corner latent and mask features at 64/32/16/8 latent resolutions |
| `use_structure_guidance` | true/false | Enables real CFRTP structure guidance or the parameter-matched zero-channel ablation |
| `structure_window` | 5 | Aggregation window for the 3D structure tensor |
| `structure_propagation_latent` | 2 | Propagates valid boundary structure two latent voxels into the unknown region |
| `control_gate_scale` | 0.1 | Bounds each multi-scale residual with a learned `tanh` gate |
| `freeze_backbone_steps` | 2000 | Trains only Guidance and gates before unfreezing the v4 backbone |
| `backbone_lr` | `1e-5` | Uses one tenth of the Guidance learning rate after unfreezing |
| `ema_rate` | 0.9999 | Maintains the weights used for validation sampling |

The VAE is frozen during LDM training. Each of the eight known corner cubes is
encoded independently and assembled into a sparse condition latent, so zero-filled
unknown voxels cannot contaminate known features through the VAE receptive field.
Target latents are covariance-whitened and robust-quantile mapped to `[-1,1]`;
sampling clips in that bounded coordinate system and applies the exact inverse
transform before VAE decoding. Low-noise training steps additionally decode
`pred_x0` through the frozen VAE and supervise intensity, normal gradients, and
3D structure tensors in a generated-side boundary band. A lightweight guidance
pyramid injects the corner latent, mask, and optional structure tensor features
at 64/32/16/8 latent resolutions. Schema-v6 checkpoints store raw and EMA model
weights, guidance configuration, VAE identity, v4 backbone identity, and all latent
transforms. The v4 backbone and latent transform can be warm-started while the new
small adapters are initially disabled by zero-valued bounded gates.

Run the parameter-matched ablation pair from identical random initialization:

```powershell
.\run_train_256_ldm_context_v5.bat <vae-checkpoint.pt>
.\run_train_256_ldm_structure_v5.bat <vae-checkpoint.pt>
```

Both scripts use the same YAML configuration, training seed, data order, learning
rate schedule, and validation noise. The context-only run retains the complete
17-channel guidance input but sets all eight structure channels to zero. Validation
uses EMA weights, and `best_checkpoint.json` selects the lowest mean boundary
orientation error (using seam alignment as the tie-breaker) instead of the last step.

For the protected v4 warm-start path used by the current Structure experiment:

```powershell
.\run_train_256_ldm_structure_warmstart_v6.bat <vae-checkpoint.pt> <v4-backbone.pt>
```

Build isolated caches before training (the old combined cache contains validation
samples and is intentionally no longer referenced):

```powershell
python scripts/cache_dataset.py --source data/processed/l2-1.485_train/size256_mask64_sourcel2-1.485 --cache data/cache/l2-1.485_train/size256_mask64_sourcel2-1.485
python scripts/cache_dataset.py --source data/processed/l2-1.485_val/size256_mask64_sourcel2-1.485 --cache data/cache/l2-1.485_val/size256_mask64_sourcel2-1.485
```

Checkpoints are saved under:

```text
outputs/checkpoints/outpaint_3d/<run_tag>/ckpt/
```

If `--output_dir` already points to a directory named `ckpt`, the script uses it
directly.

## Visualization

```bash
python scripts/visualize_3d_results.py \
  --input_folder outputs/samples/my_experiment \
  --output_dir outputs/visualizations/my_experiment --all
```

## Experiment Outputs

每个实验运行使用独立目录：

```
outputs/experiments/<run_name>/
  config.yaml
  checkpoints/
  samples/
  metrics/
  logs/
```

数据、checkpoint 和生成产物不纳入 Git，分别存放在 Z 盘和 `outputs/` 目录下。
