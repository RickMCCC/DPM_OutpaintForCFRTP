# DPM Outpaint for CFRTP

3D outpainting (extrapolation) for CFRTP (Carbon Fiber Reinforced Thermoplastic) XCT volumes using diffusion probabilistic models.

This project trains a latent diffusion model to perform 3D outpainting on XCT volumetric data. Given a central sub-volume, the model generates the surrounding structure to produce a larger complete volume.

## Project Structure

```
DPM_OutpaintForCFRTP/
  README.md
  setup.py                        # Package install
  improved_diffusion/             # Core diffusion model code
  scripts/                        # Entry-point scripts
    prepare_outpainting_3d_data.py
    train_vae_3d.py
    outpaint_3d_train.py
    outpaint_3d_train_ldm.py
    evaluate_3d_samples.py
    visualize_3d_results.py
  configs/                        # Experiment configuration files
    data/
    train/
  data/                           # Datasets (git-ignored)
    raw/
    processed/
  outputs/                        # Experiment outputs (git-ignored)
    checkpoints/
    samples/
    visualizations/
    metrics/
    logs/
  archive/                        # Legacy scripts and reference material
```

## Installation

```bash
pip install -e .
```

For additional evaluation dependencies:

```bash
pip install numpy Pillow scipy opencv-python scikit-image matplotlib lpips
```

## Data Format

The 3D outpainting dataset uses XCT volumetric data. Each sample consists of three components stored as binary `.npy` files:

- **init**: The known center sub-volume (the "seed" region visible to the model).
- **target**: The full volume ground truth.
- **full_mask**: Binary mask indicating which voxels are known (the init region is 1, the surrounding unknown is 0).

The dataset directory should contain numbered subdirectories:

```
dataset/
  0/
    init.npy
    target.npy
    full_mask.npy
  1/
    init.npy
    target.npy
    full_mask.npy
  ...
```

## Preparing Data

Slice the original XCT TIFF stack into 3D training samples:

```bash
python scripts/prepare_outpainting_3d_data.py \
  --input_dir data/raw/tiff_stack \
  --output_dir data/processed/outpaint_l2_1485 \
  --volume_size 256 \
  --sub_size 64
```

Key options:
- `--volume_size`: Full volume side length.
- `--sub_size`: Size of the known center sub-volume (must be less than volume_size).
- `--feather`: Feathering width for the mask boundary.

## Training

### 1. Train the 3D VAE

The VAE compresses 3D volumes into a compact latent representation for efficient diffusion training:

```bash
python scripts/train_vae_3d.py \
  --data_dir data/processed/l2-1.485_train \
  --val_root data/processed/l2-1.485_val \
  --output_dir outputs/checkpoints/vae_3d \
  --latent_channels 8 \
  --vae_hidden_channels 64 \
  --vae_num_res_blocks 3 \
  --vae_num_downsamples 2 \
  --batch_size 4 \
  --epochs 200 \
  --lr 1e-4 \
  --kl_weight 0.001 \
  --perceptual_weight 0.1
```

### 2. Train the 3D Outpainting Latent Diffusion Model

With a trained VAE, train the latent diffusion model for outpainting:

```bash
python scripts/outpaint_3d_train_ldm.py \
  --train_root data/processed/l2-1.485_train \
  --val_root data/processed/l2-1.485_val \
  --out_dir_base outputs/experiments/ldm_outpaint \
  --vae_checkpoint outputs/checkpoints/vae_3d/best.pt \
  --batch_size 2 \
  --epochs 2000 \
  --lr 1e-4 \
  --volume_size 256 \
  --num_channels 320 \
  --num_res_blocks 4
```

### 3. Train the 3D Outpainting Voxel Model (Direct)

Train a diffusion model directly on voxel data (no VAE compression):

```bash
python scripts/outpaint_3d_train.py \
  --train_root data/processed/l2-1.485_train \
  --val_root data/processed/l2-1.485_val \
  --out_dir_base outputs/experiments/voxel_outpaint \
  --batch_size 2 \
  --epochs 2000 \
  --lr 1e-4 \
  --volume_size 256 \
  --num_channels 128 \
  --num_res_blocks 3 \
  --diffusion_steps 1000 \
  --noise_schedule cosine
```

## Evaluation

Compute image quality metrics (MSE, PSNR, SSIM) on generated volumes:

```bash
python scripts/evaluate_3d_samples.py \
  --recon_folder outputs/samples/my_experiment \
  --train_data_root data/processed/l2-1.485_val \
  --output_file outputs/metrics/my_experiment_metrics.csv
```

## Visualization

Generate 2D slice plots and 3D voxel/surface visualizations from model outputs:

```bash
# 2D slices with adaptive thresholding
python scripts/visualize_3d_results.py \
  --input_folder outputs/samples/my_experiment \
  --output_dir outputs/visualizations/my_experiment \
  --visualize_2d_slices \
  --threshold_method adaptive_percentile \
  --percentile 75

# 3D voxel rendering
python scripts/visualize_3d_results.py \
  --input_folder outputs/samples/my_experiment \
  --output_dir outputs/visualizations/my_experiment \
  --visualize_3d_voxels \
  --threshold_method absolute \
  --fiber_threshold 128

# All visualizations
python scripts/visualize_3d_results.py \
  --input_folder outputs/samples/my_experiment \
  --output_dir outputs/visualizations/my_experiment \
  --all
```

## Experiment Management

Each experiment should produce outputs under its own directory:

```
outputs/experiments/
  2026-05-26_ldm_l2-1485_mask64/
    config.yaml
    checkpoints/
    samples/
    metrics/
    logs/
```

## Notes

- Checkpoints, generated samples, and metrics are in `outputs/` (git-ignored).
- Raw and processed datasets live under `data/` (git-ignored).
- Legacy 2D diffusion scripts and the upstream `improved-diffusion` reference copy are in `archive/`.
- Hard-coded paths in scripts should be gradually migrated to config files under `configs/`.
