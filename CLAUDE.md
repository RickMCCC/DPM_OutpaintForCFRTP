# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D outpainting (extrapolation) for CFRTP XCT volumes using diffusion probabilistic models. Given a known central sub-volume, the model generates surrounding structure. Forked from OpenAI's `improved-diffusion` and modified for 3D volumetric data.

Two training approaches:
- **Direct voxel diffusion** (`outpaint_3d_train.py`): UNet operates on 3D voxels. Simpler, uses more GPU memory.
- **Latent diffusion / LDM** (`outpaint_3d_train_ldm.py`): 3D VAE compresses volumes first, diffusion runs in latent space. Required for large volumes (256³).

## Key Architectural Details

### 3D UNet modifications (`improved_diffusion/unet.py`)

The UNet takes a `dims` parameter. When `dims=3`, all convs use `nn.Conv3d`/`nn.AvgPool3d`. Critical difference from 2D:

- **Downsample**: stride is `(1, 2, 2)` — depth dimension is never downsampled, only H and W.
- **Upsample**: only upsamples H and W via `F.interpolate(..., mode="nearest")`.
- This means the depth resolution stays constant throughout the entire UNet.

### CondUNet wrapper (`outpaint_3d_train.py`)

The direct voxel training script wraps UNetModel in `CondUNet`, which concatenates `[x_t, init, full_mask]` along channels → `in_channels=3`. The LDM script does not use this wrapper; it creates the UNet directly with `in_channels = latent_channels * 2 + 1`.

### Import pattern

All scripts add the project root to `sys.path`:
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
This lets them `import improved_diffusion` without installing the package. `pip install -e .` also works but is not required for scripts to run.

### VAE (`improved_diffusion/vae.py`)

3D convolutional VAE. Encoder downsamples by `2^num_downsamples` (spatially). Default: `num_downsamples=2` → 4× spatial compression per dimension. Latent channels default to 8. Decoder uses `nn.ConvTranspose3d`.

### Config files are reference only

The YAML files under `configs/` are templates. Scripts do **not** load them — they use argparse with hardcoded defaults. The configs document what parameters were used for specific experiments.

## Common Commands

### Installation
```bash
pip install -e .
pip install -r requirements.txt        # Core: torch, numpy, blobfile, tqdm, Pillow, mpi4py
pip install -r requirements-eval.txt    # Adds: opencv-python, scipy, scikit-image, matplotlib
pip install -r requirements-dev.txt     # Adds: lpips (for VAE perceptual loss)
```

### Data preparation
```bash
python scripts/prepare_outpainting_3d_data.py \
  --input_dir data/raw/tiff_stack \
  --output_dir data/processed/my_dataset \
  --volume_size 256 --sub_size 64 --feather 0
```
Output: `init/`, `full_mask/`, `target/` directories of PNG slices.

### Stage 1: Train 3D VAE
```bash
python scripts/train_vae_3d.py \
  --data_dir data/processed/l2-1.485_train \
  --val_root data/processed/l2-1.485_val \
  --output_dir outputs/checkpoints/vae_3d \
  --latent_channels 8 --batch_size 4 --lr 1e-4
```

### Stage 2a: Train LDM (requires trained VAE)
```bash
python scripts/outpaint_3d_train_ldm.py \
  --data_dir data/processed/l2-1.485_train \
  --val_root data/processed/l2-1.485_val \
  --vae_checkpoint outputs/checkpoints/vae_3d/best.pt \
  --output_dir outputs/experiments/ldm_outpaint \
  --num_channels 320 --num_res_blocks 4 --lr 1e-4
```

### Stage 2b: Train direct voxel model (no VAE needed)
```bash
python scripts/outpaint_3d_train.py \
  --train_root data/processed/l2-1.485_train \
  --val_root data/processed/l2-1.485_val \
  --out_dir_base outputs/experiments/voxel_outpaint \
  --num_channels 128 --num_res_blocks 3 --lr 1e-4
```

### Evaluation
```bash
python scripts/evaluate_3d_samples.py \
  --recon_folder outputs/samples/my_experiment \
  --train_data_root data/processed/l2-1.485_val \
  --output_file outputs/metrics/results.csv
```

### Visualization
```bash
python scripts/visualize_3d_results.py \
  --input_folder outputs/samples/my_experiment \
  --output_dir outputs/visualizations/my_experiment --all
```

## Windows-Specific Notes

- **mpi4py**: Requires MS-MPI installed separately on Windows. Importing `improved_diffusion.dist_util` will fail without it, though it's only needed for distributed training.
- **`KMP_DUPLICATE_LIB_OK`**: `evaluate_3d_samples.py` sets this env var to suppress OpenMP conflicts on Windows.
- **CUDA memory**: `outpaint_3d_train.py` recommends `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (Windows `set` syntax, not `export`).
- **Line endings**: The repo uses LF; git may warn about CRLF conversion on Windows. This is harmless.

## Architecture

```
improved_diffusion/      # Core library (UNet, VAE, diffusion process, utilities)
scripts/                 # Entry-point scripts (data prep, train, eval, visualize)
configs/                 # YAML config templates (reference, not loaded by scripts)
data/                    # Datasets (gitignored)
outputs/                 # Checkpoints, samples, metrics, logs (gitignored)
archive/                 # Legacy 2D scripts, upstream reference copy, old experiments
```

## Key Design Decisions

- **No tests exist**. There is no test suite, pytest config, or CI.
- **3D attention is disabled by default** (`attention_resolutions=""`) because `O((D*H*W)²)` is prohibitive for volumetric data.
- **Gradient checkpointing** (`--use_checkpoint True`) is on by default in both training scripts to manage GPU memory.
- **Channel multipliers** in LDM are large (`channel_mult=(1,2,4,4)`) compared to the voxel model, reflecting the expectation that the latent space needs more capacity.
- **The VAE is frozen** during LDM training. A `latent_scale_factor` is auto-computed from the validation set to normalize latent variance to ~1.0 before diffusion training.
- During LDM sampling, after each denoising step, the known region is overwritten with the ground truth latent init values (the `denoised_fn` pattern).
