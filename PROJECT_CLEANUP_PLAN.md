# Project Cleanup Plan

This document records a practical cleanup plan for the current
`DPM_OutpaintForCFRTP` repository. The goal is to separate source code,
datasets, experiment outputs, checkpoints, logs, and legacy material so the
project is easier to maintain, reproduce, and extend.

## Current Problems

The repository is not just "large"; several different kinds of files are mixed
together:

- Source code and experiment scripts.
- Raw and processed XCT / 3D outpainting data.
- Model checkpoints.
- Generated PNG samples and visualizations.
- Evaluation CSV files.
- Logs and local archives.
- Legacy 2D scripts.
- A nested `improved-diffusion/` repository copy.
- Local environment files and IDE metadata.

This creates several risks:

- It is hard to tell which scripts are current and which are historical.
- `scripts/` contains both runnable code and large generated artifacts.
- Checkpoints and sample images can grow into tens of GB inside the project.
- Hard-coded local paths make experiments difficult to reproduce elsewhere.
- The root README still describes upstream `improved-diffusion`, not this 3D
  CFRTP outpainting project.

## Recommended Target Structure

Use a structure like this:

```text
DPM_OutpaintForCFRTP/
  README.md
  LICENSE
  pyproject.toml
  .gitignore

  src/
    dpm_outpaint/
      diffusion/
      models/
      data/
      evaluation/
      visualization/

  scripts/
    prepare_3d_dataset.py
    train_vae_3d.py
    train_outpaint_voxel_3d.py
    train_outpaint_ldm_3d.py
    evaluate_3d.py
    visualize_3d.py

  configs/
    data/
    train/

  data/
    raw/
    processed/

  outputs/
    experiments/
    logs/

  docs/
    pipeline.md
    assets/

  archive/
    legacy_2d/
```

## Priority 1: Separate Code From Experiment Outputs

Move generated artifacts out of `scripts/`.

Current examples:

```text
scripts/outpaint_3d_checkpoint/
scripts/vae_3d_checkpoint/
outpaint_3d_ldm_checkpoint/
evaluation_results_size96_mask16_source1.576.csv
err.log
err3.log
scripts.zip
```

Suggested destinations:

```text
outputs/checkpoints/outpaint_3d/
outputs/checkpoints/vae_3d/
outputs/checkpoints/ldm_3d/
outputs/samples/
outputs/visualizations/
outputs/metrics/
outputs/logs/
archive/
```

Keep `scripts/` for entry-point scripts only.

## Priority 2: Clean Up the Root Directory

The root directory should mainly contain project-level files and top-level
folders.

Recommended root contents:

```text
README.md
LICENSE
.gitignore
pyproject.toml or setup.py
src/
scripts/
configs/
data/
outputs/
docs/
archive/
```

Recommended moves:

- `err.log`, `err3.log` -> `outputs/logs/`
- `evaluation_results_*.csv` -> `outputs/metrics/`
- `scripts.zip` -> `archive/` or delete if unnecessary
- `outpaint_3dnew/` -> `data/processed/`
- `outpaint_3d_ldm_checkpoint/` -> `outputs/checkpoints/ldm_3d/`
- `formerscripts/` -> `archive/legacy_scripts/`

## Priority 3: Resolve the Nested `improved-diffusion/` Copy

The repository currently has both:

```text
improved_diffusion/
improved-diffusion/
```

The nested `improved-diffusion/` folder appears to be a separate historical
copy with its own `.git` directory and many artifacts. This is confusing because
it is unclear which copy is the current source of truth.

Recommended option:

- Keep only the active `improved_diffusion/` package in this repository.
- Move the nested `improved-diffusion/` folder outside the project or into an
  explicit archive outside version control.

If it must stay for reference, rename and isolate it:

```text
vendor/improved-diffusion-upstream/
```

Then document that it is reference-only and not used by current training
scripts.

## Priority 4: Move Experiment Settings Into Config Files

Several scripts contain hard-coded local paths such as:

```text
F:\CFRTP XCT\nonwoven
C:\Terong\improved-diffusion\...
outpaint_3dnew/l2-1.485_train/size256_mask64_sourcel2-1.485
```

These should be moved into config files.

Suggested config layout:

```text
configs/
  data/
    l2_1485.yaml
    nonwoven.yaml
  train/
    vae_3d.yaml
    outpaint_voxel_3d.yaml
    outpaint_ldm_3d.yaml
```

Example:

```yaml
data:
  train_root: data/processed/l2-1.485/train/size256_mask64
  val_root: data/processed/l2-1.485/val/size256_mask64

output:
  root: outputs/experiments/ldm_l2_1485_mask64

model:
  latent_channels: 8
  num_channels: 320
  num_res_blocks: 4
```

Scripts should accept `--config` and keep command-line overrides for important
parameters.

## Priority 5: Give Each Experiment Its Own Directory

Use one folder per experiment run:

```text
outputs/experiments/
  2026-05-26_ldm_l2-1485_mask64/
    config.yaml
    checkpoints/
    samples/
    metrics/
    logs/
```

Benefits:

- Parameters, checkpoints, samples, and metrics stay together.
- Different experiments do not overwrite each other.
- Results are easier to compare and reproduce.
- It becomes simpler to write reports or papers from the experiment history.

## Priority 6: Separate Current 3D Scripts From Legacy 2D Scripts

The current project direction is 3D CFRTP outpainting, but there are still many
2D or upstream scripts.

Recommended organization:

```text
scripts/
  3d/
    prepare_3d_dataset.py
    train_vae_3d.py
    train_outpaint_voxel_3d.py
    train_outpaint_ldm_3d.py
    evaluate_3d.py
    visualize_3d.py

archive/
  legacy_2d/
  upstream_scripts/
```

Do not leave old and current scripts mixed together without labels.

## Priority 7: Rewrite the README

The current README is still the upstream `improved-diffusion` README. It should
be rewritten for this project.

Recommended README sections:

1. Project goal: 3D outpainting for CFRTP XCT volumes.
2. Data format: `init`, `full_mask`, `target` slice sequences.
3. Dataset preparation command.
4. 3D VAE training command.
5. 3D latent diffusion training command.
6. Evaluation and visualization commands.
7. Recommended directory structure.
8. Notes on checkpoints and large artifacts.

## Priority 8: Expand `.gitignore`

The current `.gitignore` is too small for an experiment-heavy ML project.

Recommended additions:

```gitignore
# Logs
*.log

# Local data
data/raw/
data/processed/

# Generated outputs
outputs/
runs/
checkpoints/
samples/
visualizations/

# Model artifacts
*.pt
*.pth
*.ckpt

# Generated image folders
*_pngs/

# Local archives
*.zip

# Python packaging
*.egg-info/
build/
dist/
```

If a small number of example images should be tracked, put them in:

```text
docs/assets/
```

## Priority 9: Improve Dependency Management

`setup.py` currently declares only a minimal set of dependencies. The actual
project also uses packages such as:

- `numpy`
- `Pillow`
- `scipy`
- `opencv-python`
- `scikit-image`
- `matplotlib`
- optional `lpips`

Recommended options:

- Add `requirements.txt` for basic training.
- Add `requirements-eval.txt` for evaluation and visualization.
- Eventually migrate to `pyproject.toml`.

Example:

```text
requirements.txt
requirements-eval.txt
requirements-dev.txt
```

## Low-Risk First Steps

If large restructuring feels risky while experiments are active, start with
these three steps:

1. Update `.gitignore` so new checkpoints, datasets, logs, and generated images
   do not enter version control.
2. Route all new training outputs to `outputs/experiments/<run_name>/`.
3. Rewrite README with the current 3D workflow and the intended directory
   layout.

These steps do not require changing model code or moving existing datasets
immediately, but they prevent the project from getting messier.

## Suggested Cleanup Order

1. Add or update `.gitignore`.
2. Create `outputs/`, `data/`, `configs/`, `docs/`, and `archive/`.
3. Stop writing new checkpoints into `scripts/`.
4. Move logs, metrics, samples, and checkpoints into `outputs/`.
5. Move processed datasets into `data/processed/`.
6. Archive or remove the nested `improved-diffusion/` copy.
7. Move hard-coded paths into config files.
8. Rewrite README.
9. Optionally refactor package code into `src/dpm_outpaint/`.

The key principle is simple: code should be small, tracked, and stable; data and
experiment outputs should be clearly separated, reproducible, and usually not
tracked by Git.
