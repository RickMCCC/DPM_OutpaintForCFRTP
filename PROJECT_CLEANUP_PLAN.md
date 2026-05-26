# Project Cleanup Plan / 项目清理计划

This document records a practical cleanup plan for the current
`DPM_OutpaintForCFRTP` repository. The goal is to separate source code,
datasets, experiment outputs, checkpoints, logs, and legacy material so the
project is easier to maintain, reproduce, and extend.

本文档记录了 `DPM_OutpaintForCFRTP` 仓库的清理计划，目标是将源代码、数据集、实验输出、checkpoint、日志和旧版材料分离，使项目更易于维护、复现和扩展。

## Current Problems / 当前问题

The repository is not just "large"; several different kinds of files are mixed
together:
仓库不只是"大"，多种不同类型的文件混杂在一起：

- Source code and experiment scripts. / 源代码和实验脚本。
- Raw and processed XCT / 3D outpainting data. / 原始和已处理的 XCT / 3D 外推数据。
- Model checkpoints. / 模型 checkpoint。
- Generated PNG samples and visualizations. / 生成的 PNG 样本和可视化。
- Evaluation CSV files. / 评估 CSV 文件。
- Logs and local archives. / 日志和本地存档。
- Legacy 2D scripts. / 旧版 2D 脚本。
- A nested `improved-diffusion/` repository copy. / 嵌套的 `improved-diffusion/` 仓库副本。
- Local environment files and IDE metadata. / 本地环境文件和 IDE 元数据。

This creates several risks / 这带来了几个风险：

- It is hard to tell which scripts are current and which are historical. / 难以区分哪些脚本是当前使用的，哪些是历史遗留的。
- `scripts/` contains both runnable code and large generated artifacts. / `scripts/` 同时包含可运行代码和大量生成产物。
- Checkpoints and sample images can grow into tens of GB inside the project. / checkpoint 和样本图片可能在项目内增长到数十 GB。
- Hard-coded local paths make experiments difficult to reproduce elsewhere. / 硬编码的本地路径使实验难以在其他地方复现。
- The root README still describes upstream `improved-diffusion`, not this 3D CFRTP outpainting project. / 根目录 README 仍然描述的是上游 `improved-diffusion`，而非此 3D CFRTP 外推项目。

## Recommended Target Structure / 推荐目标结构

Use a structure like this / 使用如下结构：

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

## Priority 1: Separate Code From Experiment Outputs / 优先级 1：分离代码与实验输出

Move generated artifacts out of `scripts/`. / 将生成的产物从 `scripts/` 中移出。

Current examples / 当前示例：

```text
scripts/outpaint_3d_checkpoint/
scripts/vae_3d_checkpoint/
outpaint_3d_ldm_checkpoint/
evaluation_results_size96_mask16_source1.576.csv
err.log
err3.log
scripts.zip
```

Suggested destinations / 建议目标：

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

Keep `scripts/` for entry-point scripts only. / `scripts/` 仅保留入口脚本。

## Priority 2: Clean Up the Root Directory / 优先级 2：清理根目录

The root directory should mainly contain project-level files and top-level
folders. / 根目录应主要包含项目级文件和顶层文件夹。

Recommended root contents / 推荐根目录内容：

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

Recommended moves / 推荐移动：

- `err.log`, `err3.log` -> `outputs/logs/`
- `evaluation_results_*.csv` -> `outputs/metrics/`
- `scripts.zip` -> `archive/` or delete if unnecessary / 或如不需要则删除
- `outpaint_3dnew/` -> `data/processed/`
- `outpaint_3d_ldm_checkpoint/` -> `outputs/checkpoints/ldm_3d/`
- `formerscripts/` -> `archive/legacy_scripts/`

## Priority 3: Resolve the Nested `improved-diffusion/` Copy / 优先级 3：处理嵌套的 `improved-diffusion/` 副本

The repository currently has both / 仓库当前同时存在：

```text
improved_diffusion/
improved-diffusion/
```

The nested `improved-diffusion/` folder appears to be a separate historical
copy with its own `.git` directory and many artifacts. This is confusing because
it is unclear which copy is the current source of truth.
嵌套的 `improved-diffusion/` 目录似乎是一个独立的历史副本，有自己的 `.git` 目录和大量产物。这令人困惑，因为不清楚哪个副本是当前的真实来源。

Recommended option / 推荐方案：

- Keep only the active `improved_diffusion/` package in this repository. / 在此仓库中仅保留活跃的 `improved_diffusion/` 包。
- Move the nested `improved-diffusion/` folder outside the project or into an explicit archive outside version control. / 将嵌套的 `improved-diffusion/` 目录移到项目外，或放入版本控制之外的显式归档中。

If it must stay for reference, rename and isolate it / 如果必须保留作为参考，重命名并隔离：

```text
vendor/improved-diffusion-upstream/
```

Then document that it is reference-only and not used by current training scripts. / 然后注明该目录仅供参考，当前训练脚本不使用。

## Priority 4: Move Experiment Settings Into Config Files / 优先级 4：将实验设置移入配置文件

Several scripts contain hard-coded local paths such as / 多个脚本包含硬编码的本地路径，例如：

```text
Z:\CFRTP XCT\nonwoven
C:\Terong\improved-diffusion\...
outpaint_3dnew/l2-1.485_train/size256_mask64_sourcel2-1.485
```

These should be moved into config files. / 这些应移入配置文件。

Suggested config layout / 推荐配置布局：

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

Example / 示例：

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

Scripts should accept `--config` and keep command-line overrides for important parameters. / 脚本应接受 `--config` 参数，并保留命令行覆盖重要参数的能力。

## Priority 5: Give Each Experiment Its Own Directory / 优先级 5：为每个实验分配独立目录

Use one folder per experiment run / 每个实验运行使用一个文件夹：

```text
outputs/experiments/
  2026-05-26_ldm_l2-1485_mask64/
    config.yaml
    checkpoints/
    samples/
    metrics/
    logs/
```

Benefits / 好处：

- Parameters, checkpoints, samples, and metrics stay together. / 参数、checkpoint、样本和指标集中存放。
- Different experiments do not overwrite each other. / 不同实验不会互相覆盖。
- Results are easier to compare and reproduce. / 结果更易比较和复现。
- It becomes simpler to write reports or papers from the experiment history. / 更易于根据实验历史编写报告或论文。

## Priority 6: Separate Current 3D Scripts From Legacy 2D Scripts / 优先级 6：分离当前 3D 脚本和旧版 2D 脚本

The current project direction is 3D CFRTP outpainting, but there are still many
2D or upstream scripts. / 当前项目方向是 3D CFRTP 外推，但仍有许多 2D 或上游脚本。

Recommended organization / 推荐组织方式：

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

Do not leave old and current scripts mixed together without labels. / 不要将旧脚本和当前脚本无标记地混在一起。

## Priority 7: Rewrite the README / 优先级 7：重写 README

The current README is still the upstream `improved-diffusion` README. It should
be rewritten for this project. / 当前 README 仍是上游 `improved-diffusion` 的 README，应为此项目重写。

Recommended README sections / 推荐 README 章节：

1. Project goal: 3D outpainting for CFRTP XCT volumes. / 项目目标：CFRTP XCT 体数据的 3D 外推。
2. Data format: `init`, `full_mask`, `target` slice sequences. / 数据格式：`init`、`full_mask`、`target` 切片序列。
3. Dataset preparation command. / 数据集准备命令。
4. 3D VAE training command. / 3D VAE 训练命令。
5. 3D latent diffusion training command. / 3D 潜在扩散训练命令。
6. Evaluation and visualization commands. / 评估和可视化命令。
7. Recommended directory structure. / 推荐目录结构。
8. Notes on checkpoints and large artifacts. / 关于 checkpoint 和大体积产物的说明。

## Priority 8: Expand `.gitignore` / 优先级 8：扩展 `.gitignore`

The current `.gitignore` is too small for an experiment-heavy ML project. / 当前 `.gitignore` 对于实验密集的 ML 项目来说太小了。

Recommended additions / 推荐添加：

```gitignore
# Logs / 日志
*.log

# Local data / 本地数据
data/raw/
data/processed/

# Generated outputs / 生成输出
outputs/
runs/
checkpoints/
samples/
visualizations/

# Model artifacts / 模型文件
*.pt
*.pth
*.ckpt

# Generated image folders / 生成的图片文件夹
*_pngs/

# Local archives / 本地压缩包
*.zip

# Python packaging / Python 打包
*.egg-info/
build/
dist/
```

If a small number of example images should be tracked, put them in / 如果需要跟踪少量示例图片，放入：

```text
docs/assets/
```

## Priority 9: Improve Dependency Management / 优先级 9：改进依赖管理

`setup.py` currently declares only a minimal set of dependencies. The actual
project also uses packages such as / `setup.py` 当前仅声明了最小依赖集，实际项目还使用了以下包：

- `numpy`
- `Pillow`
- `scipy`
- `opencv-python`
- `scikit-image`
- `matplotlib`
- optional / 可选 `lpips`

Recommended options / 推荐方案：

- Add `requirements.txt` for basic training. / 添加 `requirements.txt` 用于基础训练。
- Add `requirements-eval.txt` for evaluation and visualization. / 添加 `requirements-eval.txt` 用于评估和可视化。
- Eventually migrate to `pyproject.toml`. / 最终迁移到 `pyproject.toml`。

Example / 示例：

```text
requirements.txt
requirements-eval.txt
requirements-dev.txt
```

## Low-Risk First Steps / 低风险起步步骤

If large restructuring feels risky while experiments are active, start with
these three steps / 如果在实验活跃期间进行大规模重构有风险，可以从以下三步开始：

1. Update `.gitignore` so new checkpoints, datasets, logs, and generated images do not enter version control. / 更新 `.gitignore`，防止新的 checkpoint、数据集、日志和生成图片进入版本控制。
2. Route all new training outputs to `outputs/experiments/<run_name>/`. / 将所有新的训练输出路由到 `outputs/experiments/<run_name>/`。
3. Rewrite README with the current 3D workflow and the intended directory layout. / 用当前 3D 工作流和目标目录布局重写 README。

These steps do not require changing model code or moving existing datasets immediately, but they prevent the project from getting messier. / 这些步骤不需要立即修改模型代码或移动现有数据集，但能防止项目变得更加混乱。

## Suggested Cleanup Order / 建议清理顺序

1. Add or update `.gitignore`. / 添加或更新 `.gitignore`。
2. Create `outputs/`, `data/`, `configs/`, `docs/`, and `archive/`. / 创建 `outputs/`、`data/`、`configs/`、`docs/` 和 `archive/`。
3. Stop writing new checkpoints into `scripts/`. / 停止将新 checkpoint 写入 `scripts/`。
4. Move logs, metrics, samples, and checkpoints into `outputs/`. / 将日志、指标、样本和 checkpoint 移入 `outputs/`。
5. Move processed datasets into `data/processed/`. / 将已处理的数据集移入 `data/processed/`。
6. Archive or remove the nested `improved-diffusion/` copy. / 归档或移除嵌套的 `improved-diffusion/` 副本。
7. Move hard-coded paths into config files. / 将硬编码路径移入配置文件。
8. Rewrite README. / 重写 README。
9. Optionally refactor package code into `src/dpm_outpaint/`. / （可选）将包代码重构到 `src/dpm_outpaint/`。

The key principle is simple: code should be small, tracked, and stable; data and
experiment outputs should be clearly separated, reproducible, and usually not
tracked by Git.
核心原则很简单：代码应精简、受版本控制且稳定；数据和实验输出应明确分离、可复现，且通常不纳入 Git 跟踪。
