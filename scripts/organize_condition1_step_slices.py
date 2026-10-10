"""Organize seven fixed, spaced Condition-1 slices for checkpoint comparison."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_NAME = "z0300_y0300_x0315_c256_s64"
LOCAL_Z = (32, 64, 96, 128, 160, 192, 224)
STEPS = (0, 2000, 4000, 6000, 8000, *range(10000, 150001, 10000))


def source_for(step: int, z_index: int, checkpoint_dir: Path, data_root: Path) -> Path:
    if step == 0:
        return data_root / "init" / f"{SAMPLE_NAME}_z{z_index:03d}.png"
    return checkpoint_dir / f"val_sample_1_step_{step}_pngs" / f"slice_{z_index:03d}.png"


def save_contact_sheet(paths: list[Path], labels: list[str], output: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(paths), figsize=(17.5, 3.2))
    for axis, path, label in zip(axes, paths, labels):
        axis.imshow(Image.open(path), cmap="gray", vmin=0, vmax=255)
        axis.set_title(label, fontsize=14)
        axis.axis("off")
    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    checkpoint_dir = Path(
        r"Z:\Outpaint_Checkpoint\DPM_OutpaintForCFRTP\outputs\checkpoints\outpaint_3d\mask64_size256_sourcel2-1.485_ds2_ch8_res2_simple_scratch_v9_gate09_20260808_201046\ckpt"
    )
    data_root = PROJECT_ROOT / "data/processed/l2-1.485_val/size256_mask64_sourcel2-1.485"
    output_root = PROJECT_ROOT / "outputs/condition1_step_slices_z32_spacing"
    output_root.mkdir(parents=True, exist_ok=True)

    labels = [f"local Z{z:03d}\nglobal Z{300 + z:03d}" for z in LOCAL_Z]
    for step in STEPS:
        step_dir = output_root / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for z_index in LOCAL_Z:
            source = source_for(step, z_index, checkpoint_dir, data_root)
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = step_dir / f"condition1_local_z{z_index:03d}_global_z{300 + z_index:03d}.png"
            shutil.copy2(source, destination)
            copied.append(destination)
        description = "masked-input baseline" if step == 0 else f"checkpoint step {step:,}"
        save_contact_sheet(
            copied,
            labels,
            step_dir / "contact_sheet.png",
            f"Condition 1 — {description}",
        )

    target_dir = output_root / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    for z_index in LOCAL_Z:
        source = data_root / "target" / f"{SAMPLE_NAME}_z{z_index:03d}.png"
        destination = target_dir / f"target_local_z{z_index:03d}_global_z{300 + z_index:03d}.png"
        shutil.copy2(source, destination)
        targets.append(destination)
    save_contact_sheet(targets, labels, target_dir / "contact_sheet.png", "Condition 1 — Target")

    manifest = {
        "sample": "condition1",
        "sample_name": SAMPLE_NAME,
        "sample_origin_zyx": [300, 300, 315],
        "local_z_indices": list(LOCAL_Z),
        "global_z_indices": [300 + z for z in LOCAL_Z],
        "slice_spacing": 32,
        "steps": list(STEPS),
        "step_0_definition": "masked input baseline",
        "files_per_step": 7,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_root)


if __name__ == "__main__":
    main()
