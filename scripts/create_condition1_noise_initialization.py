"""Create a reproducible diffusion-noise initialization visualization."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_NAME = "z0300_y0300_x0315_c256_s64"
LOCAL_Z = (32, 64, 96, 128, 160, 192, 224)
SEED = 20260713


def load_volume(folder: Path) -> np.ndarray:
    paths = sorted(folder.glob(f"{SAMPLE_NAME}_z*.png"))
    if len(paths) != 256:
        raise ValueError(f"Expected 256 slices in {folder}, found {len(paths)}")
    return np.stack([np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in paths])


def main() -> None:
    data_root = PROJECT_ROOT / "data/processed/l2-1.485_val/size256_mask64_sourcel2-1.485"
    output_dir = PROJECT_ROOT / "outputs/condition1_step_slices_z32_spacing/noise_initialization"
    output_dir.mkdir(parents=True, exist_ok=True)

    init = load_volume(data_root / "init")
    mask = load_volume(data_root / "full_mask") > 127
    rng = np.random.default_rng(SEED)

    # This mirrors visual x_T initialization: N(0,1), clipped to [-1,1],
    # then mapped to uint8. Known corner voxels are restored from init.
    noise = np.clip(rng.standard_normal(init.shape, dtype=np.float32), -1.0, 1.0)
    noise_u8 = np.rint((noise + 1.0) * 127.5).astype(np.uint8)
    visualization = np.where(mask, noise_u8, init).astype(np.uint8)

    paths = []
    for z_index in LOCAL_Z:
        path = output_dir / f"noise_init_local_z{z_index:03d}_global_z{300 + z_index:03d}.png"
        Image.fromarray(visualization[z_index], mode="L").save(path)
        paths.append(path)

    fig, axes = plt.subplots(1, 7, figsize=(17.5, 3.2))
    for axis, path, z_index in zip(axes, paths, LOCAL_Z):
        axis.imshow(Image.open(path), cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"local Z{z_index:03d}\nglobal Z{300 + z_index:03d}", fontsize=14)
        axis.axis("off")
    fig.suptitle("Condition 1 — Noise initialization (not a checkpoint output)", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_dir / "contact_sheet.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "sample": "condition1",
        "sample_name": SAMPLE_NAME,
        "type": "noise_initialization_visualization",
        "checkpoint_output": False,
        "seed": SEED,
        "noise": "standard normal clipped to [-1, 1] and mapped to [0, 255]",
        "known_region": "copied from init where full_mask == 0",
        "unknown_region": "noise where full_mask == 255",
        "local_z_indices": list(LOCAL_Z),
        "global_z_indices": [300 + z for z in LOCAL_Z],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
