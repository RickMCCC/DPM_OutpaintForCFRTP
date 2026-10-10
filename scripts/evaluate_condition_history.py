"""Evaluate the two fixed validation conditions over checkpoint history.

This reuses the legacy E_px_3d, E_G_3d and FID_3d definitions, while resizing
the current 256^3 volumes to 96^3 so the scale matches the earlier evaluation.
Step 0 is the masked input baseline because no step-0 checkpoint was saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_3d_samples import (  # noqa: E402
    compute_E_G_3d,
    compute_E_px_3d,
    compute_fid_3d,
)


CONDITION_NAMES = (
    "z0300_y0300_x0300_c256_s64",
    "z0300_y0300_x0315_c256_s64",
)


def load_png_volume(folder: Path, output_size: int = 96) -> np.ndarray:
    files = sorted(folder.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG slices found in {folder}")
    slices = []
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read {path}")
        image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
        slices.append(image.astype(np.float32) / 255.0)
    volume = np.stack(slices)
    if volume.shape[0] != output_size:
        indices = np.linspace(0, volume.shape[0] - 1, output_size).round().astype(int)
        volume = volume[indices]
    return volume


def load_named_volume(folder: Path, name: str, output_size: int = 96) -> np.ndarray:
    files = sorted(folder.glob(f"{name}_z*.png"))
    if not files:
        raise FileNotFoundError(f"No slices for {name} in {folder}")
    slices = []
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
        slices.append(image.astype(np.float32) / 255.0)
    volume = np.stack(slices)
    indices = np.linspace(0, volume.shape[0] - 1, output_size).round().astype(int)
    return volume[indices]


def evaluate(recon: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    recon_list, reference_list = [recon], [reference]
    return {
        "E_px_3d": float(compute_E_px_3d(recon_list, reference_list)),
        "E_G_3d": float(compute_E_G_3d(recon_list, reference_list)),
    }


def plot_results(rows: list[dict[str, object]], output: Path) -> None:
    metrics = ("E_px_3d", "E_G_3d", "FID_3d")
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for axis, metric in zip(axes, metrics):
        selected = [row for row in rows if row["condition"] == 1]
        axis.plot(
            [row["step"] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            linewidth=2.8,
            markersize=7,
            color="#1B998B",
        )
        axis.set_ylabel(metric, fontsize=17)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_title(
        "Condition 1 generation metrics, step 0-150,000 (lower is better)",
        fontsize=20,
        pad=14,
    )
    axes[-1].set_xlabel(
        "Training step (step 0 = masked-input baseline)", fontsize=17
    )
    axes[-1].set_xticks(range(0, 150001, 10000))
    axes[-1].tick_params(axis="x", rotation=45, labelsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data/processed/l2-1.485_val/size256_mask64_sourcel2-1.485",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    references = [
        load_named_volume(args.data_root / "target", name) for name in CONDITION_NAMES
    ]
    baselines = [
        load_named_volume(args.data_root / "init", name) for name in CONDITION_NAMES
    ]
    rows: list[dict[str, object]] = []
    for step in range(0, 150001, 10000):
        step_recons = []
        step_sources = []
        for condition, name in enumerate(CONDITION_NAMES):
            if step == 0:
                recon = baselines[condition]
                source = "masked_input_baseline"
            else:
                source_dir = args.checkpoint_dir / f"val_sample_{condition}_step_{step}_pngs"
                recon = load_png_volume(source_dir)
                source = str(source_dir)
            step_recons.append(recon)
            step_sources.append(source)

        # FID is a distribution statistic. Computing it from one condition makes
        # the legacy implementation degenerate to zero, so use both conditions.
        np.random.seed(20260713 + step)
        step_fid = float(compute_fid_3d(step_recons, references))
        for condition, name in enumerate(CONDITION_NAMES):
            recon = step_recons[condition]
            values = evaluate(recon, references[condition])
            row = {
                "step": step,
                "condition": condition,
                "condition_name": name,
                "source": step_sources[condition],
                **values,
                "FID_3d": step_fid,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    csv_path = args.output_dir / "condition_metrics_step_0_150000.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(rows, args.output_dir / "condition_metrics_step_0_150000.png")
    metadata = {
        "metric_implementation": "scripts/evaluate_3d_samples.py legacy definitions",
        "comparison": "paired generated volume versus the reference for the same condition",
        "evaluation_resolution": [96, 96, 96],
        "step_0_definition": "masked initial condition; no step-0 checkpoint exists",
        "steps": list(range(0, 150001, 10000)),
        "lower_is_better": ["E_px_3d", "E_G_3d", "FID_3d"],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
