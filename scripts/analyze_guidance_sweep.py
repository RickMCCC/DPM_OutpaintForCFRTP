"""Aggregate fixed-seed guidance experiments and build visual comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw


METRIC_PATTERN = re.compile(r"val_sample_(\d+)_step_(\d+)_metrics\.json$")
PLOT_METRICS = (
    ("boundary_orientation_error", "Boundary orientation error", "lower"),
    ("seam_orientation_alignment", "Seam orientation alignment", "higher"),
    ("boundary_mse", "Boundary MSE", "lower"),
    ("unknown_mse", "Unknown MSE", "lower"),
    ("slice_ssim", "Slice SSIM", "higher"),
    ("bounded_latent_saturation_fraction", "Latent saturation", "lower"),
)


def parse_run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser()
    if not label.strip() or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid run: {value}")
    return label.strip(), path


def load_run(label: str, path: Path) -> dict:
    config_path = path / "config_used.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    rows = []
    for metrics_path in sorted(path.glob("val_sample_*_step_*_metrics.json")):
        match = METRIC_PATTERN.match(metrics_path.name)
        if match is None:
            continue
        values = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "label": label,
            "path": str(path),
            "sample": int(match.group(1)),
            "step": int(match.group(2)),
        }
        row.update(
            {
                key: float(value)
                for key, value in values.items()
                if isinstance(value, (int, float)) and value is not None
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError(f"no validation metrics found in {path}")
    return {
        "label": label,
        "path": path,
        "config": config,
        "gate_scale": float(config.get("control_gate_scale", "nan")),
        "rows": rows,
    }


def aggregate_steps(run: dict) -> list[dict]:
    grouped = defaultdict(list)
    for row in run["rows"]:
        grouped[row["step"]].append(row)
    aggregates = []
    for step, rows in sorted(grouped.items()):
        numeric_keys = set.intersection(
            *[
                {
                    key
                    for key, value in row.items()
                    if isinstance(value, (int, float)) and key not in {"sample", "step"}
                }
                for row in rows
            ]
        )
        aggregate = {
            "label": run["label"],
            "gate_scale": run["gate_scale"],
            "step": step,
            "sample_count": len(rows),
        }
        aggregate.update({key: mean(row[key] for row in rows) for key in numeric_keys})
        aggregates.append(aggregate)
    return aggregates


def select_best(run: dict, aggregates: list[dict]) -> dict:
    best_path = run["path"] / "best_checkpoint.json"
    if best_path.exists():
        selection = json.loads(best_path.read_text(encoding="utf-8"))
        step = int(selection["step"])
        matching = next((row for row in aggregates if row["step"] == step), None)
        if matching is not None:
            return matching
    return min(
        aggregates,
        key=lambda row: (
            row.get("boundary_orientation_error", math.inf),
            -row.get("seam_orientation_alignment", -math.inf),
        ),
    )


def summarize_tail(run: dict, aggregates: list[dict], start_step: int) -> dict:
    rows = [row for row in aggregates if row["step"] >= start_step]
    if not rows:
        rows = aggregates
    numeric_keys = set.intersection(
        *[
            {
                key
                for key, value in row.items()
                if isinstance(value, (int, float))
                and key not in {"gate_scale", "step", "sample_count"}
            }
            for row in rows
        ]
    )
    summary = {
        "label": run["label"],
        "gate_scale": run["gate_scale"],
        "start_step": min(row["step"] for row in rows),
        "end_step": max(row["step"] for row in rows),
        "checkpoint_count": len(rows),
    }
    summary.update({key: median(row[key] for row in rows) for key in numeric_keys})
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    preferred = [key for key in ("label", "gate_scale", "step", "sample", "sample_count") if key in keys]
    fieldnames = preferred + [key for key in keys if key not in preferred and key != "path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_value(row: dict, key: str) -> str:
    value = row.get(key)
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def write_report(
    path: Path,
    runs: list[dict],
    best_rows: list[dict],
    tail_rows: list[dict],
    common_steps: list[int],
) -> None:
    lines = [
        "# Guidance Sweep Report",
        "",
        "All validation rows use the saved fixed validation seeds and are averaged across samples.",
        "",
        "## Best Checkpoint Per Run",
        "",
        "| Run | Gate | Step | Orientation error | Seam alignment | Boundary MSE | Unknown MSE | SSIM | Saturation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            "| {label} | {gate} | {step} | {orientation} | {seam} | {boundary} | {unknown} | {ssim} | {saturation} |".format(
                label=row["label"],
                gate=metric_value(row, "gate_scale"),
                step=row["step"],
                orientation=metric_value(row, "boundary_orientation_error"),
                seam=metric_value(row, "seam_orientation_alignment"),
                boundary=metric_value(row, "boundary_mse"),
                unknown=metric_value(row, "unknown_mse"),
                ssim=metric_value(row, "slice_ssim"),
                saturation=metric_value(row, "bounded_latent_saturation_fraction"),
            )
        )
    lines.extend(
        [
            "",
            "## Late-Training Median",
            "",
            "This table is the more robust parameter comparison because it is not selected from a single lucky checkpoint.",
            "",
            "| Run | Gate | Window | Orientation error | Seam alignment | Boundary MSE | Unknown MSE | SSIM | Saturation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tail_rows:
        lines.append(
            "| {label} | {gate} | {start}-{end} | {orientation} | {seam} | {boundary} | {unknown} | {ssim} | {saturation} |".format(
                label=row["label"],
                gate=metric_value(row, "gate_scale"),
                start=row["start_step"],
                end=row["end_step"],
                orientation=metric_value(row, "boundary_orientation_error"),
                seam=metric_value(row, "seam_orientation_alignment"),
                boundary=metric_value(row, "boundary_mse"),
                unknown=metric_value(row, "unknown_mse"),
                ssim=metric_value(row, "slice_ssim"),
                saturation=metric_value(row, "bounded_latent_saturation_fraction"),
            )
        )
    lines.extend(["", "## Comparability", ""])
    reference = runs[0]["config"]
    ignored = {"control_gate_scale", "output_dir", "resume_checkpoint", "eval_only"}
    for run in runs[1:]:
        differences = sorted(
            key
            for key in set(reference) | set(run["config"])
            if key not in ignored and reference.get(key) != run["config"].get(key)
        )
        state = "comparable" if not differences else "differs: " + ", ".join(differences)
        lines.append(f"- `{runs[0]['label']}` vs `{run['label']}`: {state}")
    lines.extend(
        [
            "",
            "## Common Validation Steps",
            "",
            ", ".join(str(step) for step in common_steps) if common_steps else "None",
            "",
            "Checkpoint choice should prioritize boundary orientation continuity, then seam alignment, while rejecting regressions in unknown MSE, clipping, or latent saturation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(path: Path, aggregates_by_run: dict[str, list[dict]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (metric, title, direction) in zip(axes.flat, PLOT_METRICS):
        for label, rows in aggregates_by_run.items():
            points = [(row["step"], row.get(metric)) for row in rows if row.get(metric) is not None]
            axis.plot([point[0] for point in points], [point[1] for point in points], label=label, linewidth=1.5)
        axis.set_title(f"{title} ({direction})")
        axis.set_xlabel("Step")
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def central_orthogonal_image(run_path: Path, step: int, sample: int, axis: str) -> Path | None:
    directory = run_path / f"val_sample_{sample}_step_{step}_pngs" / "orthogonal"
    candidates = []
    for image_path in directory.glob(f"{axis}_*.png"):
        try:
            index = int(image_path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((index, image_path))
    if not candidates:
        return None
    midpoint = (min(index for index, _ in candidates) + max(index for index, _ in candidates)) / 2
    return min(candidates, key=lambda item: abs(item[0] - midpoint))[1]


def make_contact_sheet(path: Path, runs: list[dict], step_by_label: dict[str, int], sample: int = 0) -> bool:
    axes = ("z", "y", "x")
    tile = 256
    label_width = 230
    header_height = 38
    row_height = tile + 26
    canvas = Image.new("RGB", (label_width + tile * len(axes), header_height + row_height * len(runs)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, axis in enumerate(axes):
        draw.text((label_width + column * tile + 8, 10), axis.upper(), fill="black")
    found = False
    for row_index, run in enumerate(runs):
        step = step_by_label[run["label"]]
        y = header_height + row_index * row_height
        draw.text((8, y + 8), f"{run['label']}\ngate={run['gate_scale']:.3g}\nstep={step}", fill="black")
        for column, axis in enumerate(axes):
            image_path = central_orthogonal_image(run["path"], step, sample, axis)
            if image_path is None:
                continue
            with Image.open(image_path) as source:
                image = source.convert("RGB").resize((tile, tile))
            canvas.paste(image, (label_width + column * tile, y))
            found = True
    if found:
        canvas.save(path)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_run_spec, help="LABEL=checkpoint-directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--common-step", type=int, default=100000)
    parser.add_argument("--tail-start", type=int, default=100000)
    args = parser.parse_args()

    runs = [load_run(label, path) for label, path in args.run]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates_by_run = {run["label"]: aggregate_steps(run) for run in runs}
    aggregates = [row for rows in aggregates_by_run.values() for row in rows]
    best_rows = [select_best(run, aggregates_by_run[run["label"]]) for run in runs]
    tail_rows = [
        summarize_tail(run, aggregates_by_run[run["label"]], args.tail_start)
        for run in runs
    ]
    step_sets = [{row["step"] for row in aggregates_by_run[run["label"]]} for run in runs]
    common_steps = sorted(set.intersection(*step_sets))

    write_csv(output_dir / "all_metrics.csv", [row for run in runs for row in run["rows"]])
    write_csv(output_dir / "step_means.csv", aggregates)
    write_csv(output_dir / "best_checkpoints.csv", best_rows)
    write_csv(output_dir / "late_training_medians.csv", tail_rows)
    write_report(output_dir / "report.md", runs, best_rows, tail_rows, common_steps)
    make_plot(output_dir / "metric_curves.png", aggregates_by_run)
    make_contact_sheet(
        output_dir / "best_checkpoint_contact_sheet.png",
        runs,
        {row["label"]: row["step"] for row in best_rows},
    )
    if args.common_step in common_steps:
        make_contact_sheet(
            output_dir / f"step_{args.common_step}_contact_sheet.png",
            runs,
            {run["label"]: args.common_step for run in runs},
        )
    print(f"Wrote guidance analysis to {output_dir}")


if __name__ == "__main__":
    main()
