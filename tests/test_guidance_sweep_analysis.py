import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_guidance_sweep import aggregate_steps, load_run, select_best


class GuidanceSweepAnalysisTests(unittest.TestCase):
    def test_aggregates_samples_and_honors_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            (run_dir / "config_used.json").write_text(
                json.dumps({"control_gate_scale": 0.9}), encoding="utf-8"
            )
            for sample, orientation, seam in ((0, 35.0, 0.8), (1, 33.0, 0.9)):
                (run_dir / f"val_sample_{sample}_step_2000_metrics.json").write_text(
                    json.dumps(
                        {
                            "boundary_orientation_error": orientation,
                            "seam_orientation_alignment": seam,
                            "unknown_mse": 0.05,
                        }
                    ),
                    encoding="utf-8",
                )
            (run_dir / "best_checkpoint.json").write_text(
                json.dumps({"step": 2000}), encoding="utf-8"
            )

            run = load_run("gate09", run_dir)
            aggregates = aggregate_steps(run)
            best = select_best(run, aggregates)

            self.assertEqual(len(aggregates), 1)
            self.assertEqual(best["step"], 2000)
            self.assertEqual(best["sample_count"], 2)
            self.assertAlmostEqual(best["boundary_orientation_error"], 34.0)
            self.assertAlmostEqual(best["seam_orientation_alignment"], 0.85)
            self.assertAlmostEqual(best["gate_scale"], 0.9)


if __name__ == "__main__":
    unittest.main()
