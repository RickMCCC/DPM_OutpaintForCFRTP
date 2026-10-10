# Guidance Scale Sweep (2026-08-22)

## Controlled setup

- Architecture: schema-v9 simple scratch model with context and structure guidance.
- VAE: `vae_3d_ckpt_22000.pt`, SHA256 recorded in each checkpoint.
- Training runs: identical seed, data order, validation seeds, losses, and architecture.
- Only training `control_gate_scale` differs: `0.9`, `1.2`, or `1.5`.
- Inference sweep: one fixed raw checkpoint (`gate=0.9`, step 138000), identical
  inputs and diffusion noise; only inference `control_gate_scale` differs.

## Completed training-scale comparison

The three runs all reached step 150001. Median metrics over steps 100000-150000
are more reliable than selecting one unusually favorable checkpoint.

| Training gate | Orientation error | Seam alignment | Boundary MSE | Unknown MSE | SSIM |
|---:|---:|---:|---:|---:|---:|
| 0.9 | 35.4860 | 0.7974 | 0.04446 | 0.05424 | 0.24877 |
| 1.2 | **34.9404** | **0.8071** | 0.04390 | **0.05328** | **0.24964** |
| 1.5 | 36.1692 | 0.7816 | **0.04379** | 0.05395 | 0.24877 |

The stable training range is therefore near `0.9-1.2`; `1.5` is already too
strong for orientation continuity despite a small boundary-MSE improvement.

## Fixed-checkpoint coarse inference sweep

| Inference scale | Orientation error | Seam alignment | Boundary MSE | Unknown MSE | SSIM | Saturation |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 48.9172 | 0.7279 | 0.05230 | 0.05974 | 0.23296 | 0.00282 |
| 0.3 | 36.8744 | 0.8198 | 0.04472 | 0.05537 | 0.24733 | 0.00201 |
| 0.6 | 35.2041 | 0.8295 | 0.04150 | 0.05175 | 0.24683 | 0.00146 |
| 0.9 | 34.2965 | 0.8454 | 0.03864 | 0.04829 | 0.24982 | **0.00118** |
| 1.2 | 33.7642 | **0.8477** | **0.03752** | **0.04628** | **0.25145** | 0.00119 |
| 1.5 | **33.6497** | 0.8456 | 0.03858 | 0.04631 | 0.25096 | 0.00142 |
| 1.8 | 33.9149 | 0.8411 | 0.04140 | 0.04808 | 0.24845 | 0.00180 |

The no-guidance result proves that the branch contributes materially. The
benefit saturates around `1.2-1.5`, then reverses by `1.8`.

## Four-sample fine inference sweep

| Inference scale | Orientation error | Seam alignment | Boundary MSE | Unknown MSE | SSIM | Saturation |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 33.7992 | 0.8442 | 0.03874 | 0.04809 | 0.25290 | **0.00121** |
| 1.1 | 33.6431 | **0.8446** | 0.03841 | 0.04746 | **0.25299** | 0.00122 |
| 1.2 | 33.5365 | 0.8443 | **0.03832** | 0.04704 | 0.25280 | 0.00126 |
| 1.3 | 33.4670 | 0.8438 | 0.03847 | **0.04685** | 0.25233 | 0.00132 |
| 1.4 | **33.4277** | 0.8431 | 0.03885 | 0.04689 | 0.25156 | 0.00141 |
| 1.5 | 33.4289 | 0.8420 | 0.03946 | 0.04714 | 0.25054 | 0.00153 |

## Recommendation

- Default training gate: `control_gate_scale=1.2`.
- Default inference gate: `1.2`.
- Direction-priority inference: `1.3`; the orientation gain above this is too
  small to justify lower SSIM, weaker seam alignment, and higher saturation.
- Do not use `>=1.5` as the default and do not continue increasing the scale.

The learned effective gates show that most useful control enters at the
64/32/16 latent scales; the middle gate remains nearly zero. The sign of an
individual scalar gate is not directly interpretable because its adapter can
flip sign with it.

## Reproduction

Run a fixed-checkpoint inference sweep:

```powershell
.\run_eval_guidance_scale_sweep.ps1 `
  -Checkpoint <schema-v9-checkpoint.pt> `
  -Scales @(1.0, 1.1, 1.2, 1.3, 1.4, 1.5) `
  -EvalNumSamples 4
```

Aggregate metrics and orthogonal slices with:

```powershell
python scripts\analyze_guidance_sweep.py `
  --run scale1.2=<scale-1.2-ckpt-directory> `
  --run scale1.3=<scale-1.3-ckpt-directory> `
  --output-dir outputs\guidance_sweep_analysis\comparison
```

Local reports:

- `outputs/guidance_sweep_analysis/v9_training_gate_09_12_15`
- `outputs/guidance_sweep_analysis/inference_gate_scale_20260822_164818`
- `outputs/guidance_sweep_analysis/inference_gate_fine_20260822_165846`

Raw sweep outputs are under
`Z:\Outpaint_Checkpoint\DPM_OutpaintForCFRTP\outputs\guidance_sweeps`.

The four validation cubes are deterministic but spatially related. Before a
paper-level conclusion, repeat `1.1/1.2/1.3` on non-overlapping validation
volumes from an independent source volume.
