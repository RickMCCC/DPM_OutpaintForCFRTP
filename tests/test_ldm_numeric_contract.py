import unittest
import copy
import io
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from improved_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from scripts.outpaint_3d_train_ldm import (
    CondUNet,
    assert_datasets_disjoint,
    bounded_to_latent,
    compute_latent_stats,
    encode_known_corners,
    encode_corner_structure_guidance,
    freeze_vae,
    known_patch_structure_tensor,
    latent_to_bounded,
    latent_mask_from_voxel,
    optimizer_parameter_groups,
    propagate_structure_guidance,
    scratch_curriculum_state,
    scheduled_learning_rate,
    set_backbone_trainable,
    structure_orientation_field,
    update_best_checkpoint_selection,
    update_ema_model,
    voxel_boundary_losses,
    whiten_latent,
)
from scripts.outpainting_3d_dataset import sanitize_triplet
from improved_diffusion.unet import UNetModel


class _NamedDataset:
    def __init__(self, names):
        self.names = names


class _VolumeDataset(Dataset):
    def __init__(self):
        base = torch.linspace(-1.0, 1.0, 64).reshape(1, 4, 4, 4)
        self.items = [base, base.flip(-1)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        target = self.items[index]
        mask = torch.zeros_like(target)
        return {"target": target, "init": target, "mask": mask, "name": str(index)}


class _FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def encode(self, x):
        latent = torch.cat([x, x.square()], dim=1)
        return latent, torch.zeros_like(latent)


class _DownsampleFakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encode_calls = 0

    def encode(self, x):
        self.encode_calls += 1
        latent = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return latent, torch.zeros_like(latent)


class _ZeroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.last_input = None

    def forward(self, x, timesteps, **kwargs):
        self.last_input = x.detach().clone()
        return torch.zeros_like(x)


def _diffusion():
    return GaussianDiffusion(
        betas=np.array([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )


def _guided_model(use_structure_guidance=False, use_checkpoint=False):
    base = UNetModel(
        in_channels=3,
        model_channels=32,
        out_channels=1,
        num_res_blocks=1,
        attention_resolutions=(),
        channel_mult=(1, 1, 1, 1),
        dims=3,
        cond_channels=2,
        use_checkpoint=use_checkpoint,
    )
    torch.nn.init.normal_(base.out[-1].weight, std=0.01)
    return CondUNet(
        base,
        use_multiscale_context=True,
        use_structure_guidance=use_structure_guidance,
    )


class NumericContractTests(unittest.TestCase):
    def test_zero_guidance_preserves_base_unet_output(self):
        guided = _guided_model(use_structure_guidance=True).eval()
        plain = CondUNet(copy.deepcopy(guided.unet)).eval()
        x = torch.randn(1, 1, 16, 16, 16)
        cond = torch.randn(1, 2, 16, 16, 16)
        structure = torch.randn(1, 8, 16, 16, 16)
        timestep = torch.tensor([1])
        with torch.no_grad():
            expected = plain(x, timestep, cond=cond)
            actual = guided(x, timestep, cond=cond, structure=structure)
        self.assertTrue(torch.equal(actual, expected))
        guided(x, timestep, cond=cond, structure=structure).square().mean().backward()
        gate_gradient = sum(
            gate.grad.abs().item()
            for gate in guided.control_gates.values()
            if gate.grad is not None
        )
        self.assertGreater(gate_gradient, 0.0)

    def test_ablation_models_have_identical_parameters_and_pyramid_shapes(self):
        context = _guided_model(use_structure_guidance=False)
        structure = _guided_model(use_structure_guidance=True)
        self.assertEqual(
            sum(p.numel() for p in context.parameters()),
            sum(p.numel() for p in structure.parameters()),
        )
        guidance = torch.randn(1, 10, 16, 16, 16)
        controls, middle = context.guidance_pyramid(guidance)
        self.assertEqual(tuple(controls[1].shape[-3:]), (16, 16, 16))
        self.assertEqual(tuple(controls[2].shape[-3:]), (8, 8, 8))
        self.assertEqual(tuple(controls[4].shape[-3:]), (4, 4, 4))
        self.assertEqual(tuple(controls[8].shape[-3:]), (2, 2, 2))
        self.assertEqual(tuple(middle.shape[-3:]), (2, 2, 2))

    def test_backbone_freeze_and_optimizer_groups(self):
        model = _guided_model(use_structure_guidance=True, use_checkpoint=True)
        groups = optimizer_parameter_groups(model, backbone_lr=1e-5, guidance_lr=1e-4)
        self.assertEqual([group["name"] for group in groups], ["backbone", "guidance"])
        self.assertAlmostEqual(groups[0]["lr"], 1e-5)
        self.assertAlmostEqual(groups[1]["lr"], 1e-4)
        set_backbone_trainable(model, False)
        self.assertFalse(any(parameter.requires_grad for parameter in model.unet.parameters()))
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.guidance_pyramid.parameters())
        )
        x = torch.randn(1, 1, 16, 16, 16)
        cond = torch.randn(1, 2, 16, 16, 16)
        structure = torch.randn(1, 8, 16, 16, 16)
        model(x, torch.tensor([1]), cond=cond, structure=structure).square().mean().backward()
        self.assertFalse(any(parameter.grad is not None for parameter in model.unet.parameters()))
        self.assertTrue(any(gate.grad is not None for gate in model.control_gates.values()))
        set_backbone_trainable(model, True)
        self.assertTrue(all(parameter.requires_grad for parameter in model.unet.parameters()))

    def test_structure_tensor_flip_and_fiber_direction(self):
        coords = torch.arange(16, dtype=torch.float32)
        z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
        patch = torch.exp(-((x - 7.5).square() + (y - 7.5).square()) / 10.0)
        patch = patch.reshape(1, 1, 16, 16, 16)
        tensor, valid = known_patch_structure_tensor(patch, window=5)
        mean_tensor = (tensor * valid).mean(dim=(2, 3, 4), keepdim=True)
        direction, _ = structure_orientation_field(mean_tensor)
        self.assertGreater(direction[:, 0].abs().item(), 0.9)

        flipped, _ = known_patch_structure_tensor(patch.flip(-1), window=5)
        flipped = flipped.flip(-1)
        self.assertTrue(torch.allclose(tensor[:, :4], flipped[:, :4], atol=1e-5))
        self.assertTrue(torch.allclose(tensor[:, 4:], -flipped[:, 4:], atol=1e-5))

    def test_structure_guidance_has_no_mask_edge_or_target_leakage(self):
        init = torch.zeros(1, 1, 16, 16, 16)
        mask = torch.ones_like(init)
        for z0 in (0, 12):
            for y0 in (0, 12):
                for x0 in (0, 12):
                    init[..., z0 : z0 + 4, y0 : y0 + 4, x0 : x0 + 4] = 0.5
                    mask[..., z0 : z0 + 4, y0 : y0 + 4, x0 : x0 + 4] = 0.0
        guidance_a = encode_corner_structure_guidance(
            init, mask, num_downsamples=1, window=3, propagation_steps=2
        )
        target_a = init + mask * 0.25
        target_b = init - mask * 0.75
        guidance_b = encode_corner_structure_guidance(
            init, mask, num_downsamples=1, window=3, propagation_steps=2
        )
        self.assertFalse(torch.equal(target_a, target_b))
        self.assertTrue(torch.equal(guidance_a, guidance_b))
        self.assertLess(guidance_a[:, :7].abs().max().item(), 1e-6)
        confidence_values = set(guidance_a[:, 7].unique().tolist())
        self.assertTrue({1.0, 0.5, 0.25}.issubset(confidence_values))

    def test_ema_roundtrip_and_lr_schedule(self):
        model = torch.nn.Linear(2, 2, bias=False)
        ema = copy.deepcopy(model)
        with torch.no_grad():
            model.weight.fill_(1.0)
            ema.weight.zero_()
        update_ema_model(ema, model, rate=0.5)
        self.assertTrue(torch.equal(ema.weight, torch.full_like(ema.weight, 0.5)))
        buffer = io.BytesIO()
        torch.save(ema.state_dict(), buffer)
        buffer.seek(0)
        restored = copy.deepcopy(model)
        restored.load_state_dict(torch.load(buffer))
        self.assertTrue(torch.equal(restored.weight, ema.weight))
        self.assertAlmostEqual(scheduled_learning_rate(499, 20000, 1e-4, 2e-5, 500), 1e-4)
        self.assertAlmostEqual(scheduled_learning_rate(19999, 20000, 1e-4, 2e-5, 500), 2e-5)

    def test_best_checkpoint_prefers_orientation_then_seam(self):
        directory = os.path.abspath("outputs/test_best_checkpoint_selection")
        os.makedirs(directory, exist_ok=True)
        selection_path = os.path.join(directory, "best_checkpoint.json")
        if os.path.isfile(selection_path):
            os.remove(selection_path)
        try:
            first, updated = update_best_checkpoint_selection(
                directory,
                "ldm_ckpt_2000.pt",
                2000,
                [{"boundary_orientation_error": 20.0, "seam_orientation_alignment": 0.7}],
            )
            self.assertTrue(updated)
            self.assertEqual(first["step"], 2000)
            second, updated = update_best_checkpoint_selection(
                directory,
                "ldm_ckpt_4000.pt",
                4000,
                [{"boundary_orientation_error": 20.0, "seam_orientation_alignment": 0.8}],
            )
            self.assertTrue(updated)
            self.assertEqual(second["step"], 4000)
            selected, updated = update_best_checkpoint_selection(
                directory,
                "ldm_ckpt_6000.pt",
                6000,
                [{"boundary_orientation_error": 21.0, "seam_orientation_alignment": 0.9}],
            )
            self.assertFalse(updated)
            self.assertEqual(selected["step"], 4000)
            with open(selection_path, encoding="utf-8") as file:
                self.assertEqual(json.load(file)["checkpoint"], "ldm_ckpt_4000.pt")
        finally:
            if os.path.isfile(selection_path):
                os.remove(selection_path)
            os.rmdir(directory)

    def test_sanitize_triplet_neutralizes_unknown_voxels(self):
        target = torch.tensor([[[[-0.5, 0.75]]]])
        init = target.clone()
        init[..., 1] = -1.0
        mask = torch.tensor([[[[0.0, 1.0]]]])
        result = sanitize_triplet(init, mask, target)
        self.assertEqual(result["init"][..., 0].item(), target[..., 0].item())
        self.assertEqual(result["init"][..., 1].item(), 0.0)
        self.assertEqual(set(result["mask"].unique().tolist()), {0.0, 1.0})

    def test_uint8_cache_roundtrip_preserves_source_precision(self):
        source = torch.arange(256, dtype=torch.uint8)
        normalized = source.float() / 127.5 - 1.0
        encoded = ((normalized + 1.0) * 127.5).round().to(torch.uint8)
        self.assertTrue(torch.equal(source, encoded))

    def test_latent_mask_matches_two_layer_vae_geometry(self):
        mask = torch.zeros(1, 1, 256, 256, 256)
        mask[..., 64:192, 64:192, 64:192] = 1.0
        latent_mask = latent_mask_from_voxel(mask, num_downsamples=2)
        self.assertEqual(tuple(latent_mask.shape), (1, 1, 64, 64, 64))
        self.assertEqual(latent_mask[..., 31, 31, 31].item(), 1.0)

    def test_region_losses_are_normalized_independently(self):
        diffusion = _diffusion()
        model = _ZeroModel()
        x_start = torch.zeros(1, 1, 1, 1, 2)
        noise = torch.tensor([[[[[1.0, 2.0]]]]])
        mask = torch.tensor([[[[[0.0, 1.0]]]]])
        boundary = torch.tensor([[[[[0.0, 1.0]]]]])
        terms = diffusion.training_losses(
            model,
            x_start,
            torch.zeros(1, dtype=torch.long),
            noise=noise,
            loss_mask=mask,
            boundary_mask=boundary,
            loss_unknown_weight=4.0,
            loss_known_weight=1.0,
            loss_boundary_weight=0.2,
        )
        self.assertAlmostEqual(terms["loss_unknown"].item(), 4.0)
        self.assertAlmostEqual(terms["loss_known"].item(), 1.0)
        self.assertAlmostEqual(terms["loss_boundary"].item(), 4.0)
        self.assertAlmostEqual(terms["loss"].item(), 4.2, places=5)

    def test_region_weights_can_be_explicit_without_normalization(self):
        diffusion = _diffusion()
        model = _ZeroModel()
        x_start = torch.zeros(1, 1, 1, 1, 2)
        noise = torch.tensor([[[[[1.0, 2.0]]]]])
        mask = torch.tensor([[[[[0.0, 1.0]]]]])
        terms = diffusion.training_losses(
            model,
            x_start,
            torch.zeros(1, dtype=torch.long),
            noise=noise,
            loss_mask=mask,
            boundary_mask=mask,
            loss_unknown_weight=1.0,
            loss_known_weight=0.1,
            loss_boundary_weight=0.5,
            normalize_region_weights=False,
        )
        self.assertAlmostEqual(terms["loss"].item(), 6.1, places=5)

    def test_scratch_curriculum_stages_and_loss_ramps(self):
        args = SimpleNamespace(
            scratch_curriculum=True,
            curriculum_base_steps=10_000,
            curriculum_context_steps=10_000,
            curriculum_structure_steps=20_000,
            curriculum_context_gate_scale=0.3,
            curriculum_boundary_start=5_000,
            curriculum_boundary_ramp_steps=5_000,
            curriculum_voxel_start=10_000,
            curriculum_voxel_ramp_steps=5_000,
            control_gate_scale=0.9,
            loss_boundary_weight=0.5,
            voxel_intensity_weight=0.5,
            voxel_normal_gradient_weight=0.25,
            voxel_structure_weight=0.1,
        )
        base = scratch_curriculum_state(0, args)
        self.assertEqual(base["stage"], "base")
        self.assertEqual(base["control_gate_scale"], 0.0)
        self.assertEqual(base["loss_boundary_weight"], 0.0)

        boundary_ramp = scratch_curriculum_state(7_500, args)
        self.assertAlmostEqual(boundary_ramp["loss_boundary_weight"], 0.25)

        context = scratch_curriculum_state(15_000, args)
        self.assertEqual(context["stage"], "context")
        self.assertAlmostEqual(context["control_gate_scale"], 0.15)
        self.assertEqual(context["structure_guidance_scale"], 0.0)
        self.assertAlmostEqual(context["voxel_intensity_weight"], 0.5)

        structure = scratch_curriculum_state(30_000, args)
        self.assertEqual(structure["stage"], "structure")
        self.assertAlmostEqual(structure["control_gate_scale"], 0.6)
        self.assertAlmostEqual(structure["structure_guidance_scale"], 0.5)
        self.assertAlmostEqual(structure["voxel_structure_weight"], 0.05)

        joint = scratch_curriculum_state(40_000, args)
        self.assertEqual(joint["stage"], "joint")
        self.assertAlmostEqual(joint["control_gate_scale"], 0.9)
        self.assertAlmostEqual(joint["voxel_structure_weight"], 0.1)

    def test_training_losses_can_return_predicted_xstart(self):
        diffusion = _diffusion()
        model = _ZeroModel()
        x_start = torch.randn(1, 1, 2, 2, 2)
        terms = diffusion.training_losses(
            model,
            x_start,
            torch.zeros(1, dtype=torch.long),
            noise=torch.zeros_like(x_start),
            return_pred_xstart=True,
        )
        self.assertTrue(torch.allclose(terms["pred_xstart"], x_start, atol=1e-5))

    def test_voxel_boundary_losses_detect_a_directional_seam(self):
        coords = torch.arange(16, dtype=torch.float32)
        z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
        target = torch.sin(0.45 * x + 0.2 * z).reshape(1, 1, 16, 16, 16)
        mask = torch.zeros_like(target)
        mask[..., 8:] = 1.0

        exact = voxel_boundary_losses(target, target, mask, width=3, structure_window=3)
        for value in exact.values():
            self.assertLess(value.abs().max().item(), 1e-7)

        prediction = torch.sin(-0.45 * x + 0.2 * z).reshape(1, 1, 16, 16, 16)
        losses = voxel_boundary_losses(
            prediction, target, mask, width=3, structure_window=3
        )
        self.assertGreater(losses["voxel_intensity"].item(), 0.1)
        self.assertGreater(losses["voxel_normal_gradient"].item(), 0.01)
        self.assertGreater(losses["voxel_structure"].item(), 0.01)

    def test_sampling_injects_timestep_matched_known_latent(self):
        diffusion = _diffusion()
        model = _ZeroModel()
        known = torch.full((1, 1, 2, 2, 2), 0.5)
        known_noise = torch.full_like(known, -0.25)
        mask = torch.ones(1, 1, 2, 2, 2)
        timestep = torch.zeros(1, dtype=torch.long)
        expected = diffusion.q_sample(known, timestep, noise=known_noise)
        diffusion.p_sample_loop(
            model,
            known.shape,
            noise=torch.randn_like(known),
            known_latent=known,
            known_mask=mask,
            known_noise=known_noise,
        )
        self.assertTrue(torch.allclose(model.last_input, expected))

    def test_corner_condition_is_encoded_as_eight_independent_patches(self):
        vae = _DownsampleFakeVAE()
        init = torch.zeros(1, 1, 8, 8, 8)
        mask = torch.ones_like(init)
        value = 1.0
        for z0 in (0, 6):
            for y0 in (0, 6):
                for x0 in (0, 6):
                    init[..., z0 : z0 + 2, y0 : y0 + 2, x0 : x0 + 2] = value
                    mask[..., z0 : z0 + 2, y0 : y0 + 2, x0 : x0 + 2] = 0.0
                    value += 1.0
        condition = encode_known_corners(vae, init, mask, num_downsamples=1)
        self.assertEqual(tuple(condition.shape), (1, 1, 4, 4, 4))
        self.assertEqual(vae.encode_calls, 8)
        self.assertEqual(torch.count_nonzero(condition).item(), 8)
        self.assertEqual(
            set(condition[condition != 0].tolist()), set(float(i) for i in range(1, 9))
        )

    def test_latent_whitening_bounds_and_inverse_transform(self):
        vae = _FakeVAE()
        freeze_vae(vae)
        before = vae.weight.detach().clone()
        stats = compute_latent_stats(
            vae,
            _VolumeDataset(),
            torch.device("cpu"),
            quantile_low=0.0,
            quantile_high=1.0,
            quantile_max_vectors=1024,
        )
        self.assertFalse(vae.training)
        self.assertFalse(vae.weight.requires_grad)
        self.assertTrue(torch.equal(before, vae.weight.detach()))

        encoded = []
        dataset = _VolumeDataset()
        for index in range(len(dataset)):
            z, _ = vae.encode(dataset[index]["target"].unsqueeze(0))
            encoded.append(whiten_latent(z, stats))
        normalized = torch.cat(encoded)
        self.assertTrue(
            torch.allclose(normalized.mean((0, 2, 3, 4)), torch.zeros(2), atol=1e-6)
        )
        vectors = normalized.permute(0, 2, 3, 4, 1).reshape(-1, 2)
        covariance = vectors.T @ vectors / vectors.shape[0]
        self.assertTrue(torch.allclose(covariance, torch.eye(2), atol=1e-5))

        original, _ = vae.encode(dataset[0]["target"].unsqueeze(0))
        bounded = latent_to_bounded(original, stats)
        self.assertGreaterEqual(bounded.min().item(), -1.0)
        self.assertLessEqual(bounded.max().item(), 1.0)
        unbounded = latent_to_bounded(original, stats, clamp=False)
        restored = bounded_to_latent(unbounded, stats)
        self.assertLess((restored - original).abs().max().item(), 1e-6)

    def test_dataset_overlap_checks_names_and_coordinates(self):
        with self.assertRaises(ValueError):
            assert_datasets_disjoint(
                _NamedDataset(["z0000_y0000_x0000_c256_s64"]),
                _NamedDataset(["z0000_y0000_x0000_c256_s64"]),
            )
        with self.assertRaises(ValueError):
            assert_datasets_disjoint(
                _NamedDataset(["z0000_y0000_x0000_c256_s64"]),
                _NamedDataset(["z0100_y0000_x0000_c256_s64"]),
            )


if __name__ == "__main__":
    unittest.main()
