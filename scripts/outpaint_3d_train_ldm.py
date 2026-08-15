"""
Train a 3D latent diffusion model for CFRTP outpainting.

The model predicts the unknown region in VAE latent space conditioned on:
  - z_init: latent encodings assembled from the eight known corner cubes
  - mask:   downsampled unknown-region mask, where 1 means unknown

Compared with the earlier experimental script, this version uses safer defaults
for the small current dataset, supports YAML configs, weights the diffusion loss
toward unknown regions, saves the resolved run config, and validates on multiple
samples.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion import logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
from improved_diffusion.script_util import add_dict_to_argparser, model_and_diffusion_defaults
from improved_diffusion.unet import UNetModel
from improved_diffusion.vae import VAE
from scripts.outpainting_3d_dataset import CachedOutpainting3DDataset, Outpainting3DTripletDataset


CHECKPOINT_SCHEMA_VERSION = 9


def small_adapter_conv3d(channels: int) -> th.nn.Conv3d:
    layer = th.nn.Conv3d(channels, channels, kernel_size=1)
    th.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
    th.nn.init.zeros_(layer.bias)
    return layer


class GuidancePyramid(th.nn.Module):
    """Lightweight multi-scale condition encoder with small output adapters."""

    def __init__(self, input_channels: int, control_channels: Dict[int, int]):
        super().__init__()
        self.scales = tuple(sorted(control_channels))
        if self.scales != (1, 2, 4, 8):
            raise ValueError(f"GuidancePyramid expects scales (1,2,4,8), got {self.scales}")

        feature_channels = [control_channels[scale] for scale in self.scales]
        self.stem = th.nn.Sequential(
            th.nn.Conv3d(input_channels, feature_channels[0], 3, padding=1),
            th.nn.SiLU(),
            th.nn.Conv3d(feature_channels[0], feature_channels[0], 3, padding=1),
            th.nn.SiLU(),
        )
        self.down_blocks = th.nn.ModuleList()
        for in_channels, out_channels in zip(feature_channels[:-1], feature_channels[1:]):
            self.down_blocks.append(
                th.nn.Sequential(
                    th.nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1),
                    th.nn.SiLU(),
                    th.nn.Conv3d(out_channels, out_channels, 3, padding=1),
                    th.nn.SiLU(),
                )
            )
        self.output_adapters = th.nn.ModuleDict(
            {
                str(scale): small_adapter_conv3d(channels)
                for scale, channels in control_channels.items()
            }
        )
        middle_channels = feature_channels[-1]
        self.middle = th.nn.Sequential(
            th.nn.Conv3d(middle_channels, middle_channels, 3, padding=1),
            th.nn.SiLU(),
        )
        self.middle_adapter = small_adapter_conv3d(middle_channels)

    def forward(self, guidance: th.Tensor):
        features = self.stem(guidance)
        controls = {1: self.output_adapters["1"](features)}
        for scale, block in zip(self.scales[1:], self.down_blocks):
            features = block(features)
            controls[scale] = self.output_adapters[str(scale)](features)
        return controls, self.middle_adapter(self.middle(features))


class CondUNet(th.nn.Module):
    """Wraps UNet to inject condition in two ways:
    1) channel-concatenation with x_t  →  local spatial features
    2) cross-attention via `cond=`    →  global spatial relationships
    """

    def __init__(
        self,
        base_unet: th.nn.Module,
        use_multiscale_context: bool = False,
        use_structure_guidance: bool = False,
        structure_channels: int = 8,
        control_gate_scale: float = 0.1,
    ):
        super().__init__()
        self.unet = base_unet
        self.use_multiscale_context = bool(use_multiscale_context)
        self.use_structure_guidance = bool(use_structure_guidance)
        self.structure_channels = int(structure_channels)
        self.control_gate_scale = float(control_gate_scale)
        self.structure_guidance_scale = 1.0
        self.guidance_pyramid = (
            GuidancePyramid(
                int(base_unet.cond_channels) + self.structure_channels,
                base_unet.control_channels,
            )
            if self.use_multiscale_context
            else None
        )
        self.control_gates = (
            th.nn.ParameterDict(
                {
                    **{
                        str(scale): th.nn.Parameter(th.zeros(()))
                        for scale in base_unet.control_channels
                    },
                    "middle": th.nn.Parameter(th.zeros(())),
                }
            )
            if self.use_multiscale_context
            else None
        )

    def forward(self, x, t, cond=None, structure=None, **kwargs):
        if cond is None:
            raise ValueError("cond is required: concat(z_init, latent_mask)")
        controls = None
        control_middle = None
        if self.guidance_pyramid is not None and self.control_gate_scale > 0.0:
            if structure is None or not self.use_structure_guidance:
                structure = th.zeros(
                    cond.shape[0],
                    self.structure_channels,
                    *cond.shape[-3:],
                    device=cond.device,
                    dtype=cond.dtype,
                )
            if structure.shape != (
                cond.shape[0],
                self.structure_channels,
                *cond.shape[-3:],
            ):
                raise ValueError(
                    f"structure shape {tuple(structure.shape)} is incompatible with "
                    f"condition {tuple(cond.shape)}"
                )
            structure = structure * float(self.structure_guidance_scale)
            controls, control_middle = self.guidance_pyramid(
                th.cat([cond, structure], dim=1)
            )
            controls = {
                scale: self.control_gate_scale
                * th.tanh(self.control_gates[str(scale)])
                * residual
                for scale, residual in controls.items()
            }
            control_middle = (
                self.control_gate_scale
                * th.tanh(self.control_gates["middle"])
                * control_middle
            )
        x_in = th.cat([x, cond], dim=1)
        return self.unet(
            x_in,
            t,
            cond=cond,
            control=controls,
            control_middle=control_middle,
            **kwargs,
        )


def create_argparser():
    defaults = dict(
        config="",
        data_dir="data/processed/l2-1.485_train/size256_mask64_sourcel2-1.485",
        train_cache_dir="",
        val_cache_dir="",
        val_root="data/processed/l2-1.485_val/size256_mask64_sourcel2-1.485",
        output_dir="outputs/checkpoints/outpaint_3d",
        schedule_sampler="uniform",
        lr=1e-4,
        min_lr=2e-5,
        lr_warmup_steps=500,
        training_steps=20_000,
        training_seed=20260714,
        ema_rate=0.9999,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        log_interval=10,
        save_interval=2000,
        resume_checkpoint="",
        backbone_checkpoint="",
        freeze_backbone_steps=2000,
        backbone_lr=1e-5,
        vae_checkpoint="",
        latent_stats_max_samples=0,
        latent_quantile_low=0.001,
        latent_quantile_high=0.999,
        latent_quantile_max_vectors=1_000_000,
        latent_channels=8,
        vae_hidden_channels=32,
        vae_num_res_blocks=2,
        vae_num_downsamples=3,
        num_channels=32,
        num_res_blocks=2,
        use_multiscale_context=True,
        use_structure_guidance=True,
        structure_window=5,
        structure_propagation_latent=2,
        control_gate_scale=0.1,
        scratch_curriculum=False,
        curriculum_base_steps=10_000,
        curriculum_context_steps=10_000,
        curriculum_structure_steps=20_000,
        curriculum_context_gate_scale=0.3,
        curriculum_boundary_start=5_000,
        curriculum_boundary_ramp_steps=5_000,
        curriculum_voxel_start=10_000,
        curriculum_voxel_ramp_steps=5_000,
        normalize_region_weights=True,
        smooth_voxel_timestep_weight=False,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="cosine",
        use_checkpoint=True,
        image_size=256,
        crop_size=256,
        eval_crop_size=256,
        eval_num_samples=2,
        loss_known_weight=1.0,
        loss_unknown_weight=4.0,
        loss_boundary_weight=0.2,
        voxel_boundary_width=12,
        voxel_boundary_max_timestep=250,
        voxel_intensity_weight=0.5,
        voxel_normal_gradient_weight=0.25,
        voxel_structure_weight=0.1,
        structure_tensor_window=5,
        augment_flip=True,
        allow_spatial_overlap=False,
        validation_seed=20260713,
        validation_use_ema=True,
        remote_output_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    # Re-apply intentional LDM defaults after the upstream defaults are merged.
    defaults.update(
        dict(
            image_size=256,
            num_channels=32,
            num_res_blocks=2,
            learn_sigma=True,
            diffusion_steps=1000,
            noise_schedule="cosine",
            use_checkpoint=True,
            attention_resolutions="",
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
        )
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_yaml_config(path: str) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --config. Install requirements.txt or run `pip install PyYAML`."
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def flatten_config(config: Dict) -> Dict:
    mapping = {
        ("data", "train_root"): "data_dir",
        ("data", "data_dir"): "data_dir",
        ("data", "cache_dir"): "train_cache_dir",
        ("data", "train_cache_dir"): "train_cache_dir",
        ("data", "val_cache_dir"): "val_cache_dir",
        ("data", "val_root"): "val_root",
        ("output", "dir"): "output_dir",
        ("output", "output_dir"): "output_dir",
        ("model", "volume_size"): "image_size",
        ("diffusion", "steps"): "diffusion_steps",
    }
    flat = {}
    for section, values in config.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            flat[mapping.get((section, key), key)] = value
    return flat


def apply_config(args, parser):
    if not args.config:
        return args
    config_values = flatten_config(load_yaml_config(args.config))
    cli_keys = {
        token[2:].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    for key, value in config_values.items():
        if not hasattr(args, key) or key in cli_keys or value is None:
            continue
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)
    return args


def center_crop_volume(tensor: th.Tensor, crop_size: int) -> th.Tensor:
    if crop_size <= 0 or tensor.shape[-1] <= crop_size:
        return tensor
    d0 = (tensor.shape[-3] - crop_size) // 2
    h0 = (tensor.shape[-2] - crop_size) // 2
    w0 = (tensor.shape[-1] - crop_size) // 2
    return tensor[..., d0 : d0 + crop_size, h0 : h0 + crop_size, w0 : w0 + crop_size]


def resolve_output_dir(args) -> str:
    basename = os.path.basename(os.path.normpath(args.data_dir))
    if basename.startswith("size") and "_mask" in basename:
        parts = basename.split("_")
        if len(parts) >= 3:
            basename = f"{parts[1]}_{parts[0]}_{parts[2]}"
    run_tag = f"{basename}_ds{args.vae_num_downsamples}_ch{args.latent_channels}_res{args.vae_num_res_blocks}"
    output_path = Path(args.output_dir)
    if output_path.name == "ckpt":
        return str(output_path)
    return str(output_path / run_tag / "ckpt")


def save_resolved_config(args) -> str:
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "config_used.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, sort_keys=True)
    return path


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_dataset(data_dir: str, cache_dir: str | None):
    if cache_dir and os.path.isdir(cache_dir):
        print(f"Using cached dataset: {cache_dir}")
        return CachedOutpainting3DDataset(cache_dir=cache_dir)
    print(f"Using PNG dataset: {data_dir}")
    return Outpainting3DTripletDataset(root_dir=data_dir)


def _parse_volume_box(name: str):
    match = re.match(r"z(\d+)_y(\d+)_x(\d+)_c(\d+)_", name)
    if match is None:
        return None
    z, y, x, size = (int(value) for value in match.groups())
    return (z, y, x, z + size, y + size, x + size)


def assert_datasets_disjoint(train_dataset, val_dataset, check_spatial_overlap: bool = True):
    train_names = set(train_dataset.names)
    val_names = set(val_dataset.names)
    duplicate_names = sorted(train_names & val_names)
    if duplicate_names:
        raise ValueError(
            "Training and validation datasets overlap by sample name: "
            + ", ".join(duplicate_names[:5])
        )
    if not check_spatial_overlap:
        return
    train_boxes = [(name, _parse_volume_box(name)) for name in train_names]
    val_boxes = [(name, _parse_volume_box(name)) for name in val_names]
    for train_name, train_box in train_boxes:
        if train_box is None:
            continue
        for val_name, val_box in val_boxes:
            if val_box is None:
                continue
            overlap = not (
                train_box[3] <= val_box[0]
                or train_box[0] >= val_box[3]
                or train_box[4] <= val_box[1]
                or train_box[1] >= val_box[4]
                or train_box[5] <= val_box[2]
                or train_box[2] >= val_box[5]
            )
            if overlap:
                raise ValueError(
                    "Training and validation volumes overlap spatially: "
                    f"{train_name} vs {val_name}. Split source coordinates before caching."
                )


def freeze_vae(vae: th.nn.Module):
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)


def load_pretrained_backbone(model, checkpoint_path: str, vae_checkpoint_hash: str):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise ValueError(f"Backbone checkpoint not found: {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 4:
        raise ValueError("Warm-start backbone must be a schema-v4 LDM checkpoint")
    if checkpoint.get("vae_checkpoint_hash") != vae_checkpoint_hash:
        raise ValueError("Warm-start backbone uses a different VAE checkpoint")
    source_state = checkpoint.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("Warm-start checkpoint does not contain a model state_dict")
    prefix = "unet."
    backbone_state = {
        key[len(prefix) :]: value
        for key, value in source_state.items()
        if key.startswith(prefix)
    }
    if len(backbone_state) != len(source_state):
        raise ValueError("Schema-v4 checkpoint contains unexpected non-backbone tensors")
    model.unet.load_state_dict(backbone_state, strict=True)
    return checkpoint, file_sha256(checkpoint_path)


def set_backbone_trainable(model, trainable: bool):
    for parameter in model.unet.parameters():
        parameter.requires_grad_(trainable)


def optimizer_parameter_groups(model, backbone_lr: float, guidance_lr: float):
    backbone_parameters = list(model.unet.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    guidance_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids
    ]
    if not guidance_parameters:
        raise ValueError("Warm-start training requires trainable guidance parameters")
    return [
        {"params": backbone_parameters, "lr": backbone_lr, "name": "backbone"},
        {"params": guidance_parameters, "lr": guidance_lr, "name": "guidance"},
    ]


def max_filter3d(tensor: th.Tensor, radius: int) -> th.Tensor:
    kernel_size = radius * 2 + 1
    filtered = F.max_pool3d(
        tensor, kernel_size=(kernel_size, 1, 1), stride=1, padding=(radius, 0, 0)
    )
    filtered = F.max_pool3d(
        filtered, kernel_size=(1, kernel_size, 1), stride=1, padding=(0, radius, 0)
    )
    return F.max_pool3d(
        filtered, kernel_size=(1, 1, kernel_size), stride=1, padding=(0, 0, radius)
    )


def latent_mask_from_voxel(mask_vox: th.Tensor, num_downsamples: int) -> th.Tensor:
    stride = 2 ** int(num_downsamples)
    receptive_field = 2 ** (int(num_downsamples) + 1) - 1
    dilated = max_filter3d(mask_vox, receptive_field // 2)
    return dilated[..., ::stride, ::stride, ::stride].clamp_(0.0, 1.0)


def latent_boundary_mask(mask_latent: th.Tensor, width: int = 2) -> th.Tensor:
    known_nearby = max_filter3d(1.0 - mask_latent, width)
    return (mask_latent * known_nearby).clamp_(0.0, 1.0)


def voxel_unknown_boundary_band(mask_vox: th.Tensor, width: int) -> th.Tensor:
    """Return the generated-side band within `width` voxels of known data."""
    known_nearby = max_filter3d(1.0 - mask_vox, int(width))
    return (mask_vox * known_nearby).clamp(0.0, 1.0)


def spatial_gradients(volume: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    padded = F.pad(volume, (1, 1, 1, 1, 1, 1), mode="replicate")
    grad_z = 0.5 * (
        padded[:, :, 2:, 1:-1, 1:-1] - padded[:, :, :-2, 1:-1, 1:-1]
    )
    grad_y = 0.5 * (
        padded[:, :, 1:-1, 2:, 1:-1] - padded[:, :, 1:-1, :-2, 1:-1]
    )
    grad_x = 0.5 * (
        padded[:, :, 1:-1, 1:-1, 2:] - padded[:, :, 1:-1, 1:-1, :-2]
    )
    return grad_z, grad_y, grad_x


def masked_mean_per_sample(value: th.Tensor, mask: th.Tensor) -> th.Tensor:
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], *mask.shape[2:])
    if mask.shape != value.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is incompatible with {tuple(value.shape)}"
        )
    reduce_dims = tuple(range(1, value.ndim))
    return (value * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(1.0)


def normalized_structure_tensor(volume: th.Tensor, window: int) -> th.Tensor:
    if window <= 0 or window % 2 == 0:
        raise ValueError("structure_tensor_window must be a positive odd integer")
    grad_z, grad_y, grad_x = spatial_gradients(volume)
    components = th.cat(
        [
            grad_z.square(),
            grad_y.square(),
            grad_x.square(),
            grad_z * grad_y,
            grad_z * grad_x,
            grad_y * grad_x,
        ],
        dim=1,
    )
    components = F.avg_pool3d(
        components, kernel_size=window, stride=1, padding=window // 2
    )
    trace = components[:, 0:1] + components[:, 1:2] + components[:, 2:3]
    return components / trace.clamp_min(1e-6)


def known_patch_structure_tensor(volume: th.Tensor, window: int):
    """Compute a tensor field without treating patch padding as image content."""
    if window <= 0 or window % 2 == 0:
        raise ValueError("structure_window must be a positive odd integer")
    grad_z, grad_y, grad_x = spatial_gradients(volume)
    gradient_valid = th.ones_like(volume)
    gradient_valid[..., 0, :, :] = 0.0
    gradient_valid[..., -1, :, :] = 0.0
    gradient_valid[..., :, 0, :] = 0.0
    gradient_valid[..., :, -1, :] = 0.0
    gradient_valid[..., :, :, 0] = 0.0
    gradient_valid[..., :, :, -1] = 0.0
    components = th.cat(
        [
            grad_z.square(),
            grad_y.square(),
            grad_x.square(),
            grad_z * grad_y,
            grad_z * grad_x,
            grad_y * grad_x,
        ],
        dim=1,
    )
    kernel_volume = float(window ** 3)
    valid_count = F.avg_pool3d(
        gradient_valid, window, stride=1, padding=window // 2
    ) * kernel_volume
    component_sum = F.avg_pool3d(
        components * gradient_valid,
        window,
        stride=1,
        padding=window // 2,
    ) * kernel_volume
    components = component_sum / valid_count.clamp_min(1.0)
    trace = components[:, 0:1] + components[:, 1:2] + components[:, 2:3]
    components = components / trace.clamp_min(1e-6)
    return components, (valid_count > 0.0).to(volume.dtype)


def structure_coherence(components: th.Tensor) -> th.Tensor:
    if components.shape[1] != 6:
        raise ValueError("structure tensor must have six independent components")
    b, _, d, h, w = components.shape
    matrices = th.zeros(
        b, d, h, w, 3, 3, device=components.device, dtype=components.dtype
    )
    matrices[..., 0, 0] = components[:, 0]
    matrices[..., 1, 1] = components[:, 1]
    matrices[..., 2, 2] = components[:, 2]
    matrices[..., 0, 1] = matrices[..., 1, 0] = components[:, 3]
    matrices[..., 0, 2] = matrices[..., 2, 0] = components[:, 4]
    matrices[..., 1, 2] = matrices[..., 2, 1] = components[:, 5]
    eigenvalues = th.linalg.eigvalsh(matrices.float())
    coherence = (eigenvalues[..., 2] - eigenvalues[..., 0]) / eigenvalues[
        ..., 2
    ].clamp_min(1e-6)
    return coherence.clamp(0.0, 1.0).to(components.dtype).unsqueeze(1)


def masked_pool3d(features: th.Tensor, validity: th.Tensor, stride: int):
    numerator = F.avg_pool3d(
        features * validity, kernel_size=stride, stride=stride
    )
    denominator = F.avg_pool3d(validity, kernel_size=stride, stride=stride)
    return numerator / denominator.clamp_min(1e-6), (denominator > 0.0).to(features.dtype)


def propagate_structure_guidance(
    features: th.Tensor, validity: th.Tensor, steps: int
) -> th.Tensor:
    propagated = features
    occupied = (validity > 0.0).to(features.dtype)
    confidence = occupied.clone()
    for distance in range(1, int(steps) + 1):
        neighbor_count = F.avg_pool3d(occupied, 3, stride=1, padding=1) * 27.0
        neighbor_sum = F.avg_pool3d(
            propagated * occupied, 3, stride=1, padding=1
        ) * 27.0
        frontier = (occupied < 0.5) & (neighbor_count > 0.0)
        candidate = neighbor_sum / neighbor_count.clamp_min(1.0)
        propagated = th.where(frontier.expand_as(propagated), candidate, propagated)
        confidence = th.where(
            frontier,
            th.full_like(confidence, 0.5 ** distance),
            confidence,
        )
        occupied = th.maximum(occupied, frontier.to(occupied.dtype))
    return th.cat([propagated, confidence], dim=1)


@th.no_grad()
def encode_corner_structure_guidance(
    init_vox: th.Tensor,
    mask_vox: th.Tensor,
    num_downsamples: int,
    window: int = 5,
    propagation_steps: int = 2,
) -> th.Tensor:
    """Return [tensor(6), coherence, confidence] using init corners only."""
    if init_vox.ndim != 5 or mask_vox.shape != init_vox.shape:
        raise ValueError("init and mask must have matching [B,C,D,H,W] shapes")
    stride = 2 ** int(num_downsamples)
    full_shape = tuple((int(size) + stride - 1) // stride for size in init_vox.shape[-3:])
    assembled = th.zeros(
        init_vox.shape[0], 7, *full_shape, device=init_vox.device, dtype=init_vox.dtype
    )
    assembled_validity = th.zeros(
        init_vox.shape[0], 1, *full_shape, device=init_vox.device, dtype=init_vox.dtype
    )
    known = mask_vox < 0.5

    for batch_index in range(init_vox.shape[0]):
        known_volume = known[batch_index, 0]
        intervals = (
            _edge_intervals(known_volume.any(dim=(1, 2))),
            _edge_intervals(known_volume.any(dim=(0, 2))),
            _edge_intervals(known_volume.any(dim=(0, 1))),
        )
        for z_bounds in intervals[0]:
            for y_bounds in intervals[1]:
                for x_bounds in intervals[2]:
                    z0, z1 = z_bounds
                    y0, y1 = y_bounds
                    x0, x1 = x_bounds
                    patch_mask = mask_vox[
                        batch_index : batch_index + 1, :, z0:z1, y0:y1, x0:x1
                    ]
                    if patch_mask.max().item() > 0.5:
                        raise ValueError("Known-region mask is not a complete eight-corner geometry")
                    patch = init_vox[
                        batch_index : batch_index + 1, :, z0:z1, y0:y1, x0:x1
                    ]
                    tensor, validity = known_patch_structure_tensor(patch, window)
                    tensor, pooled_validity = masked_pool3d(tensor, validity, stride)
                    coherence = structure_coherence(tensor) * pooled_validity
                    patch_features = th.cat([tensor, coherence], dim=1) * pooled_validity
                    latent_shape = patch_features.shape[-3:]
                    starts = (
                        0 if z0 == 0 else full_shape[0] - latent_shape[0],
                        0 if y0 == 0 else full_shape[1] - latent_shape[1],
                        0 if x0 == 0 else full_shape[2] - latent_shape[2],
                    )
                    slices = (
                        slice(starts[0], starts[0] + latent_shape[0]),
                        slice(starts[1], starts[1] + latent_shape[1]),
                        slice(starts[2], starts[2] + latent_shape[2]),
                    )
                    assembled[batch_index : batch_index + 1, :, *slices] = patch_features
                    assembled_validity[
                        batch_index : batch_index + 1, :, *slices
                    ] = pooled_validity
    return propagate_structure_guidance(
        assembled, assembled_validity, propagation_steps
    )


def voxel_boundary_losses(
    prediction: th.Tensor,
    target: th.Tensor,
    mask_vox: th.Tensor,
    width: int,
    structure_window: int,
) -> Dict[str, th.Tensor]:
    """Measure decoded seam quality while holding the known side exactly fixed."""
    if prediction.shape != target.shape or mask_vox.shape != target.shape:
        raise ValueError("prediction, target, and mask must have matching shapes")

    boundary = voxel_unknown_boundary_band(mask_vox, width)
    composite = mask_vox * prediction + (1.0 - mask_vox) * target
    intensity = masked_mean_per_sample((composite - target).abs(), boundary)

    pred_grad = spatial_gradients(composite)
    target_grad = spatial_gradients(target)
    smooth_mask = F.avg_pool3d(mask_vox, kernel_size=5, stride=1, padding=2)
    normal = spatial_gradients(smooth_mask)
    normal_magnitude = th.sqrt(sum(component.square() for component in normal) + 1e-8)
    normal = tuple(component / normal_magnitude for component in normal)
    pred_normal_gradient = sum(g * n for g, n in zip(pred_grad, normal))
    target_normal_gradient = sum(g * n for g, n in zip(target_grad, normal))
    normal_support = boundary * (normal_magnitude > 1e-4).to(boundary.dtype)
    normal_gradient = masked_mean_per_sample(
        (pred_normal_gradient - target_normal_gradient).abs(), normal_support
    )

    pred_structure = normalized_structure_tensor(composite, structure_window)
    with th.no_grad():
        target_structure = normalized_structure_tensor(target, structure_window)
    structure = masked_mean_per_sample(
        (pred_structure - target_structure).abs(), boundary
    )
    return {
        "voxel_intensity": intensity,
        "voxel_normal_gradient": normal_gradient,
        "voxel_structure": structure,
    }


def _edge_intervals(axis_has_known: th.Tensor) -> List[Tuple[int, int]]:
    values = axis_has_known.detach().bool().cpu().tolist()
    leading = 0
    while leading < len(values) and values[leading]:
        leading += 1
    trailing = 0
    while trailing < len(values) and values[len(values) - 1 - trailing]:
        trailing += 1
    if leading == 0 or trailing == 0 or leading + trailing > len(values):
        raise ValueError("Known mask must contain disjoint cubes at both ends of every axis")
    return [(0, leading), (len(values) - trailing, len(values))]


@th.no_grad()
def encode_known_corners(vae, init_vox: th.Tensor, mask_vox: th.Tensor, num_downsamples: int):
    """Encode each known corner independently and assemble a sparse full latent."""
    if init_vox.ndim != 5 or mask_vox.shape != init_vox.shape:
        raise ValueError("init and mask must have matching [B,C,D,H,W] shapes")
    if init_vox.shape[1] != 1:
        raise ValueError("Corner encoding currently expects one input channel")

    stride = 2 ** int(num_downsamples)
    full_shape = tuple((int(size) + stride - 1) // stride for size in init_vox.shape[-3:])
    assembled = None
    known = mask_vox < 0.5

    for batch_index in range(init_vox.shape[0]):
        known_volume = known[batch_index, 0]
        intervals = (
            _edge_intervals(known_volume.any(dim=(1, 2))),
            _edge_intervals(known_volume.any(dim=(0, 2))),
            _edge_intervals(known_volume.any(dim=(0, 1))),
        )
        for z_bounds in intervals[0]:
            for y_bounds in intervals[1]:
                for x_bounds in intervals[2]:
                    z0, z1 = z_bounds
                    y0, y1 = y_bounds
                    x0, x1 = x_bounds
                    corner_mask = mask_vox[
                        batch_index : batch_index + 1, :, z0:z1, y0:y1, x0:x1
                    ]
                    if corner_mask.max().item() > 0.5:
                        raise ValueError("Known-region mask is not a complete eight-corner geometry")
                    corner = init_vox[
                        batch_index : batch_index + 1, :, z0:z1, y0:y1, x0:x1
                    ]
                    corner_latent, _ = vae.encode(corner)
                    if assembled is None:
                        assembled = th.zeros(
                            init_vox.shape[0],
                            corner_latent.shape[1],
                            *full_shape,
                            device=corner_latent.device,
                            dtype=corner_latent.dtype,
                        )
                    latent_shape = corner_latent.shape[-3:]
                    starts = (
                        0 if z0 == 0 else full_shape[0] - latent_shape[0],
                        0 if y0 == 0 else full_shape[1] - latent_shape[1],
                        0 if x0 == 0 else full_shape[2] - latent_shape[2],
                    )
                    assembled[
                        batch_index : batch_index + 1,
                        :,
                        starts[0] : starts[0] + latent_shape[0],
                        starts[1] : starts[1] + latent_shape[1],
                        starts[2] : starts[2] + latent_shape[2],
                    ] = corner_latent

    if assembled is None:
        raise ValueError("No known corner cubes were encoded")
    return assembled


def _apply_channel_matrix(latent: th.Tensor, matrix: th.Tensor) -> th.Tensor:
    return th.einsum("ij,bjdhw->bidhw", matrix, latent)


def whiten_latent(latent: th.Tensor, stats: Dict) -> th.Tensor:
    return _apply_channel_matrix(latent - stats["mean"], stats["whitener"])


def latent_to_bounded(latent: th.Tensor, stats: Dict, clamp: bool = True) -> th.Tensor:
    bounded = (whiten_latent(latent, stats) - stats["quantile_center"]) / stats[
        "quantile_scale"
    ].clamp_min(1e-6)
    return bounded.clamp(-1.0, 1.0) if clamp else bounded


def bounded_to_latent(bounded: th.Tensor, stats: Dict) -> th.Tensor:
    whitened = bounded * stats["quantile_scale"] + stats["quantile_center"]
    return _apply_channel_matrix(whitened, stats["dewhitener"]) + stats["mean"]


@th.no_grad()
def compute_latent_stats(
    vae,
    dataset,
    device,
    max_samples: int = 0,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
    quantile_max_vectors: int = 1_000_000,
):
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("latent quantiles must satisfy 0 <= low < high <= 1")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    total = cross = None
    vector_count = 0
    sample_count = 0
    for index, batch_dict in enumerate(loader):
        if max_samples > 0 and index >= max_samples:
            break
        target = batch_dict["target"].to(device=device, dtype=th.float32)
        z_target, _ = vae.encode(target)
        vectors = z_target.permute(0, 2, 3, 4, 1).reshape(-1, z_target.shape[1]).double()
        current_total = vectors.sum(dim=0).cpu()
        current_cross = (vectors.T @ vectors).cpu()
        if total is None:
            total = th.zeros_like(current_total)
            cross = th.zeros_like(current_cross)
        total += current_total
        cross += current_cross
        vector_count += vectors.shape[0]
        sample_count += 1

    if total is None or sample_count == 0:
        raise ValueError("Cannot compute latent statistics from an empty training dataset")

    mean = total / vector_count
    covariance = cross / vector_count - th.outer(mean, mean)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = th.linalg.eigh(covariance)
    eigen_floor = max(float(eigenvalues.max().item()) * 1e-6, 1e-10)
    eigenvalues = eigenvalues.clamp_min(eigen_floor)
    whitener = (eigenvectors * eigenvalues.rsqrt().unsqueeze(0)) @ eigenvectors.T
    dewhitener = (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T

    vector_limit = int(quantile_max_vectors)
    per_item_limit = 0 if vector_limit <= 0 else max(
        1, (vector_limit + sample_count - 1) // sample_count
    )
    quantile_vectors = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    collected = 0
    for index, batch_dict in enumerate(loader):
        if index >= sample_count:
            break
        target = batch_dict["target"].to(device=device, dtype=th.float32)
        z_target, _ = vae.encode(target)
        vectors = z_target.permute(0, 2, 3, 4, 1).reshape(-1, z_target.shape[1]).cpu().double()
        take = vectors.shape[0] if per_item_limit == 0 else min(per_item_limit, vectors.shape[0])
        if vector_limit > 0:
            take = min(take, vector_limit - collected)
        if take <= 0:
            break
        if take < vectors.shape[0]:
            indices = th.linspace(0, vectors.shape[0] - 1, steps=take).round().long()
            vectors = vectors[indices]
        whitened = (vectors - mean) @ whitener.T
        quantile_vectors.append(whitened.float())
        collected += whitened.shape[0]
        if vector_limit > 0 and collected >= vector_limit:
            break

    samples = th.cat(quantile_vectors, dim=0)
    lower = th.quantile(samples, quantile_low, dim=0)
    upper = th.quantile(samples, quantile_high, dim=0)
    scale = ((upper - lower) * 0.5).clamp_min(1e-6)
    center = (upper + lower) * 0.5
    shape = (1, -1, 1, 1, 1)
    return {
        "mean": mean.float().view(*shape),
        "covariance": covariance.float(),
        "whitener": whitener.float(),
        "dewhitener": dewhitener.float(),
        "quantile_lower": lower.view(*shape),
        "quantile_upper": upper.view(*shape),
        "quantile_center": center.view(*shape),
        "quantile_scale": scale.view(*shape),
        "num_samples": sample_count,
        "num_latent_vectors": vector_count,
        "num_quantile_vectors": samples.shape[0],
        "quantile_low": float(quantile_low),
        "quantile_high": float(quantile_high),
    }


def move_latent_stats(stats: Dict, device):
    return {
        key: value.to(device=device, dtype=th.float32) if th.is_tensor(value) else value
        for key, value in stats.items()
    }


def maybe_random_flip(init_vox, target_vox, mask_vox, enabled: bool):
    if not enabled:
        return init_vox, target_vox, mask_vox
    for dim in (-3, -2, -1):
        if th.rand((), device=init_vox.device).item() < 0.5:
            init_vox = th.flip(init_vox, dims=(dim,))
            target_vox = th.flip(target_vox, dims=(dim,))
            mask_vox = th.flip(mask_vox, dims=(dim,))
    return init_vox, target_vox, mask_vox


def load_3d_outpainting_data(
    dataset,
    batch_size,
    vae,
    device,
    latent_stats,
    vae_num_downsamples,
    crop_size=128,
    deterministic=False,
    augment_flip=False,
    use_structure_guidance=False,
    structure_window=5,
    structure_propagation_latent=2,
    training_seed=20260714,
):
    loader_generator = th.Generator()
    loader_generator.manual_seed(int(training_seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=0,
        pin_memory=True,
        generator=loader_generator,
    )

    while True:
        for batch_dict in loader:
            init_vox = batch_dict["init"].to(device=device, dtype=th.float32)
            target_vox = batch_dict["target"].to(device=device, dtype=th.float32)
            mask_vox = batch_dict["mask"].to(device=device, dtype=th.float32)

            if crop_size > 0 and any(size > crop_size for size in init_vox.shape[-3:]):
                raise ValueError(
                    "Eight-corner conditioning requires the complete mask geometry; "
                    "set crop_size to the full volume size."
                )

            init_vox, target_vox, mask_vox = maybe_random_flip(
                init_vox, target_vox, mask_vox, augment_flip
            )

            with th.no_grad():
                z_target_mu, _ = vae.encode(target_vox)
                z_corner_init = encode_known_corners(
                    vae, init_vox, mask_vox, vae_num_downsamples
                )
                if use_structure_guidance:
                    structure = encode_corner_structure_guidance(
                        init_vox,
                        mask_vox,
                        vae_num_downsamples,
                        window=int(structure_window),
                        propagation_steps=int(structure_propagation_latent),
                    )
                else:
                    structure = th.zeros(
                        init_vox.shape[0],
                        8,
                        *z_corner_init.shape[-3:],
                        device=device,
                        dtype=th.float32,
                    )

            mask_latent = latent_mask_from_voxel(mask_vox, vae_num_downsamples)
            if mask_latent.shape[-3:] != z_target_mu.shape[-3:]:
                raise ValueError(
                    f"latent mask shape {tuple(mask_latent.shape)} does not match "
                    f"latent shape {tuple(z_target_mu.shape)}"
                )
            boundary_mask = latent_boundary_mask(mask_latent)
            batch = latent_to_bounded(z_target_mu, latent_stats)
            z_init_cond = latent_to_bounded(z_corner_init, latent_stats)
            z_init_cond = z_init_cond * (1.0 - mask_latent)
            cond = th.cat([z_init_cond, mask_latent], dim=1)
            yield batch, {
                "cond": cond,
                "structure": structure,
                "loss_mask": mask_latent,
                "boundary_mask": boundary_mask,
                "target_vox": target_vox,
                "mask_vox": mask_vox,
            }


def create_diffusion(args):
    betas = gd.get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    if args.use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif args.rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    timestep_respacing = args.timestep_respacing or [args.diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(args.diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON if not args.predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not args.sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not args.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=args.rescale_timesteps,
    )


def attention_downsample_rates(args) -> Tuple[int, ...]:
    if not str(args.attention_resolutions).strip():
        return tuple()
    return tuple(int(args.image_size) // int(res) for res in str(args.attention_resolutions).split(",") if res.strip())


def volume_diagnostics(volume: th.Tensor):
    values = volume.detach().float().cpu()
    return {
        "min": values.min().item(),
        "max": values.max().item(),
        "mean": values.mean().item(),
        "std": values.std().item(),
        "q01": th.quantile(values, 0.01).item(),
        "q99": th.quantile(values, 0.99).item(),
        "clip_fraction": ((values < -1.0) | (values > 1.0)).float().mean().item(),
    }


def save_volume_pngs(volume: th.Tensor, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    raw = volume[0, 0].detach().float().cpu().numpy()
    display = ((np.clip(raw, -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)
    for d_idx in range(display.shape[0]):
        Image.fromarray(display[d_idx]).save(
            os.path.join(output_dir, f"slice_{d_idx:03d}.png")
        )
    orthogonal_dir = os.path.join(output_dir, "orthogonal")
    os.makedirs(orthogonal_dir, exist_ok=True)
    for axis, axis_name in enumerate(("z", "y", "x")):
        axis_size = display.shape[axis]
        for index in np.linspace(0, axis_size - 1, num=10, dtype=int):
            plane = np.take(display, index, axis=axis)
            Image.fromarray(plane).save(
                os.path.join(orthogonal_dir, f"{axis_name}_{index:03d}.png")
            )


def validation_metrics(pred, target, init, mask, boundary_width: int = 8):
    unknown = mask
    known = 1.0 - mask
    boundary = mask * max_filter3d(known, boundary_width)

    def masked_mse(a, b, region):
        return (((a - b).square() * region).sum() / region.sum().clamp_min(1.0)).item()

    metrics = {
        "unknown_mse": masked_mse(pred, target, unknown),
        "known_preservation_mse": masked_mse(pred, init, known),
        "boundary_mse": masked_mse(pred, target, boundary),
        "z_flicker": (pred[:, :, 1:] - pred[:, :, :-1]).abs().mean().item(),
        "target_z_flicker": (target[:, :, 1:] - target[:, :, :-1]).abs().mean().item(),
    }
    pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
    target_grad = target[:, :, 1:] - target[:, :, :-1]
    boundary_grad = boundary[:, :, 1:] * boundary[:, :, :-1]
    metrics["boundary_z_gradient_mse"] = masked_mse(
        pred_grad, target_grad, boundary_grad
    )
    try:
        from skimage.metrics import structural_similarity

        pred_np = pred[0, 0].detach().cpu().numpy()
        target_np = target[0, 0].detach().cpu().numpy()
        indices = np.linspace(0, pred_np.shape[0] - 1, num=10, dtype=int)
        metrics["slice_ssim"] = float(
            np.mean(
                [
                    structural_similarity(
                        target_np[index], pred_np[index], data_range=2.0
                    )
                    for index in indices
                ]
            )
        )
    except ImportError:
        metrics["slice_ssim"] = None
    return metrics


def structure_orientation_field(components: th.Tensor):
    b, _, d, h, w = components.shape
    matrices = th.zeros(
        b, d, h, w, 3, 3, device=components.device, dtype=th.float32
    )
    values = components.float()
    matrices[..., 0, 0] = values[:, 0]
    matrices[..., 1, 1] = values[:, 1]
    matrices[..., 2, 2] = values[:, 2]
    matrices[..., 0, 1] = matrices[..., 1, 0] = values[:, 3]
    matrices[..., 0, 2] = matrices[..., 2, 0] = values[:, 4]
    matrices[..., 1, 2] = matrices[..., 2, 1] = values[:, 5]
    flat_matrices = matrices.reshape(-1, 3, 3)
    if not th.isfinite(flat_matrices).all():
        raise FloatingPointError("Non-finite structure tensor in orientation metrics")

    # cuSOLVER rejects very large batches of tiny matrices on some CUDA builds.
    eigenvalue_chunks = []
    direction_chunks = []
    for matrix_chunk in flat_matrices.split(32_768):
        eigenvalues, eigenvectors = th.linalg.eigh(matrix_chunk)
        eigenvalue_chunks.append(eigenvalues)
        direction_chunks.append(eigenvectors[..., :, 0])
    eigenvalues = th.cat(eigenvalue_chunks, dim=0).reshape(b, d, h, w, 3)
    direction = (
        th.cat(direction_chunks, dim=0)
        .reshape(b, d, h, w, 3)
        .permute(0, 4, 1, 2, 3)
    )
    coherence = (
        (eigenvalues[..., 2] - eigenvalues[..., 0])
        / eigenvalues[..., 2].clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    return direction, coherence.unsqueeze(1)


def masked_local_average(features: th.Tensor, mask: th.Tensor, window: int):
    kernel_volume = float(window ** 3)
    count = F.avg_pool3d(mask, window, stride=1, padding=window // 2) * kernel_volume
    summed = F.avg_pool3d(
        features * mask, window, stride=1, padding=window // 2
    ) * kernel_volume
    return summed / count.clamp_min(1.0), count


@th.no_grad()
def orientation_validation_metrics(pred, target, mask):
    pool = 4
    # Batched 3x3 eigendecomposition is unreliable for 64^3 fields in cuSOLVER.
    # These checkpoint-only metrics are small enough to evaluate robustly on CPU.
    pred_low = F.avg_pool3d(pred, pool, stride=pool).cpu()
    target_low = F.avg_pool3d(target, pool, stride=pool).cpu()
    mask_low = (
        F.avg_pool3d(mask, pool, stride=pool).cpu() >= 0.5
    ).to(pred_low.dtype)
    pred_tensor = normalized_structure_tensor(pred_low, window=3)
    target_tensor = normalized_structure_tensor(target_low, window=3)
    pred_direction, pred_coherence = structure_orientation_field(pred_tensor)
    target_direction, target_coherence = structure_orientation_field(target_tensor)

    boundary = voxel_unknown_boundary_band(mask_low, width=3)
    reliable = boundary * (target_coherence >= 0.1).to(boundary.dtype)
    direction_dot = (pred_direction * target_direction).sum(dim=1, keepdim=True).abs()
    orientation_error = th.rad2deg(
        th.acos(direction_dot.clamp(0.0, 1.0))
    )
    reliable_count = reliable.sum().clamp_min(1.0)

    known = 1.0 - mask_low
    known_tensor, known_count = masked_local_average(pred_tensor, known, window=7)
    unknown_tensor, unknown_count = masked_local_average(pred_tensor, mask_low, window=7)
    known_direction, known_coherence = structure_orientation_field(known_tensor)
    unknown_direction, unknown_coherence = structure_orientation_field(unknown_tensor)
    seam_valid = (
        (known_count > 0.0)
        & (unknown_count > 0.0)
        & (known_coherence >= 0.1)
        & (unknown_coherence >= 0.1)
    ).to(pred.dtype)
    seam_dot = (known_direction * unknown_direction).sum(dim=1, keepdim=True).abs()
    seam_count = seam_valid.sum().clamp_min(1.0)
    return {
        "seam_orientation_alignment": (
            (seam_dot * seam_valid).sum() / seam_count
        ).item(),
        "boundary_orientation_error": (
            (orientation_error * reliable).sum() / reliable_count
        ).item(),
        "boundary_coherence_error": (
            ((pred_coherence - target_coherence).abs() * reliable).sum()
            / reliable_count
        ).item(),
    }


def sample_validation(
    model, diffusion, vae, args, device, val_items: List[Dict], latent_stats, step: int
):
    was_training = model.training
    model.eval()
    validation_results = []
    with th.no_grad():
        for val_idx, val_batch in enumerate(val_items):
            val_init = val_batch["init"][None].to(device=device, dtype=th.float32)
            val_mask = val_batch["mask"][None].to(device=device, dtype=th.float32)
            val_target = val_batch["target"][None].to(device=device, dtype=th.float32)
            val_init = center_crop_volume(val_init, int(args.eval_crop_size))
            val_mask = center_crop_volume(val_mask, int(args.eval_crop_size))
            val_target = center_crop_volume(val_target, int(args.eval_crop_size))

            val_mask_latent = latent_mask_from_voxel(
                val_mask, args.vae_num_downsamples
            )
            z_val_corners = encode_known_corners(
                vae, val_init, val_mask, args.vae_num_downsamples
            )
            z_val_init_target_space = latent_to_bounded(z_val_corners, latent_stats)
            z_val_init_target_space = z_val_init_target_space * (1.0 - val_mask_latent)
            z_val_init_cond = z_val_init_target_space
            val_cond = th.cat([z_val_init_cond, val_mask_latent], dim=1)
            if args.use_structure_guidance:
                val_structure = encode_corner_structure_guidance(
                    val_init,
                    val_mask,
                    args.vae_num_downsamples,
                    window=int(args.structure_window),
                    propagation_steps=int(args.structure_propagation_latent),
                )
            else:
                val_structure = th.zeros(
                    1,
                    8,
                    *val_cond.shape[-3:],
                    device=device,
                    dtype=val_cond.dtype,
                )

            generator = th.Generator(device=device)
            generator.manual_seed(int(args.validation_seed) + val_idx)
            sample_shape = (1, args.latent_channels, *z_val_corners.shape[-3:])
            initial_noise = th.randn(sample_shape, device=device, generator=generator)
            known_noise = th.randn(sample_shape, device=device, generator=generator)

            logger.log(f"Sampling validation volume {val_idx} at step {step}...")
            sample = diffusion.p_sample_loop(
                model,
                sample_shape,
                noise=initial_noise,
                clip_denoised=True,
                model_kwargs={"cond": val_cond, "structure": val_structure},
                progress=True,
                known_latent=z_val_init_target_space,
                known_mask=1.0 - val_mask_latent,
                known_noise=known_noise,
            )

            decoded_latent = bounded_to_latent(sample, latent_stats)
            out_vox = vae.decode(decoded_latent)
            out_vox = val_mask * out_vox + (1.0 - val_mask) * val_init
            diagnostics = volume_diagnostics(out_vox)
            diagnostics.update(
                {
                    "bounded_latent_min": sample.min().item(),
                    "bounded_latent_max": sample.max().item(),
                    "bounded_latent_saturation_fraction": (
                        sample.abs() >= 0.999
                    ).float().mean().item(),
                }
            )
            diagnostics.update(
                validation_metrics(out_vox, val_target, val_init, val_mask)
            )
            diagnostics.update(
                orientation_validation_metrics(out_vox, val_target, val_mask)
            )
            validation_results.append(diagnostics)

            val_save_path = os.path.join(args.output_dir, f"val_sample_{val_idx}_step_{step}.pt")
            th.save(
                {
                    "init": val_init.cpu(),
                    "mask": val_mask.cpu(),
                    "target": val_target.cpu(),
                    "pred": out_vox.cpu(),
                    "diagnostics": diagnostics,
                },
                val_save_path,
            )
            diagnostics_path = os.path.join(
                args.output_dir, f"val_sample_{val_idx}_step_{step}_metrics.json"
            )
            with open(diagnostics_path, "w", encoding="utf-8") as file:
                json.dump(diagnostics, file, indent=2, sort_keys=True)
            save_volume_pngs(
                out_vox,
                os.path.join(args.output_dir, f"val_sample_{val_idx}_step_{step}_pngs"),
            )
            logger.log(f"Validation diagnostics: {json.dumps(diagnostics, sort_keys=True)}")
            logger.log(f"Saved validation sample {val_idx} to {val_save_path}")
    if was_training:
        model.train()
    return validation_results


def update_best_checkpoint_selection(output_dir, checkpoint_path, step, diagnostics):
    if not diagnostics:
        raise ValueError("Cannot select a checkpoint without validation diagnostics")
    numeric_keys = set.intersection(
        *[
            {key for key, value in item.items() if isinstance(value, (int, float))}
            for item in diagnostics
        ]
    )
    validation_mean = {
        key: float(np.mean([item[key] for item in diagnostics]))
        for key in sorted(numeric_keys)
    }
    primary = validation_mean["boundary_orientation_error"]
    tie_breaker = validation_mean["seam_orientation_alignment"]
    selection_path = os.path.join(output_dir, "best_checkpoint.json")
    previous = None
    if os.path.isfile(selection_path):
        with open(selection_path, "r", encoding="utf-8") as file:
            previous = json.load(file)
    is_better = previous is None
    if previous is not None:
        previous_mean = previous["validation_mean"]
        previous_score = (
            float(previous_mean["boundary_orientation_error"]),
            -float(previous_mean["seam_orientation_alignment"]),
        )
        is_better = (primary, -tie_breaker) < previous_score
    if is_better:
        selection = {
            "checkpoint": os.path.basename(checkpoint_path),
            "step": int(step),
            "selection_metric": "boundary_orientation_error",
            "tie_breaker": "seam_orientation_alignment",
            "validation_mean": validation_mean,
        }
        with open(selection_path, "w", encoding="utf-8") as file:
            json.dump(selection, file, indent=2, sort_keys=True)
        return selection, True
    return previous, False


def set_training_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True


@th.no_grad()
def update_ema_model(ema_model: th.nn.Module, model: th.nn.Module, rate: float):
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(rate).add_(parameter.detach(), alpha=1.0 - rate)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def scheduled_learning_rate(
    completed_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    next_step = completed_steps + 1
    if warmup_steps > 0 and next_step <= warmup_steps:
        return base_lr * next_step / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((next_step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def linear_ramp(completed_steps: int, start_step: int, ramp_steps: int) -> float:
    if completed_steps < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(max((completed_steps - start_step) / ramp_steps, 0.0), 1.0)


def scratch_curriculum_state(completed_steps: int, args) -> Dict[str, float | str]:
    if not bool(args.scratch_curriculum):
        return {
            "stage": "joint",
            "control_gate_scale": float(args.control_gate_scale),
            "structure_guidance_scale": 1.0,
            "loss_boundary_weight": float(args.loss_boundary_weight),
            "voxel_intensity_weight": float(args.voxel_intensity_weight),
            "voxel_normal_gradient_weight": float(args.voxel_normal_gradient_weight),
            "voxel_structure_weight": float(args.voxel_structure_weight),
        }

    base_end = max(0, int(args.curriculum_base_steps))
    context_steps = max(0, int(args.curriculum_context_steps))
    context_end = base_end + context_steps
    structure_steps = max(0, int(args.curriculum_structure_steps))
    structure_end = context_end + structure_steps
    target_gate = max(0.0, float(args.control_gate_scale))
    context_gate = min(
        max(0.0, float(args.curriculum_context_gate_scale)), target_gate
    )

    if completed_steps < base_end:
        stage = "base"
        gate_scale = 0.0
        structure_scale = 0.0
    elif completed_steps < context_end:
        stage = "context"
        progress = linear_ramp(completed_steps, base_end, context_steps)
        gate_scale = context_gate * progress
        structure_scale = 0.0
    elif completed_steps < structure_end:
        stage = "structure"
        progress = linear_ramp(completed_steps, context_end, structure_steps)
        gate_scale = context_gate + (target_gate - context_gate) * progress
        structure_scale = progress
    else:
        stage = "joint"
        gate_scale = target_gate
        structure_scale = 1.0

    boundary_progress = linear_ramp(
        completed_steps,
        int(args.curriculum_boundary_start),
        int(args.curriculum_boundary_ramp_steps),
    )
    voxel_progress = linear_ramp(
        completed_steps,
        int(args.curriculum_voxel_start),
        int(args.curriculum_voxel_ramp_steps),
    )
    return {
        "stage": stage,
        "control_gate_scale": gate_scale,
        "structure_guidance_scale": structure_scale,
        "loss_boundary_weight": float(args.loss_boundary_weight) * boundary_progress,
        "voxel_intensity_weight": float(args.voxel_intensity_weight) * voxel_progress,
        "voxel_normal_gradient_weight": (
            float(args.voxel_normal_gradient_weight) * voxel_progress
        ),
        "voxel_structure_weight": (
            float(args.voxel_structure_weight) * structure_scale
        ),
    }


def apply_curriculum_to_model(model: CondUNet, state: Dict[str, float | str]):
    model.control_gate_scale = float(state["control_gate_scale"])
    model.structure_guidance_scale = float(state["structure_guidance_scale"])


def save_ldm_checkpoint(
    model,
    ema_model,
    optimizer,
    step,
    args,
    vae_checkpoint_hash,
    backbone_checkpoint_hash,
    latent_stats,
):
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_stats = {
        key: value.detach().cpu() if th.is_tensor(value) else value
        for key, value in latent_stats.items()
    }
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "vae_checkpoint": os.path.abspath(args.vae_checkpoint),
        "vae_checkpoint_hash": vae_checkpoint_hash,
        "backbone_checkpoint": os.path.abspath(args.backbone_checkpoint)
        if args.backbone_checkpoint
        else "",
        "backbone_checkpoint_hash": backbone_checkpoint_hash,
        "latent_stats": checkpoint_stats,
        "guidance_schema": {
            "channels": [
                "Tzz",
                "Tyy",
                "Txx",
                "Tzy",
                "Tzx",
                "Tyx",
                "coherence",
                "confidence",
            ],
            "structure_window": int(args.structure_window),
            "propagation_latent": int(args.structure_propagation_latent),
            "use_structure_guidance": bool(args.use_structure_guidance),
        },
        "config": vars(args),
    }
    ckpt_path = os.path.join(args.output_dir, f"ldm_ckpt_{step}.pt")
    th.save(checkpoint, ckpt_path)
    logger.log(f"Saved raw+EMA checkpoint to {ckpt_path}")
    return ckpt_path


def main():
    parser = create_argparser()
    args = apply_config(parser.parse_args(), parser)
    if args.scratch_curriculum:
        if args.resume_checkpoint or args.backbone_checkpoint:
            raise ValueError(
                "Scratch curriculum cannot load resume_checkpoint or backbone_checkpoint"
            )
        if int(args.freeze_backbone_steps) != 0:
            raise ValueError("Scratch curriculum requires freeze_backbone_steps=0")
    args.output_dir = resolve_output_dir(args)
    set_training_seed(int(args.training_seed))

    logger.configure()
    logger.log(f"Saved resolved config to {save_resolved_config(args)}")
    logger.log("creating model and diffusion...")

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    vae = VAE(
        in_channels=1,
        latent_channels=args.latent_channels,
        hidden_channels=args.vae_hidden_channels,
        num_res_blocks=args.vae_num_res_blocks,
        num_downsamples=args.vae_num_downsamples,
        dims=3,
    ).to(device)

    if args.vae_checkpoint and os.path.isfile(args.vae_checkpoint):
        logger.log(f"Loading pre-trained VAE weights from {args.vae_checkpoint}...")
        vae_checkpoint_hash = file_sha256(args.vae_checkpoint)
        try:
            ckpt_data = th.load(args.vae_checkpoint, map_location=device)
            # 兼容新旧 checkpoint 格式
            vae_sd = ckpt_data["model"] if isinstance(ckpt_data, dict) and "model" in ckpt_data else ckpt_data
            vae.load_state_dict(vae_sd)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load VAE checkpoint. Check that --latent_channels, "
                "--vae_hidden_channels, --vae_num_res_blocks, and "
                "--vae_num_downsamples match the checkpoint architecture."
            ) from exc
    else:
        raise ValueError("Please provide a valid --vae_checkpoint path to train the LDM.")
    freeze_vae(vae)
    logger.log(f"Frozen VAE checkpoint SHA256: {vae_checkpoint_hash}")

    channel_mult = (1, 2, 4, 4)
    base_unet = UNetModel(
        in_channels=args.latent_channels * 2 + 1,
        model_channels=args.num_channels,
        out_channels=args.latent_channels * 2 if args.learn_sigma else args.latent_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=attention_downsample_rates(args),
        dropout=args.dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=args.use_checkpoint,
        num_heads=args.num_heads,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dims=3,
        cond_channels=args.latent_channels + 1,  # z_init_scaled + mask_latent
    )
    model = CondUNet(
        base_unet,
        use_multiscale_context=args.use_multiscale_context,
        use_structure_guidance=args.use_structure_guidance,
        control_gate_scale=args.control_gate_scale,
    )
    backbone_source_state = None
    backbone_checkpoint_hash = ""
    if args.backbone_checkpoint and not args.resume_checkpoint:
        logger.log(f"Warm-starting UNet backbone from {args.backbone_checkpoint}...")
        backbone_source_state, backbone_checkpoint_hash = load_pretrained_backbone(
            model, args.backbone_checkpoint, vae_checkpoint_hash
        )
        logger.log(
            "Loaded schema-v4 backbone at step "
            f"{int(backbone_source_state.get('step', 0))}; SHA256: "
            f"{backbone_checkpoint_hash}"
        )
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    guidance_parameter_count = (
        sum(parameter.numel() for parameter in model.guidance_pyramid.parameters())
        if model.guidance_pyramid is not None
        else 0
    )
    logger.log(
        f"LDM parameters: {parameter_count / 1e6:.2f}M "
        f"(guidance {guidance_parameter_count / 1e6:.2f}M)"
    )
    if args.use_multiscale_context and parameter_count > 22_000_000:
        raise ValueError("ControlNet-lite parameter budget exceeded 22M")
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    diffusion = create_diffusion(args)

    logger.log("creating isolated training and validation datasets...")
    train_dataset = make_dataset(args.data_dir, args.train_cache_dir)
    val_dataset = make_dataset(args.val_root, args.val_cache_dir)
    assert_datasets_disjoint(
        train_dataset, val_dataset, check_spatial_overlap=not args.allow_spatial_overlap
    )

    step = 0
    opt = th.optim.AdamW(
        optimizer_parameter_groups(model, args.backbone_lr, args.lr),
        weight_decay=args.weight_decay,
    )
    resume_state = None
    if args.resume_checkpoint:
        if not os.path.isfile(args.resume_checkpoint):
            raise ValueError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        logger.log(f"Resuming LDM from {args.resume_checkpoint}...")
        resume_state = th.load(args.resume_checkpoint, map_location=device)
        if not isinstance(resume_state, dict) or resume_state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "Legacy LDM checkpoints are incompatible with the frozen-VAE numeric contract. "
                "Start a new run instead of resuming the step-40000 model."
            )
        if resume_state.get("vae_checkpoint_hash") != vae_checkpoint_hash:
            raise ValueError("Resume checkpoint was trained with a different VAE checkpoint")
        resume_config = resume_state.get("config", {})
        for key in (
            "use_multiscale_context",
            "use_structure_guidance",
            "scratch_curriculum",
        ):
            if bool(resume_config.get(key)) != bool(getattr(args, key)):
                raise ValueError(f"Resume checkpoint has incompatible {key}")
        model.load_state_dict(resume_state["model"])
        ema_model.load_state_dict(resume_state["ema_model"])
        opt.load_state_dict(resume_state["optimizer"])
        backbone_checkpoint_hash = resume_state.get("backbone_checkpoint_hash", "")
        step = int(resume_state.get("step", 0))
        logger.log(f"Resume start step: {step}")

    if resume_state is not None:
        latent_stats = move_latent_stats(resume_state["latent_stats"], device)
        logger.log("Loaded bounded-whitened latent transform from resume checkpoint.")
    elif backbone_source_state is not None:
        latent_stats = move_latent_stats(backbone_source_state["latent_stats"], device)
        logger.log("Loaded latent transform from the schema-v4 backbone checkpoint.")
    else:
        logger.log("Computing target latent covariance and robust quantile bounds...")
        latent_stats = compute_latent_stats(
            vae,
            train_dataset,
            device,
            max_samples=int(args.latent_stats_max_samples),
            quantile_low=float(args.latent_quantile_low),
            quantile_high=float(args.latent_quantile_high),
            quantile_max_vectors=int(args.latent_quantile_max_vectors),
        )
        latent_stats = move_latent_stats(latent_stats, device)
    whitened_covariance = (
        latent_stats["whitener"]
        @ latent_stats["covariance"]
        @ latent_stats["whitener"].T
    )
    logger.log(
        "target latent mean: " + str(latent_stats["mean"].flatten().tolist())
    )
    logger.log(
        "whitened latent covariance max error: "
        f"{(whitened_covariance - th.eye(args.latent_channels, device=device)).abs().max().item():.6g}"
    )
    logger.log(
        "whitened latent quantile bounds: "
        f"{latent_stats['quantile_lower'].flatten().tolist()} / "
        f"{latent_stats['quantile_upper'].flatten().tolist()}"
    )

    eval_count = max(1, min(int(args.eval_num_samples), len(val_dataset)))
    val_items = [val_dataset[i] for i in range(eval_count)]
    data = load_3d_outpainting_data(
        dataset=train_dataset,
        batch_size=args.batch_size,
        vae=vae,
        device=device,
        latent_stats=latent_stats,
        vae_num_downsamples=args.vae_num_downsamples,
        crop_size=args.crop_size,
        augment_flip=args.augment_flip,
        use_structure_guidance=args.use_structure_guidance,
        structure_window=args.structure_window,
        structure_propagation_latent=args.structure_propagation_latent,
        training_seed=args.training_seed,
    )

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    max_steps = (
        int(args.lr_anneal_steps)
        if int(args.lr_anneal_steps) > 0
        else int(args.training_steps)
    )
    target_voxel_loss_weights = {
        "voxel_intensity": float(args.voxel_intensity_weight),
        "voxel_normal_gradient": float(args.voxel_normal_gradient_weight),
        "voxel_structure": float(args.voxel_structure_weight),
    }
    if any(weight > 0.0 for weight in target_voxel_loss_weights.values()):
        logger.log(
            "decoded voxel boundary loss target: "
            f"width={args.voxel_boundary_width}, max_t={args.voxel_boundary_max_timestep}, "
            f"weights={target_voxel_loss_weights}"
        )
    if args.scratch_curriculum:
        logger.log(
            "scratch curriculum enabled: "
            f"base={args.curriculum_base_steps}, "
            f"context={args.curriculum_context_steps}, "
            f"structure={args.curriculum_structure_steps}, "
            f"target_gate={args.control_gate_scale}"
        )

    freeze_backbone_steps = max(0, int(args.freeze_backbone_steps))
    backbone_trainable = step >= freeze_backbone_steps
    set_backbone_trainable(model, backbone_trainable)
    if backbone_trainable:
        logger.log(
            f"UNet backbone trainable at lr ratio {float(args.backbone_lr) / float(args.lr):.4f}"
        )
    else:
        logger.log(f"UNet backbone frozen through completed step {freeze_backbone_steps}")

    logger.log("training...")
    last_curriculum_stage = None
    while step < max_steps:
        should_train_backbone = step >= freeze_backbone_steps
        if should_train_backbone != backbone_trainable:
            backbone_trainable = should_train_backbone
            set_backbone_trainable(model, backbone_trainable)
            logger.log(
                f"Unfroze UNet backbone at completed step {step}; "
                f"base lr {args.backbone_lr:.8f}, guidance lr {args.lr:.8f}"
            )
        current_lr = scheduled_learning_rate(
            step,
            max_steps,
            float(args.lr),
            float(args.min_lr),
            int(args.lr_warmup_steps),
        )
        for parameter_group in opt.param_groups:
            if parameter_group.get("name") == "backbone":
                parameter_group["lr"] = (
                    current_lr * float(args.backbone_lr) / float(args.lr)
                    if backbone_trainable
                    else 0.0
                )
            else:
                parameter_group["lr"] = current_lr
        curriculum = scratch_curriculum_state(step, args)
        apply_curriculum_to_model(model, curriculum)
        apply_curriculum_to_model(ema_model, curriculum)
        if curriculum["stage"] != last_curriculum_stage:
            last_curriculum_stage = curriculum["stage"]
            logger.log(
                f"curriculum stage={last_curriculum_stage} at completed step {step}: "
                f"gate={float(curriculum['control_gate_scale']):.4f}, "
                f"structure={float(curriculum['structure_guidance_scale']):.4f}, "
                f"latent_boundary={float(curriculum['loss_boundary_weight']):.4f}"
            )
        voxel_loss_weights = {
            "voxel_intensity": float(curriculum["voxel_intensity_weight"]),
            "voxel_normal_gradient": float(
                curriculum["voxel_normal_gradient_weight"]
            ),
            "voxel_structure": float(curriculum["voxel_structure_weight"]),
        }
        use_voxel_boundary_loss = any(
            weight > 0.0 for weight in voxel_loss_weights.values()
        )
        batch_latent, cond = next(data)
        batch_latent = batch_latent.to(device=device, dtype=th.float32)
        loss_mask = cond.pop("loss_mask", None)
        boundary_mask = cond.pop("boundary_mask", None)
        target_vox = cond.pop("target_vox", None)
        mask_vox = cond.pop("mask_vox", None)
        cond = {k: v.to(device=device, dtype=th.float32) for k, v in cond.items()}
        if loss_mask is not None:
            loss_mask = loss_mask.to(device=device, dtype=th.float32)
        if boundary_mask is not None:
            boundary_mask = boundary_mask.to(device=device, dtype=th.float32)

        t, weights = schedule_sampler.sample(batch_latent.shape[0], device)
        losses = diffusion.training_losses(
            model,
            batch_latent,
            t,
            model_kwargs=cond,
            loss_mask=loss_mask,
            boundary_mask=boundary_mask,
            loss_known_weight=args.loss_known_weight,
            loss_unknown_weight=args.loss_unknown_weight,
            loss_boundary_weight=float(curriculum["loss_boundary_weight"]),
            normalize_region_weights=args.normalize_region_weights,
            return_pred_xstart=use_voxel_boundary_loss,
        )
        pred_xstart = losses.pop("pred_xstart", None)
        if use_voxel_boundary_loss:
            if pred_xstart is None or target_vox is None or mask_vox is None:
                raise RuntimeError("voxel boundary loss requires pred_xstart, target, and mask")
            selected = th.nonzero(
                t <= int(args.voxel_boundary_max_timestep), as_tuple=False
            ).flatten()
            if selected.numel() > 0:
                selected_prediction = pred_xstart.index_select(0, selected).clamp(-1.0, 1.0)
                decoded_prediction = vae.decode(
                    bounded_to_latent(selected_prediction, latent_stats)
                )
                voxel_terms = voxel_boundary_losses(
                    decoded_prediction,
                    target_vox.index_select(0, selected),
                    mask_vox.index_select(0, selected),
                    width=int(args.voxel_boundary_width),
                    structure_window=int(args.structure_tensor_window),
                )
                if args.smooth_voxel_timestep_weight:
                    max_t = max(int(args.voxel_boundary_max_timestep), 1)
                    timestep_weight = (
                        (max_t - t.index_select(0, selected)).float() / max_t
                    ).clamp(0.0, 1.0).square()
                else:
                    timestep_weight = th.ones(
                        selected.shape[0], device=device, dtype=th.float32
                    )
                for key, selected_values in voxel_terms.items():
                    selected_values = selected_values * timestep_weight
                    full_values = th.zeros_like(losses["loss"]).index_copy(
                        0, selected, selected_values
                    )
                    losses[key] = full_values
                    losses["loss"] = losses["loss"] + voxel_loss_weights[key] * full_values
        loss = (losses["loss"] * weights).mean()

        # VAE 重建损失：decoder 也参与梯度流
        if not th.isfinite(loss):
            opt.zero_grad()
            raise FloatingPointError(
                f"Non-finite loss before optimizer update at completed step {step}"
            )

        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        update_ema_model(ema_model, model, float(args.ema_rate))
        step += 1

        if step == 1 or step % args.log_interval == 0:
            log_parts = [
                f"step {step}",
                f"guidance_lr {current_lr:.8f}",
                f"backbone_lr {opt.param_groups[0]['lr']:.8f}",
                f"stage {curriculum['stage']}",
                f"gate {float(curriculum['control_gate_scale']):.4f}",
                f"structure_scale {float(curriculum['structure_guidance_scale']):.4f}",
                f"loss {loss.item():.6f}",
            ]
            for key in (
                "loss_unknown",
                "loss_known",
                "loss_boundary",
                "voxel_intensity",
                "voxel_normal_gradient",
                "voxel_structure",
            ):
                if key in losses:
                    log_parts.append(
                        f"{key} {(losses[key] * weights).mean().item():.6f}"
                    )
            logger.log(" | ".join(log_parts))

        if step % int(args.save_interval) == 0 and args.output_dir:
            validation_curriculum = scratch_curriculum_state(step, args)
            apply_curriculum_to_model(model, validation_curriculum)
            apply_curriculum_to_model(ema_model, validation_curriculum)
            ckpt_path = save_ldm_checkpoint(
                model,
                ema_model,
                opt,
                step,
                args,
                vae_checkpoint_hash,
                backbone_checkpoint_hash,
                latent_stats,
            )

            if args.remote_output_dir:
                import shutil
                try:
                    remote_dir = os.path.join(args.remote_output_dir, os.path.basename(os.path.normpath(args.output_dir)))
                    os.makedirs(remote_dir, exist_ok=True)
                    remote_path = os.path.join(remote_dir, f"ldm_ckpt_{step}.pt")
                    shutil.copy2(ckpt_path, remote_path)
                    logger.log(f"Copied checkpoint to remote: {remote_path}")
                except Exception as e:
                    logger.log(f"WARNING: Failed to copy checkpoint to remote: {e}")
            validation_model = ema_model if args.validation_use_ema else model
            validation_variant = "EMA" if args.validation_use_ema else "raw"
            logger.log(f"Validation model: {validation_variant}")
            validation_results = sample_validation(
                validation_model,
                diffusion,
                vae,
                args,
                device,
                val_items,
                latent_stats,
                step,
            )
            selection, is_best = update_best_checkpoint_selection(
                args.output_dir, ckpt_path, step, validation_results
            )
            logger.log(
                f"Best checkpoint: {selection['checkpoint']} "
                f"(boundary orientation error "
                f"{selection['validation_mean']['boundary_orientation_error']:.4f} deg)"
                + (" [updated]" if is_best else "")
            )

    if args.output_dir and step % int(args.save_interval) != 0:
        save_ldm_checkpoint(
            model,
            ema_model,
            opt,
            step,
            args,
            vae_checkpoint_hash,
            backbone_checkpoint_hash,
            latent_stats,
        )

    logger.log("training complete")


if __name__ == "__main__":
    main()
