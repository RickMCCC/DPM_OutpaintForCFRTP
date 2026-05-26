"""
Conditional outpainting sampling using trained model.
Loads a triplet (init/full_mask/target) by name, runs sampling and saves result.
"""

import argparse
import os

import numpy as np
import torch as th
from PIL import Image

from . import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_gaussian_diffusion,
)
from improved_diffusion.unet import UNetModel
from IDP_outpainting.outpainting_dataset import OutpaintingTripletDataset


class CondUNet(th.nn.Module):
    def __init__(self, base_unet: th.nn.Module):
        super().__init__()
        self.unet = base_unet

    def forward(self, x, t, cond=None, **kwargs):
        if cond is None:
            raise ValueError("cond required")
        x_in = th.cat([x, cond], dim=1)
        return self.unet(x_in, t, **kwargs)


def create_argparser():
    defaults = dict(
        data_root="E:/MyPython/improved-diffusion/IDP_outpainting/training_data",
        name="",
        image_size=256,
        num_channels=96,
        num_res_blocks=3,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        diffusion_steps=4000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        ckpt_path="",
        out_path="samples_outpaint_eval.png",
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    return parser


def to_uint8(x: th.Tensor) -> np.ndarray:
    x = x.clamp(-1, 1)
    x = ((x + 1) * 127.5).to(th.uint8)
    x = x[0, 0].cpu().numpy()
    return x


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    ds = OutpaintingTripletDataset(root_dir=args.data_root)
    if args.name:
        names = [n for n in ds.names if n == args.name]
        assert names, f"name {args.name} not found"
        ds.names = names
    init = ds[0]["init"].unsqueeze(0)
    mask = ds[0]["mask"].unsqueeze(0)
    target = ds[0]["target"].unsqueeze(0)

    # resize to image_size
    Ht = Wt = int(args.image_size)
    import torch.nn.functional as F
    init = F.interpolate(init, size=(Ht, Wt), mode="bilinear", align_corners=False)
    mask = F.interpolate(mask, size=(Ht, Wt), mode="nearest")

    image_size = int(args.image_size)
    attention_ds = []
    for res in str(args.attention_resolutions).split(','):
        attention_ds.append(image_size // int(res))
    out_channels = (1 if not args.learn_sigma else 2)
    base_unet = UNetModel(
        in_channels=3,
        model_channels=args.num_channels,
        out_channels=out_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=args.dropout,
        channel_mult=(1, 2, 3, 4) if image_size == 256 else (1, 2, 2, 2),
        num_classes=None,
        use_checkpoint=args.use_checkpoint,
        num_heads=args.num_heads,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
    )
    model = CondUNet(base_unet).to(dist_util.dev())
    model.eval()

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt = th.load(args.ckpt_path, map_location=dist_util.dev())
        model.load_state_dict(ckpt["model"], strict=False)

    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        sigma_small=False,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing=args.timestep_respacing,
    )

    init = init.to(dist_util.dev())
    mask = mask.to(dist_util.dev())
    cond = th.cat([init, mask], dim=1)

    def denoised_fn(x0):
        return mask * x0 + (1.0 - mask) * init

    sample = diffusion.p_sample_loop(
        model,
        (1, 1, Ht, Wt),
        clip_denoised=True,
        denoised_fn=denoised_fn,
        model_kwargs={"cond": cond},
        device=dist_util.dev(),
        progress=True,
    )

    img = to_uint8(sample)
    Image.fromarray(img, mode="L").save(args.out_path)
    print(f"saved to {args.out_path}")


if __name__ == "__main__":
    main()


