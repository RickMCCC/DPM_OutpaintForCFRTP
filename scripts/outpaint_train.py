"""
Minimal outpainting training script using triplets (init/full_mask/target).

Model input: concat(x_t, init, full_mask) along channels -> in_channels=3
Target: target image (x_start) in [-1,1]

This wraps UNet to accept extra cond channels via model_kwargs={"cond": cond}.
"""

import argparse
import os
import sys

import torch as th
from torch.utils.data import DataLoader


# 获取当前脚本的目录（script文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（improved-diffusion文件夹）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到Python路径
sys.path.insert(0, project_root)

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_gaussian_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.unet import UNetModel
from outpainting_dataset import OutpaintingTripletDataset


class CondUNet(th.nn.Module):
    def __init__(self, base_unet: th.nn.Module):
        super().__init__()
        self.unet = base_unet

    def forward(self, x, t, cond=None, **kwargs):
        # x: (B,1,H,W) noisy input; cond: (B,2,H,W) = concat(init, full_mask)
        if cond is None:
            raise ValueError("cond is required: concat(init, full_mask) with channels=2")
        x_in = th.cat([x, cond], dim=1)
        return self.unet(x_in, t, **kwargs)


def create_argparser():
    defaults = dict(
        train_root="E:/MyPython/improved-diffusion/samples_outpaint/size256_mask50_sourcenonwoven",
        out_dir_base="samples_outpaint",
        source_tag="",
        batch_size=4,
        microbatch=0,
        lr=1e-4,
        log_interval=10,
        save_interval=5000,
        eval_interval=5000,
        samples_dir="samples_outpaint_train",
        resume_checkpoint="",
        schedule_sampler="uniform",
        weight_decay=1e-4,
        lr_anneal_steps=0,
        epochs=100, # 这里尽量可以放多一点
        num_workers=0,
    )
    # pull model/diffusion defaults and adjust in_channels
    md = model_and_diffusion_defaults()
    md.update(dict(image_size=256, num_channels=64, num_res_blocks=1, diffusion_steps=2000))
    defaults.update(md)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    #dist_util.setup_dist()
    logger.configure()

    # Resolve train_root and parse tags from its basename: size{H}_mask{M}_source{TAG}
    if not os.path.isdir(args.train_root):
        raise FileNotFoundError(f"train_root not found: {args.train_root}")
    base_name = os.path.basename(os.path.normpath(args.train_root))
    import re
    m = re.match(r"size(\d+)_mask(\d+)_source(.+)", base_name)
    if m:
        parsed_size, parsed_mask, parsed_source = m.group(1), m.group(2), m.group(3)
    else:
        # fallback to provided args or defaults
        parsed_size = str(int(args.image_size))
        parsed_mask = str(int(args.mask_size))
        parsed_source = str(args.source_tag) if args.source_tag else "unknown"

    # Dataset
    ds = OutpaintingTripletDataset(root_dir=args.train_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Prepare a fixed small eval set (same targets every time)
    Ht = Wt = int(args.image_size)
    import torch.nn.functional as F
    fixed_k = min(5, len(ds))
    fixed_items = [ds[i] for i in range(fixed_k)] if fixed_k > 0 else []
    if fixed_items:
        e_init_fixed = th.cat([it["init"][None, ...] for it in fixed_items], dim=0)
        e_mask_fixed = th.cat([it["mask"][None, ...] for it in fixed_items], dim=0)
        e_target_fixed = th.cat([it["target"][None, ...] for it in fixed_items], dim=0)
        if e_target_fixed.shape[-1] != Wt or e_target_fixed.shape[-2] != Ht:
            e_target_fixed = F.interpolate(e_target_fixed, size=(Ht, Wt), mode="bilinear", align_corners=False)
            e_init_fixed = F.interpolate(e_init_fixed, size=(Ht, Wt), mode="bilinear", align_corners=False)
            e_mask_fixed = F.interpolate(e_mask_fixed, size=(Ht, Wt), mode="nearest")
        e_init_fixed = e_init_fixed.to(dist_util.dev())
        e_mask_fixed = e_mask_fixed.to(dist_util.dev())
        e_target_fixed = e_target_fixed.to(dist_util.dev())

    # Tag for outputs (checkpoints and eval samples), under scripts/outpaint_checkpoint
    run_tag = f"mask{parsed_mask}_size{parsed_size}_source{parsed_source}"
    out_root = os.path.join(current_dir, "outpaint_checkpoint", run_tag)
    tagged_samples_dir = out_root
    os.makedirs(tagged_samples_dir, exist_ok=True)
    ckpt_dir = os.path.join(tagged_samples_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model: UNet with in_channels=3 (x_t + 2 cond channels)
    # Build attention resolutions list
    image_size = int(args.image_size)
    attention_ds = []
    for res in str(args.attention_resolutions).split(","):
        attention_ds.append(image_size // int(res))
    out_channels = (1 if not args.learn_sigma else 2)
    base_unet = UNetModel(
        in_channels=3,
        model_channels=args.num_channels,
        out_channels=out_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=args.dropout,
        channel_mult=(1, 2, 3, 4) if image_size == 64 else ((1, 2, 3, 4) if image_size == 256 else (1, 2, 2, 2)),
        num_classes=(None),
        use_checkpoint=args.use_checkpoint,
        num_heads=args.num_heads,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
    )
    model = CondUNet(base_unet).to(dist_util.dev())
    # wrap to accept cond
    model.train()

    # Diffusion
    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        sigma_small=args.sigma_small,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing=args.timestep_respacing,
    )

    opt = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.samples_dir, exist_ok=True)

    global_step = 0
    for epoch in range(int(args.epochs)):
        for batch in dl:
            init = batch["init"].to(dist_util.dev())
            mask = batch["mask"].to(dist_util.dev())
            target = batch["target"].to(dist_util.dev())
            B = target.shape[0]

            # Resize to 256x256
            Ht, Wt = args.image_size, args.image_size
            if target.shape[-1] != Wt or target.shape[-2] != Ht:
                import torch.nn.functional as F
                target = F.interpolate(target, size=(Ht, Wt), mode="bilinear", align_corners=False)
                init = F.interpolate(init, size=(Ht, Wt), mode="bilinear", align_corners=False)
                mask = F.interpolate(mask, size=(Ht, Wt), mode="nearest")

            # cond: (B,2,H,W)
            cond = th.cat([init, mask], dim=1)

            # sample random timesteps
            t = th.randint(0, diffusion.num_timesteps, (B,), device=dist_util.dev())

            # compute loss
            losses = diffusion.training_losses(
                model,
                x_start=target,
                t=t,
                model_kwargs={"cond": cond},
            )
            loss = losses["loss"].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if global_step % int(args.log_interval) == 0:
                logger.log(f"epoch {epoch} step {global_step} loss {loss.item():.6f}")

            # periodic evaluation with sampling and visualization
            if int(args.eval_interval) > 0 and global_step % int(args.eval_interval) == 0:
                model.eval()
                with th.no_grad():
                    if fixed_items:
                        e_init = e_init_fixed
                        e_mask = e_mask_fixed
                        e_target = e_target_fixed
                        k = fixed_k
                    else:
                        # fallback: skip if no data
                        model.train()
                        continue

                    e_cond = th.cat([e_init, e_mask], dim=1)

                    def denoised_fn(x0, e_init=e_init, e_mask=e_mask):
                        return e_mask * x0 + (1.0 - e_mask) * e_init

                    sample = diffusion.p_sample_loop(
                        model,
                        (k, 1, Ht, Wt),
                        clip_denoised=True,
                        denoised_fn=denoised_fn,
                        model_kwargs={"cond": e_cond},
                        device=dist_util.dev(),
                        progress=False,
                    )
                    # save individual samples
                    def to_uint8(x):
                        x = x.clamp(-1, 1)
                        x = ((x + 1) * 127.5).to(th.uint8)
                        return x
                    out_dir = os.path.join(tagged_samples_dir, f"step_{global_step}")
                    os.makedirs(out_dir, exist_ok=True)
                    for i in range(k):
                        from PIL import Image
                        # save generated sample
                        out_png = os.path.join(out_dir, f"sample_{i}.png")
                        Image.fromarray(to_uint8(sample[i])[0].cpu().numpy(), mode="L").save(out_png)
                        # save corresponding ground-truth target for comparison
                        tgt_png = os.path.join(out_dir, f"target_{i}.png")
                        Image.fromarray(to_uint8(e_target[i])[0].cpu().numpy(), mode="L").save(tgt_png)
                model.train()

            if int(args.save_interval) > 0 and global_step % int(args.save_interval) == 0 and global_step > 0:
                ckpt = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": global_step,
                }
                out_path = os.path.join(ckpt_dir, f"outpaint_ckpt_{global_step}.pt")
                th.save(ckpt, out_path)

            global_step += 1

    logger.log("training complete")


if __name__ == "__main__":
    main()


