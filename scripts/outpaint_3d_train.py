"""
3D Outpainting training script using triplets (init/full_mask/target) for 3D volumes.

Model input: concat(x_t, init, full_mask) along channels -> in_channels=3
Target: target volume (x_start) in [-1,1], shape (B, 1, D, H, W)

This wraps UNet to accept extra cond channels via model_kwargs={"cond": cond}.
Uses 3D convolutions (dims=3) instead of 2D.
"""

# 注意先运行 set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 增加显存使用效率
'''
配置项	值	说明
volume_size	128	目标尺寸
num_channels	8	最小化
channel_mult	(1,1,1,2)	最小化
batch_size	1	必须为1
diffusion_steps	500	减少步数
learn_sigma	False	减少输出通道
use_checkpoint	True	启用梯度检查点
attention	禁用	3D下显存占用过大
'''

import argparse
import os
import sys

import numpy as np
import torch as th
from torch.utils.data import DataLoader

# 获取当前脚本的目录（script文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
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
from scripts.outpainting_3d_dataset import Outpainting3DTripletDataset


class CondUNet(th.nn.Module):
    def __init__(self, base_unet: th.nn.Module):
        super().__init__()
        self.unet = base_unet

    def forward(self, x, t, cond=None, **kwargs):
        # x: (B,1,D,H,W) noisy input; cond: (B,2,D,H,W) = concat(init, full_mask)
        if cond is None:
            raise ValueError("cond is required: concat(init, full_mask) with channels=2")
        x_in = th.cat([x, cond], dim=1)  # (B,3,D,H,W)
        return self.unet(x_in, t, **kwargs)


def create_argparser():
    defaults = dict(
        # 训练/验证数据根目录（与 prepare_outpainting_3d_data.py 输出保持一致）
        train_root="outpaint_3dnew/l2-1.485_train/size256_mask64_sourcel2-1.485",
        val_root="outpaint_3dnew/l2-1.485_val/size256_mask64_sourcel2-1.485",
        out_dir_base="samples_outpaint_3d",
        source_tag="",
        batch_size=1,  # Can use 2 for 64³
        microbatch=0,
        lr=1e-4,
        log_interval=10,
        save_interval=5000,
        eval_interval=5000,
        samples_dir="samples_outpaint_3d_train",
        resume_checkpoint="",
        schedule_sampler="uniform",
        weight_decay=1e-4,
        lr_anneal_steps=0,
        epochs=140,
        num_workers=0,
        volume_size=256,  # 3D volume size (D=H=W) - 64³
    )
    # pull model/diffusion defaults and adjust in_channels
    md = model_and_diffusion_defaults()
    md.update(dict(
        image_size=256,  # For 3D, 64³
        num_channels=32,  # Reasonable for 64³
        num_res_blocks=1,
        diffusion_steps=2000,
        use_checkpoint=True,  # Keep checkpointing for memory efficiency
        attention_resolutions="",  # Keep attention disabled for 3D
        learn_sigma=True,  # Restored to default
    ))
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
        parsed_size = str(int(args.volume_size))
        parsed_mask = "32"  # default
        parsed_source = str(args.source_tag) if args.source_tag else "unknown"

    # Dataset
    ds = Outpainting3DTripletDataset(root_dir=args.train_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Prepare a fixed small eval set (same targets every time)
    Dt = Ht = Wt = int(args.volume_size)
    import torch.nn.functional as F
    fixed_k = min(2, len(ds))  # Can use 2 for 64³
    fixed_items = [ds[i] for i in range(fixed_k)] if fixed_k > 0 else []
    if fixed_items:
        e_init_fixed = th.cat([it["init"][None, ...] for it in fixed_items], dim=0)
        e_mask_fixed = th.cat([it["mask"][None, ...] for it in fixed_items], dim=0)
        e_target_fixed = th.cat([it["target"][None, ...] for it in fixed_items], dim=0)
        # 3D volumes should already be correct size, but check anyway
        if e_target_fixed.shape[-1] != Wt or e_target_fixed.shape[-2] != Ht or e_target_fixed.shape[-3] != Dt:
            e_target_fixed = F.interpolate(e_target_fixed, size=(Dt, Ht, Wt), mode="trilinear", align_corners=False)
            e_init_fixed = F.interpolate(e_init_fixed, size=(Dt, Ht, Wt), mode="trilinear", align_corners=False)
            e_mask_fixed = F.interpolate(e_mask_fixed, size=(Dt, Ht, Wt), mode="nearest")
        e_init_fixed = e_init_fixed.to(th.device("cuda"))
        e_mask_fixed = e_mask_fixed.to(th.device("cuda"))
        e_target_fixed = e_target_fixed.to(th.device("cuda"))

    # Tag for outputs (checkpoints and eval samples), under scripts/outpaint_3d_checkpoint
    run_tag = f"mask{parsed_mask}_size{parsed_size}_source{parsed_source}"
    out_root = os.path.join(current_dir, "outpaint_3d_checkpoint", run_tag)
    tagged_samples_dir = out_root
    os.makedirs(tagged_samples_dir, exist_ok=True)
    ckpt_dir = os.path.join(tagged_samples_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model: UNet with in_channels=3 (x_t + 2 cond channels), dims=3 for 3D
    volume_size = int(args.volume_size)
    attention_ds = []
    # For 3D, attention is too expensive (O((D*H*W)^2)), so disable it
    if str(args.attention_resolutions).strip():
        for res in str(args.attention_resolutions).split(","):
            if res.strip():
                attention_ds.append(volume_size // int(res))
    # If attention_ds is empty, no attention layers will be used
    out_channels = (1 if not args.learn_sigma else 2)
    
    # For 3D, use smaller channel_mult to save memory
    if volume_size == 64:
        channel_mult = (1, 2, 2, 2)  # Reasonable for 64³
    elif volume_size == 128:
        channel_mult = (1, 1, 1, 2)  # Very conservative for 128³ to save memory
    else:
        channel_mult = (1, 2, 2, 2)
    
    base_unet = UNetModel(
        in_channels=3,
        model_channels=args.num_channels,
        out_channels=out_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=args.dropout,
        channel_mult=channel_mult,
        num_classes=(None),
        use_checkpoint=args.use_checkpoint,
        num_heads=args.num_heads,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dims=3,  # 3D convolutions
    )
    model = CondUNet(base_unet).to(dist_util.dev())
    model.train()
    
    # Clear cache before training and reset peak memory stats
    if th.cuda.is_available():
        th.cuda.empty_cache()
        th.cuda.reset_peak_memory_stats()  # Reset peak memory tracking
        print(f"GPU memory allocated: {th.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {th.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

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
            init = batch["init"].to(dist_util.dev())  # (B, 1, D, H, W)
            mask = batch["mask"].to(dist_util.dev())  # (B, 1, D, H, W)
            target = batch["target"].to(dist_util.dev())  # (B, 1, D, H, W)
            B = target.shape[0]

            # 3D volumes should already be correct size, no resize needed
            # But verify dimensions match expected volume_size
            D, H, W = target.shape[-3], target.shape[-2], target.shape[-1]
            if D != Dt or H != Ht or W != Wt:
                target = F.interpolate(target, size=(Dt, Ht, Wt), mode="trilinear", align_corners=False)
                init = F.interpolate(init, size=(Dt, Ht, Wt), mode="trilinear", align_corners=False)
                mask = F.interpolate(mask, size=(Dt, Ht, Wt), mode="nearest")

            # cond: (B,2,D,H,W)
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
            
            # Clear cache periodically to prevent memory fragmentation
            if global_step % 100 == 0 and th.cuda.is_available():
                th.cuda.empty_cache()

            if global_step % int(args.log_interval) == 0:
                if th.cuda.is_available():
                    mem_allocated = th.cuda.memory_allocated() / 1024**3
                    mem_reserved = th.cuda.memory_reserved() / 1024**3
                    logger.log(f"epoch {epoch} step {global_step} loss {loss.item():.6f} | GPU mem: {mem_allocated:.2f}GB/{mem_reserved:.2f}GB")
                else:
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
                        model.train()
                        continue

                    e_cond = th.cat([e_init, e_mask], dim=1)

                    def denoised_fn(x0, e_init=e_init, e_mask=e_mask):
                        return e_mask * x0 + (1.0 - e_mask) * e_init

                    sample = diffusion.p_sample_loop(
                        model,
                        (k, 1, Dt, Ht, Wt),
                        clip_denoised=True,
                        denoised_fn=denoised_fn,
                        model_kwargs={"cond": e_cond},
                        device=dist_util.dev(),
                        progress=False,
                    )
                    # Save complete 3D samples (all slices) for each sample
                    def to_uint8(x):
                        x = x.clamp(-1, 1)
                        x = ((x + 1) * 127.5).to(th.uint8)
                        return x
                    
                    out_dir = os.path.join(tagged_samples_dir, f"step_{global_step}")
                    os.makedirs(out_dir, exist_ok=True)
                    
                    from PIL import Image
                    for i in range(k):
                        # Convert to numpy and remove channel dimension: (D, H, W)
                        sample_vol = to_uint8(sample[i])[0].cpu().numpy()  # (D, H, W)
                        target_vol = to_uint8(e_target[i])[0].cpu().numpy()  # (D, H, W)
                        init_vol = to_uint8(e_init[i])[0].cpu().numpy()  # (D, H, W)
                        mask_vol = (e_mask[i][0].cpu().numpy() * 255.0).astype(np.uint8)  # (D, H, W)
                        
                        # Save all Z slices for each volume
                        for z_idx in range(Dt):
                            # Sample
                            sample_slice = sample_vol[z_idx]
                            sample_path = os.path.join(out_dir, f"sample_{i}_z{z_idx:03d}.png")
                            Image.fromarray(sample_slice, mode="L").save(sample_path)
                            
                            # Target
                            target_slice = target_vol[z_idx]
                            target_path = os.path.join(out_dir, f"target_{i}_z{z_idx:03d}.png")
                            Image.fromarray(target_slice, mode="L").save(target_path)
                            
                            # Init
                            init_slice = init_vol[z_idx]
                            init_path = os.path.join(out_dir, f"init_{i}_z{z_idx:03d}.png")
                            Image.fromarray(init_slice, mode="L").save(init_path)
                            
                            # Mask
                            mask_slice = mask_vol[z_idx]
                            mask_path = os.path.join(out_dir, f"mask_{i}_z{z_idx:03d}.png")
                            Image.fromarray(mask_slice, mode="L").save(mask_path)
                        
                        logger.log(f"Saved complete 3D sample {i} at step {global_step} ({Dt} slices)")
                    
                    # Automatically run evaluation and save results
                    try:
                        logger.log(f"Running automatic evaluation for step {global_step}...")
                        # Import evaluation function
                        import sys
                        eval_script_path = os.path.join(os.path.dirname(__file__), 'evaluate_3d_samples.py')
                        if os.path.exists(eval_script_path):
                            # Import the evaluation module
                            import importlib.util
                            spec = importlib.util.spec_from_file_location("evaluate_3d_samples", eval_script_path)
                            eval_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(eval_module)
                            
                            # Generate single CSV file name (without step info, all steps will append to this file)
                            eval_name = os.path.basename(os.path.normpath(args.val_root))
                            eval_output_file = f"evaluation_3doutpaint_results_{eval_name}.csv"
                            eval_output_path = os.path.join(current_dir, args.samples_dir, eval_output_file)
                            
                            # Ensure directory exists
                            os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
                            
                            # Run evaluation with step info (will append to CSV)
                            eval_results = eval_module.run_evaluation(
                                recon_folder=out_dir,
                                train_data_root=args.val_root,
                                output_file=eval_output_path,
                                step=global_step
                            )
                            logger.log(f"Evaluation completed. Results appended to: {eval_output_path}")
                        else:
                            logger.log(f"Warning: Evaluation script not found at {eval_script_path}, skipping evaluation.")
                    except Exception as e:
                        logger.log(f"Warning: Failed to run automatic evaluation: {e}")
                        import traceback
                        logger.log(traceback.format_exc())
                    
                    # Clear cache after evaluation
                    if th.cuda.is_available():
                        th.cuda.empty_cache()
                model.train()

            if int(args.save_interval) > 0 and global_step % int(args.save_interval) == 0 and global_step > 0:
                ckpt = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": global_step,
                }
                out_path = os.path.join(ckpt_dir, f"outpaint_3d_ckpt_{global_step}.pt")
                th.save(ckpt, out_path)

            global_step += 1

    logger.log("training complete")


if __name__ == "__main__":
    main()