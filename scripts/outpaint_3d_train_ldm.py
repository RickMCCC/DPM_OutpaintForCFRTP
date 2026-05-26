"""
3D outpainting training script with Latent Diffusion Models

This script extends the original outpaint_3d_train.py to use Latent Diffusion Models (LDMs),
which compress the input data into a lower-dimensional latent space before diffusion.

Usage example:
  python scripts/outpaint_3d_train_ldm.py \
    --data_dir "samples_outpaint_3d/size128_mask32_source1.576" \
    --output_dir "outpaint_3d_ldm_checkpoint" \
    --batch_size 2 \
    --image_size 128 \
    --num_channels 64 \
    --num_res_blocks 2 \
    --learn_sigma True \
    --diffusion_steps 1000 \
    --noise_schedule cosine \
    --latent_channels 8
"""

import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader

from improved_diffusion import logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.vae import VAE
from scripts.outpainting_3d_dataset import Outpainting3DTripletDataset

def create_argparser():
    defaults = dict(
        # 与 outpaint_3d_train.py 保持一致的数据根目录
        data_dir="outpaint_3dnew/l2-1.485_train/size256_mask64_sourcel2-1.485",
        val_root="outpaint_3dnew/l2-1.485_val/size256_mask64_sourcel2-1.485",
        output_dir="outpaint_3d_ldm_checkpoint",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=2000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # VAE / latent 相关参数（都可通过命令行修改）
        vae_checkpoint="",        # 必须传入训练好的 VAE 权重！
        latent_scale_factor=0.0,  # 新增：用于约束潜空间方差(0.0表示启动时根据首个Batch自动计算)
        latent_channels=8,        # 潜在空间通道数（提升以保留高频纹理）
        vae_hidden_channels=32,   # VAE 隐层通道数（提升以增加重建网络容量）
        vae_num_res_blocks=3,     # VAE 残差块数量（提升网络深度以对应压缩改变）
        vae_num_downsamples=2,    # VAE 的下采样次数 (修改为 2 次 = 4x 压缩)·
        num_channels=320, # [主宰级提升] 直接对标 Stable Diffusion 原生架构的基础通道数
        num_res_blocks=4, # 保持残差块深度，配合 320 超大通道数
        learn_sigma=False,  
        diffusion_steps=500,
        noise_schedule="cosine",
        use_checkpoint=True,  # 启用梯度检查点
        image_size=256,
        crop_size=128,        # 当 > 0 时，随机裁剪 3D 训练块
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

class CondUNet(th.nn.Module):
    def __init__(self, base_unet: th.nn.Module):
        super().__init__()
        self.unet = base_unet

    def forward(self, x, t, cond=None, **kwargs):
        # x: (B,C,D,H,W) noisy latent; cond: (B,C+1,D,H,W) = concat(z_init, latent_mask)
        if cond is None:
            raise ValueError("cond is required: concat(z_init, full_mask) with channels=C+1")
        x_in = th.cat([x, cond], dim=1)  # (B,2C+1,D,H,W) or similar
        return self.unet(x_in, t, **kwargs)

    def convert_to_fp16(self):
        # 调用底层unet的convert_to_fp16方法
        if hasattr(self.unet, 'convert_to_fp16'):
            self.unet.convert_to_fp16()


def load_3d_outpainting_data(
    data_dir,
    batch_size,
    image_size,
    vae,
    device,
    latent_channels,
    latent_scale_factor=1.0,
    crop_size=128,
    class_cond=False,
    deterministic=False,
):
    """
    Load 3D outpainting data for latent diffusion training.

    - 从 Outpainting3DTripletDataset 读取体素空间的 (init, mask, target)
    - 使用 VAE 将 init/target 编码到 latent 空间
    - 将 mask 下采样到 latent 分辨率
    - 返回:
        batch: z_target,  形状 (B, C_latent, D', H', W')
        cond:  {"cond": concat(z_init, latent_mask)}, 形状 (B, C_latent+1, D', H', W')
    """
    dataset = Outpainting3DTripletDataset(root_dir=data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=0,
        pin_memory=True,
    )

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    while True:
        for batch_dict in loader:
            # 体素空间数据: (B,1,D,H,W)，range: init/target in [-1,1], mask in [0,1]
            # 这里取消了 fp16，直接保留为 fp32
            init_vox = batch_dict["init"].to(device=device, dtype=th.float32)
            target_vox = batch_dict["target"].to(device=device, dtype=th.float32)
            mask_vox = batch_dict["mask"].to(device=device, dtype=th.float32)

            # 方案二: 3D 分块随机裁剪训练 (Patch-based Training)
            if crop_size > 0 and init_vox.shape[-1] > crop_size:
                # 随机生成 D, H, W 的起点
                d_idx = th.randint(0, init_vox.shape[-3] - crop_size + 1, (1,)).item()
                h_idx = th.randint(0, init_vox.shape[-2] - crop_size + 1, (1,)).item()
                w_idx = th.randint(0, init_vox.shape[-1] - crop_size + 1, (1,)).item()
                
                init_vox = init_vox[..., d_idx:d_idx+crop_size, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size]
                target_vox = target_vox[..., d_idx:d_idx+crop_size, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size]
                mask_vox = mask_vox[..., d_idx:d_idx+crop_size, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size]

            # 编码到 latent 空间
            with th.no_grad():
                z_target_mu, _ = vae.encode(target_vox)  # (B,C_latent,D',H',W')
                z_init_mu, _ = vae.encode(init_vox)      # (B,C_latent,D',H',W')

            # 将 mask 下采样到 latent 分辨率 (B,1,D',H',W')
            latent_spatial = z_target_mu.shape[-3:]
            mask_latent = F.interpolate(
                mask_vox, size=latent_spatial, mode="nearest"
            )

            # 释放原始 Float16 张量并清空 CUDA 缓存，极致节省显存
            del init_vox, target_vox, mask_vox
            th.cuda.empty_cache()

            # 扩散的 x_start 在 latent 空间，被缩放系数规范化，使其方差大致逼近 1.0
            batch = z_target_mu * latent_scale_factor

            # 条件: concat(z_init, mask_latent) -> (B, C_latent+1, D',H',W')
            # 务必保证参考条件与 x_start 同频缩放
            z_init_scaled = z_init_mu * latent_scale_factor
            cond = th.cat([z_init_scaled, mask_latent], dim=1)

            # 返回 (x_start, model_kwargs) 形式，TrainLoop 会作为 latent diffusion 训练
            yield batch, {"cond": cond}


def main():
    args = create_argparser().parse_args()
    
    # 动态设定输出文件夹路径
    basename = os.path.basename(os.path.normpath(args.data_dir))
    # 将类似于 "size256_mask64_sourcel2-1.485" 转换为 "mask64_size256_sourcel2-1.485"
    if basename.startswith("size") and "_mask" in basename:
        parts = basename.split("_")
        if len(parts) >= 3:
            basename = f"{parts[1]}_{parts[0]}_{parts[2]}"
            
    args.output_dir = os.path.join(
        "Z:/Outpaint_Checkpoint",
        f"{basename}_ds{args.vae_num_downsamples}_ch{args.latent_channels}_res{args.vae_num_res_blocks}",
        "ckpt"
    )
    
    logger.configure()

    logger.log("creating model and diffusion...")
    
    # 1. 创建VAE模型
    vae = VAE(
        in_channels=1,  # 3D体积数据是单通道的
        latent_channels=args.latent_channels,
        hidden_channels=args.vae_hidden_channels,
        num_res_blocks=args.vae_num_res_blocks,
        num_downsamples=args.vae_num_downsamples,
        dims=3
    )
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    vae = vae.to(device)
    
    # 极为关键：加载刚刚辛辛苦苦训好的 VAE 权重！
    if args.vae_checkpoint and os.path.isfile(args.vae_checkpoint):
        logger.log(f"Loading Pre-trained VAE weights from {args.vae_checkpoint}...")
        vae.load_state_dict(th.load(args.vae_checkpoint, map_location=device))
    else:
        logger.log("WARNING: NO VAE CHECKPOINT PROVIDED! VAE is randomly initialized! Generating gray noise...")
        raise ValueError("Please provide a valid --vae_checkpoint path to train the LDM!")
    
    # 2. 直接创建UNet模型
    from improved_diffusion.unet import UNetModel
    from improved_diffusion.respace import SpacedDiffusion, space_timesteps
    from improved_diffusion import gaussian_diffusion as gd
    
    # [主宰级架构配置]
    # 我们采取极其硬核的倍乘：底层的特征图会被狂暴放大至 320 * 4 = 1280 通道，专门用于捕获深层抽象纹理。
    channel_mult = (1, 2, 4, 4)
    
    # 全局注意力铺设：在空间特征图大小降为 16^3, 8^3, 和极深层的 4^3 时，这三层全部启用 Attention 模块
    # 注意力机制会极大提升全局视野，填补大范围空白毫无压力。
    attention_ds = [16, 8, 4]
    
    # 创建UNet模型:
    #   - x:      (B, C_latent, D',H',W')
    #   - cond:   (B, C_latent+1, D',H',W')  = concat(z_init, latent_mask)
    #   - 输入:   concat(x, cond) -> (B, 2*C_latent+1, D',H',W')
    model = UNetModel(
        in_channels=args.latent_channels * 2 + 1,
        model_channels=args.num_channels,
        out_channels=args.latent_channels * 2 if args.learn_sigma else args.latent_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=args.dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=args.use_checkpoint,
        num_heads=args.num_heads,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dims=3  # 3D模型
    )
    
    # 创建扩散模型
    betas = gd.get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    if args.use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif args.rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not args.timestep_respacing:
        timestep_respacing = [args.diffusion_steps]
    else:
        timestep_respacing = args.timestep_respacing
    
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(args.diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON if not args.predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not args.sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not args.learn_sigma else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=args.rescale_timesteps,
    )
    
    # 3. 包装UNet为条件UNet，并移动到设备上（关闭半精度）
    model = CondUNet(model)
    model = model.to(device)

    logger.log("creating data loader...")
    data = load_3d_outpainting_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        vae=vae,
        device=device,
        latent_channels=args.latent_channels,
        latent_scale_factor=args.latent_scale_factor,
        crop_size=args.crop_size,
        class_cond=args.class_cond,
    )

    logger.log("creating validation data loader...")
    val_dataset = Outpainting3DTripletDataset(root_dir=args.val_root)
    # 强制 shuffle=False，并只提取第一条作为固定测试集，方便一直比照它的同一张图效果演进
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_batch_fixed = next(iter(val_loader))

    # ---- [新增核心机制]：自动匹配潜空间缩放系数 (Latent Scaling Factor) ----
    if args.latent_scale_factor == 0.0:
        logger.log("Auto-calculating latent scale factor from the validation set...")
        with th.no_grad():
            auto_target = val_batch_fixed["target"].to(device=device, dtype=th.float32)
            auto_mu, _ = vae.encode(auto_target)
            emp_std = th.std(auto_mu).item()
            if emp_std > 0:
                args.latent_scale_factor = 1.0 / emp_std
                logger.log(f"-> Empirical std: {emp_std:.4f} | Set latent_scale_factor = {args.latent_scale_factor:.6f}")
            else:
                args.latent_scale_factor = 1.0
                logger.log("-> Warning: std is 0, setting latent_scale_factor=1.0")
    else:
        logger.log(f"Using provided latent_scale_factor: {args.latent_scale_factor}")

    logger.log("training...")

    # 优化器与时间步采样器（在 latent 空间上训练）
    opt = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    step = 0
    max_steps = args.lr_anneal_steps if args.lr_anneal_steps > 0 else 100000

    while step < max_steps:
        # 从生成器取一个 batch（latent x_start 和 cond），保持默认精度 (fp32)
        batch_latent, cond = next(data)
        batch_latent = batch_latent.to(device=device, dtype=th.float32)
        cond = {k: v.to(device=device, dtype=th.float32) for k, v in cond.items()}

        B = batch_latent.shape[0]
        # 采样时间步
        t, weights = schedule_sampler.sample(B, device)

        # 在 latent 空间上计算扩散损失
        losses = diffusion.training_losses(
            model,
            batch_latent,
            t,
            model_kwargs=cond,
        )
        loss = (losses["loss"] * weights).mean()

        # Handle fp16 NaN gracefully or fallback
        if th.isnan(loss):
            logger.log(f"step {step} NaN Loss detected! Skipping update.")
            opt.zero_grad()
            step += 1
            continue

        opt.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients leading to NaNs
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_interval == 0:
            logger.log(f"step {step} loss {loss.item():.6f}")

        # 简单的 checkpoint 保存到 output_dir
        if step % args.save_interval == 0 and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, f"ldm_ckpt_{step}.pt")
            th.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step,
                },
                ckpt_path,
            )
            logger.log(f"Saved checkpoint to {ckpt_path}")

            # ======== 验证集生成及推断 ========
            logger.log(f"Generating validation sample at step {step}...")
            model.eval()
            with th.no_grad():
                # 永远只使用固定的第一个 batch
                val_batch = val_batch_fixed
                
                val_init = val_batch["init"].to(device=device, dtype=th.float32)
                val_mask = val_batch["mask"].to(device=device, dtype=th.float32)
                
                # 移除裁剪逻辑，直接以验证集的真实尺寸（例如 256^3）进行全尺寸 3D outpaint 生成！
                # 警告：对于 24G 显存，256^3 的全尺寸推理是可以支持的，因为在推理时无需保存梯度图。
                # 如果未来遇到推理时爆显存，可以使用切片生成等策略，但此处我们直接还原原图。
                
                # 编码并拼接条件，极为重要：生成时也要同比例缩放参考 z_val_init
                z_val_init, _ = vae.encode(val_init)
                z_val_init_scaled = z_val_init * args.latent_scale_factor
                
                val_mask_latent = F.interpolate(val_mask, size=z_val_init.shape[-3:], mode="nearest")
                val_cond = th.cat([z_val_init_scaled, val_mask_latent], dim=1)
                
                sample_shape = (1, args.latent_channels, *z_val_init.shape[-3:])
                
                # 参考 former_scripts/outpaint_3d_train.py，强制非mask区域（已知区域）在每一步去噪后保持为真实 latent
                def denoised_fn(x0, e_init=z_val_init_scaled, e_mask=val_mask_latent):
                    return e_mask * x0 + (1.0 - e_mask) * e_init
                
                # 采样阶段
                logger.log("Sampling latent volume...")
                sample = diffusion.p_sample_loop(
                    model,
                    sample_shape,
                    clip_denoised=False, # 潜空间数据域一般不需要强制限制到[-1,1]
                    denoised_fn=denoised_fn,
                    model_kwargs={"cond": val_cond},
                    progress=True,
                )
                
                logger.log("Decoding latent volume to voxel space...")
                # [解码前的神级反向操作]：必须除以 scale factor，让其退却为 VAE 原本熟悉的取值区间
                sample = sample / args.latent_scale_factor
                out_vox = vae.decode(sample)
                
                # 由于 VAE 编解码是有损的（会产生灰度偏移和模糊），虽然我们在潜空间用 denoised_fn
                # 做了约束，但解码回体素空间后仍会有误差。因此在最终输出前，在体素空间再进行一次严格覆盖：
                out_vox = val_mask * out_vox + (1.0 - val_mask) * val_init
                
                val_save_path = os.path.join(args.output_dir, f"val_sample_step_{step}.pt")
                th.save({
                    "init": val_init.cpu(),      
                    "mask": val_mask.cpu(),      
                    "pred": out_vox.cpu(),       
                }, val_save_path)
                logger.log(f"Saved generated validation sample to {val_save_path}")
                
                # 额外保存生成结果作为 PNG 序列以便直观预览
                import numpy as np
                from PIL import Image
                png_dir = os.path.join(args.output_dir, f"val_sample_step_{step}_pngs")
                os.makedirs(png_dir, exist_ok=True)
                
                # out_vox 的 shape 通常是 (B, 1, D, H, W), 这里取 batch 中第一个样本
                # 我们沿着 D (Depth) 维度逐层保存 PNG，值域从 [-1, 1] 映射到 [0, 255]
                pred_volume = out_vox[0, 0].cpu().numpy()
                pred_volume = np.clip((pred_volume + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
                
                for d_idx in range(pred_volume.shape[0]):
                    slice_img = Image.fromarray(pred_volume[d_idx])
                    slice_img.save(os.path.join(png_dir, f"slice_{d_idx:03d}.png"))
                logger.log(f"Saved complete PNG sequence to {png_dir}")

            model.train()
            # ==================================

        step += 1

    logger.log("training complete")

if __name__ == "__main__":
    main()
