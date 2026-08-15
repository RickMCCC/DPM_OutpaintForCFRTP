import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import lpips
except ImportError:
    pass

from improved_diffusion import logger
from improved_diffusion.vae import VAE, kl_loss
from scripts.outpainting_3d_dataset import Outpainting3DTripletDataset, CachedOutpainting3DDataset

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="data/processed/l2-1.485_train/size256_mask64_sourcel2-1.485", help="Path to training data")
    parser.add_argument("--val_root", type=str, default="data/processed/l2-1.485_val/size256_mask64_sourcel2-1.485", help="Path to validation data")
    parser.add_argument("--train_cache_dir", type=str, default="", help="Path to cached training .pt dataset")
    parser.add_argument("--val_cache_dir", type=str, default="", help="Path to cached validation .pt dataset")
    parser.add_argument("--output_dir", type=str, default="vae_3d_checkpoint", help="Local directory to save checkpoints")
    parser.add_argument("--remote_output_dir", type=str, default="", help="Remote dir to copy checkpoints to after saving (e.g. Z:/Outpaint_Checkpoint/...)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=0, help="Optional global step limit; 0 uses epochs")
    parser.add_argument("--log_interval", type=int, default=10, help="Steps between logging")
    parser.add_argument("--save_interval", type=int, default=2000, help="Steps between model saving")
    
    # VAE Architecture parameters (Must match what you plan to use in LDM!)
    parser.add_argument("--latent_channels", type=int, default=8, help="Number of latent channels")
    parser.add_argument("--vae_hidden_channels", type=int, default=64, help="Base hidden channels for VAE")
    parser.add_argument("--vae_num_res_blocks", type=int, default=2, help="Number of residual blocks in VAE")
    parser.add_argument("--vae_num_downsamples", type=int, default=2, help="Number of downsampling layers (2 = 4x compression)")
    parser.add_argument("--kl_weight", type=float, default=1e-3, help="Final KL divergence weight")
    parser.add_argument("--kl_warmup_steps", type=int, default=10000, help="Linear KL warmup duration")
    parser.add_argument("--perceptual_weight", type=float, default=0.1, help="Weight for LPIPS Perceptual loss")
    parser.add_argument("--lpips_slices_per_volume", type=int, default=16, help="Number of depth slices sampled for LPIPS")
    
    # 3D Patch parameters
    parser.add_argument("--crop_size", type=int, default=128, help="Random 3D crop size during training to save VRAM")
    
    # Checkpoint loading
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Path to checkpoint to resume from (e.g., path/to/vae_3d_ckpt_16000.pt)")
    return parser


def load_yaml_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --config is used") from exc
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def flatten_config(config):
    mapping = {
        ("data", "train_dir"): "data_dir",
        ("data", "val_dir"): "val_root",
        ("output", "dir"): "output_dir",
        ("model", "hidden_channels"): "vae_hidden_channels",
        ("model", "num_res_blocks"): "vae_num_res_blocks",
        ("model", "num_downsamples"): "vae_num_downsamples",
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
    cli_keys = {
        token[2:].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    for key, value in flatten_config(load_yaml_config(args.config)).items():
        if hasattr(args, key) and key not in cli_keys and value is not None:
            setattr(args, key, value)
    return args


def make_dataset(data_dir, cache_dir):
    if cache_dir and os.path.isdir(cache_dir):
        return CachedOutpainting3DDataset(cache_dir=cache_dir)
    return Outpainting3DTripletDataset(root_dir=data_dir)

def main():
    parser = create_argparser()
    args = apply_config(parser.parse_args(), parser)
    
    # 动态设定输出文件夹路径
    basename = os.path.basename(os.path.normpath(args.data_dir))
    if basename.startswith("size") and "_mask" in basename:
        parts = basename.split("_")
        if len(parts) >= 3:
            basename = f"{parts[1]}_{parts[0]}_{parts[2]}"
            
    # 本地输出目录（C 盘，快速读写）
    tag = f"{basename}_ds{args.vae_num_downsamples}_ch{args.latent_channels}_res{args.vae_num_res_blocks}"
    output_base = "outputs/checkpoints/vae_3d" if args.output_dir == "vae_3d_checkpoint" else args.output_dir
    output_path = Path(output_base)
    if output_path.name != "ckpt":
        output_path = output_path / tag / "ckpt"
    args.output_dir = str(output_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # 远程输出目录（Z 盘，训练完成后拷贝）
    if args.remote_output_dir:
        args.remote_output_dir = os.path.join(args.remote_output_dir, tag, "ckpt")
    with open(os.path.join(args.output_dir, "config_used.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False, sort_keys=True)
    
    logger.configure()
    logger.log("Creating 3D VAE model...")
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # 初始化 LPIPS 模型
    loss_fn_vgg = None
    if args.perceptual_weight > 0:
        try:
            logger.log("Loading LPIPS model for perceptual loss...")
            loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
            loss_fn_vgg.eval()  # 固定 VGG 权重
            for parameter in loss_fn_vgg.parameters():
                parameter.requires_grad_(False)
        except NameError:
            logger.log("WARNING: lpips module not found. Perceptual loss is disabled. Please run `pip install lpips`.")
            args.perceptual_weight = 0.0
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    vae = VAE(
        in_channels=1,
        latent_channels=args.latent_channels,
        hidden_channels=args.vae_hidden_channels,
        num_res_blocks=args.vae_num_res_blocks,
        num_downsamples=args.vae_num_downsamples,
        dims=3
    ).to(device)

    optimizer = th.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=1e-4)

    # 如果有之前的VAE权重，加载断点继续训练
    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        logger.log(f"Resuming from checkpoint: {args.resume_checkpoint}")
        ckpt_data = th.load(args.resume_checkpoint, map_location=device)
        if isinstance(ckpt_data, dict) and "model" in ckpt_data:
            vae.load_state_dict(ckpt_data["model"])
            if "optimizer" in ckpt_data:
                try:
                    optimizer.load_state_dict(ckpt_data["optimizer"])
                except ValueError:
                    logger.log("Optimizer state mismatch; starting fresh optimizer.")
            step = int(ckpt_data.get("step", 0)) + 1
            logger.log(f"Resumed at step: {step}")
        else:
            vae.load_state_dict(ckpt_data)
            try:
                import re
                m = re.search(r"ckpt_(\d+)\.", args.resume_checkpoint)
                if m:
                    step = int(m.group(1))
                    logger.log(f"Resumed at step: {step}")
            except Exception:
                pass
    
    logger.log(f"Loading datasets...")
    dataset = make_dataset(args.data_dir, args.train_cache_dir)
    logger.log(f"Training samples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # 验证集：优先用缓存
    val_dataset = make_dataset(args.val_root, args.val_cache_dir)
    duplicate_names = sorted(set(dataset.names) & set(val_dataset.names))
    if duplicate_names:
        raise ValueError(
            "Training and validation datasets overlap: " + ", ".join(duplicate_names[:5])
        )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_batch_fixed = next(iter(val_loader))
    logger.log("Validation batch cached in memory.")
    
    # 这里的 step 变量在之前已经根据断点被推断（如果使用 resume_checkpoint）
    if 'step' not in locals():
        step = 0
        
    logger.log("Starting VAE training...")
    
    for epoch in range(args.epochs):
        for batch_dict in loader:
            if args.max_steps > 0 and step >= args.max_steps:
                break
            # VAE 只关心能不能把图像学懂，所以我们只喂给它 target 也就是完整无缺的高分辨率真实体素块！
            target_vox = batch_dict["target"].to(device=device, dtype=th.float32)
            
            # 随机裁剪 3D Patch 以防止 OOM
            if args.crop_size > 0 and target_vox.shape[-1] > args.crop_size:
                d_idx = th.randint(0, target_vox.shape[-3] - args.crop_size + 1, (1,)).item()
                h_idx = th.randint(0, target_vox.shape[-2] - args.crop_size + 1, (1,)).item()
                w_idx = th.randint(0, target_vox.shape[-1] - args.crop_size + 1, (1,)).item()
                target_vox = target_vox[..., d_idx:d_idx+args.crop_size, h_idx:h_idx+args.crop_size, w_idx:w_idx+args.crop_size]
            
            # 前向传播
            optimizer.zero_grad()
            recon, mu, logvar = vae(target_vox)
            
            # 计算重建损失 (L1 Loss 更利于保持边缘锐利度，MSE 更平滑)
            recon_loss = F.l1_loss(recon, target_vox) + F.mse_loss(recon, target_vox)
            
            # 计算 KL 散度 (逼迫潜变量符合正态分布)
            kld_loss = kl_loss(mu, logvar)
            
            # 3D 转换 2D 计算 LPIPS (沿着 Depth 维度切片)
            p_loss = th.tensor(0.0, device=device)
            if loss_fn_vgg is not None and args.perceptual_weight > 0:
                # target_vox / recon shape: [B, 1, D, H, W] -> [B*D, 1, H, W]
                B, C, D, H, W = target_vox.shape
                slice_count = max(1, min(int(args.lpips_slices_per_volume), D))
                slice_indices = th.linspace(
                    0, D - 1, steps=slice_count, device=target_vox.device
                ).round().long()
                t_2d = target_vox[:, :, slice_indices].permute(0, 2, 1, 3, 4).reshape(-1, 1, H, W)
                r_2d = recon[:, :, slice_indices].permute(0, 2, 1, 3, 4).reshape(-1, 1, H, W)
                
                # lpips 需要 3 通道输入
                t_2d = t_2d.repeat(1, 3, 1, 1)
                r_2d = r_2d.repeat(1, 3, 1, 1)
                
                # 为了防止 OOM，将 2D 切片分批输入 LPIPS 计算差异
                chunk_size = 4
                lpips_losses = []
                for i in range(0, t_2d.size(0), chunk_size):
                    lpips_losses.append(
                        loss_fn_vgg(r_2d[i:i+chunk_size], t_2d[i:i+chunk_size])
                    )
                p_loss = th.cat(lpips_losses).mean()

            # 总损失
            kl_progress = min(1.0, step / max(1, int(args.kl_warmup_steps)))
            effective_kl_weight = float(args.kl_weight) * kl_progress
            total_loss = recon_loss + effective_kl_weight * kld_loss + args.perceptual_weight * p_loss
            
            if not th.isfinite(total_loss):
                logger.log(f"step {step} non-finite loss detected; skipping update.")
                optimizer.zero_grad()
                step += 1
                continue
                
            total_loss.backward()
            th.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            
            if step % args.log_interval == 0:
                lpips_val = p_loss.item() if isinstance(p_loss, th.Tensor) else 0.0
                channel_mean = mu.mean(dim=(0, 2, 3, 4)).detach().cpu().tolist()
                channel_std = mu.std(dim=(0, 2, 3, 4)).detach().cpu().tolist()
                clip_fraction = ((recon < -1.0) | (recon > 1.0)).float().mean().item()
                logger.log(
                    f"step {step} | Total: {total_loss.item():.4f} | "
                    f"Recon: {recon_loss.item():.4f} | KL: {kld_loss.item():.6f} | "
                    f"KL weight: {effective_kl_weight:.6g} | LPIPS: {lpips_val:.4f} | "
                    f"logvar mean: {logvar.mean().item():.4f} | clip: {clip_fraction:.6f} | "
                    f"mu mean: {channel_mean} | mu std: {channel_std}"
                )
                
            if step > 0 and step % args.save_interval == 0:
                ckpt_path = os.path.join(args.output_dir, f"vae_3d_ckpt_{step}.pt")
                th.save({
                    "model": vae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "config": vars(args),
                }, ckpt_path)
                logger.log(f"Saved VAE checkpoint locally to {ckpt_path}")

                # 异步拷贝到远程目录
                if args.remote_output_dir:
                    import shutil
                    try:
                        os.makedirs(args.remote_output_dir, exist_ok=True)
                        remote_path = os.path.join(args.remote_output_dir, f"vae_3d_ckpt_{step}.pt")
                        shutil.copy2(ckpt_path, remote_path)
                        logger.log(f"Copied checkpoint to remote: {remote_path}")
                    except Exception as e:
                        logger.log(f"WARNING: Failed to copy checkpoint to remote: {e}")
                
                # 可视化在验证集上的重建效果
                vae.eval()
                with th.no_grad():
                    val_target = val_batch_fixed["target"].to(device=device, dtype=th.float32)
                    
                    # 验证集也裁出一块中心来看看效果 (如果显存够大可以直接整图过)
                    if args.crop_size > 0 and val_target.shape[-1] > args.crop_size:
                        c_d = (val_target.shape[-3] - args.crop_size) // 2
                        c_h = (val_target.shape[-2] - args.crop_size) // 2
                        c_w = (val_target.shape[-1] - args.crop_size) // 2
                        val_target = val_target[..., c_d:c_d+args.crop_size, c_h:c_h+args.crop_size, c_w:c_w+args.crop_size]
                    
                    val_recon, val_mu, val_logvar = vae(val_target)
                    val_metrics = {
                        "mse": F.mse_loss(val_recon, val_target).item(),
                        "logvar_mean": val_logvar.mean().item(),
                        "kl": kl_loss(val_mu, val_logvar).item(),
                        "clip_fraction": (
                            (val_recon < -1.0) | (val_recon > 1.0)
                        ).float().mean().item(),
                    }
                    with open(
                        os.path.join(args.output_dir, f"vae_val_step_{step}_metrics.json"),
                        "w",
                        encoding="utf-8",
                    ) as file:
                        json.dump(val_metrics, file, indent=2, sort_keys=True)
                    logger.log(f"VAE validation metrics: {json.dumps(val_metrics, sort_keys=True)}")
                    
                    png_dir = os.path.join(args.output_dir, f"vae_val_step_{step}_pngs")
                    os.makedirs(png_dir, exist_ok=True)
                    
                    # 保存一些对比切片图像 (真实 vs 重建)
                    # 同样取 batch 和 channel 0
                    true_vol = val_target[0, 0].cpu().numpy()
                    recon_vol = val_recon[0, 0].cpu().numpy()
                    
                    # 映射 [-1, 1] 到 [0, 255]
                    true_vol = np.clip((true_vol + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
                    recon_vol = np.clip((recon_vol + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
                    
                    # 挑10个等距切面保存作对比
                    depth = true_vol.shape[0]
                    indices_to_save = np.linspace(0, depth - 1, num=10, dtype=int)
                    
                    for z in indices_to_save:
                        # 拼接到一张图： 左侧真实，右侧重建
                        combined = np.concatenate([true_vol[z], recon_vol[z]], axis=1)
                        img = Image.fromarray(combined)
                        img.save(os.path.join(png_dir, f"compare_z{z:03d}.png"))
                        
                vae.train()
                logger.log(f"Saved reconstruction testing visually to {png_dir}")
                
            step += 1

        if args.max_steps > 0 and step >= args.max_steps:
            break

    logger.log("VAE training complete!")

if __name__ == "__main__":
    main()
