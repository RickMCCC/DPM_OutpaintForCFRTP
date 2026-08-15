"""
VAE 中间输出诊断脚本：
检查 encoder mu 分布、decoder 输出范围、重建质量等。

用法:
  python scripts/diagnose_vae.py \
    --vae_checkpoint outputs/checkpoints/vae_3d/mask64_size256_sourcel2-1.485_ds2_ch8_res2/ckpt/vae_3d_ckpt_45000.pt \
    --val_root "Z:/CFRTP XCT/processed/size256_mask64_sourcel2-1.485" \
    --output_dir outputs/diagnostics/vae
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from improved_diffusion.vae import VAE
from scripts.outpainting_3d_dataset import Outpainting3DTripletDataset


def save_slice_images(tensor_3d, path, title=""):
    """保存 3D 体中间切片为 PNG"""
    mid = tensor_3d.shape[0] // 2
    arr = tensor_3d[mid].cpu().numpy()
    # 归一化到 [0, 255]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    Image.fromarray(arr.astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/diagnostics/vae")
    parser.add_argument("--latent_channels", type=int, default=8)
    parser.add_argument("--vae_hidden_channels", type=int, default=64)
    parser.add_argument("--vae_num_res_blocks", type=int, default=2)
    parser.add_argument("--vae_num_downsamples", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 VAE
    vae = VAE(
        in_channels=1,
        latent_channels=args.latent_channels,
        hidden_channels=args.vae_hidden_channels,
        num_res_blocks=args.vae_num_res_blocks,
        num_downsamples=args.vae_num_downsamples,
    ).to(device)

    ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    vae.load_state_dict(vae_sd)
    vae.eval()
    print(f"Loaded VAE from {args.vae_checkpoint}")

    # 加载验证集
    dataset = Outpainting3DTripletDataset(root_dir=args.val_root)
    num_samples = min(args.num_samples, len(dataset))

    all_mu_values = []
    all_logvar_values = []
    all_recon_errors = []

    with torch.no_grad():
        for i in range(num_samples):
            item = dataset[i]
            target = item["target"][None].to(device=device, dtype=torch.float32)  # (1, 1, D, H, W)
            init = item["init"][None].to(device=device, dtype=torch.float32)
            mask = item["mask"][None].to(device=device, dtype=torch.float32)

            print(f"\n--- Sample {i} ---")
            print(f"  Input range: [{target.min().item():.3f}, {target.max().item():.3f}]")

            # Encode
            mu, logvar = vae.encode(target)
            std = torch.exp(0.5 * logvar)

            print(f"  mu  — mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}, "
                  f"min: {mu.min().item():.4f}, max: {mu.max().item():.4f}")
            print(f"  logvar — mean: {logvar.mean().item():.4f}, std: {logvar.std().item():.4f}")
            print(f"  std (exp(0.5*logvar)) — mean: {std.mean().item():.4f}")

            all_mu_values.append(mu)
            all_logvar_values.append(logvar)

            # Decode from mu (no reparameterization)
            recon = vae.decode(mu)
            print(f"  Recon range: [{recon.min().item():.3f}, {recon.max().item():.3f}]")

            recon_err = F.mse_loss(recon, target).item()
            all_recon_errors.append(recon_err)
            print(f"  MSE(recon, target): {recon_err:.6f}")

            # Centered latent
            channel_mean = mu.mean(dim=(0, 2, 3, 4), keepdim=True)
            channel_std = mu.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-6)
            mu_centered = mu - channel_mean
            scaled = mu_centered / channel_std
            print(f"  After centering — mean: {mu_centered.mean().item():.4f}, std: {mu_centered.std().item():.4f}")
            print(f"  After scaling   — mean: {scaled.mean().item():.4f}, std: {scaled.std().item():.4f}")

            # 保存切片图片
            save_slice_images(target[0, 0], os.path.join(args.output_dir, f"sample{i}_target.png"), "Target")
            save_slice_images(recon[0, 0], os.path.join(args.output_dir, f"sample{i}_recon.png"), "Recon")
            save_slice_images(mu[0, 0], os.path.join(args.output_dir, f"sample{i}_mu_ch0.png"), "Mu ch0")
            save_slice_images(mu_centered[0, 0], os.path.join(args.output_dir, f"sample{i}_mu_centered_ch0.png"), "Mu Centered ch0")

            # Recon - Target 差异图
            diff = (recon - target).abs()[0, 0]
            save_slice_images(diff, os.path.join(args.output_dir, f"sample{i}_diff.png"), "Diff")

            # 保存 init 和 mask 用于对比
            save_slice_images(init[0, 0], os.path.join(args.output_dir, f"sample{i}_init.png"), "Init")
            save_slice_images(mask[0, 0], os.path.join(args.output_dir, f"sample{i}_mask.png"), "Mask")

    # 汇总统计
    all_mu = torch.cat(all_mu_values, dim=0)
    all_logvar = torch.cat(all_logvar_values, dim=0)
    channel_mean = all_mu.mean(dim=(0, 2, 3, 4))
    channel_std = all_mu.std(dim=(0, 2, 3, 4))

    print(f"\n{'='*50}")
    print(f"汇总统计 ({num_samples} samples):")
    print(f"  mu — global mean: {all_mu.mean().item():.4f}, global std: {all_mu.std().item():.4f}")
    print(f"  mu — min: {all_mu.min().item():.4f}, max: {all_mu.max().item():.4f}")
    print(f"  logvar — mean: {all_logvar.mean().item():.4f}")
    print(f"  per-channel mean: {channel_mean.tolist()}")
    print(f"  per-channel std: {channel_std.tolist()}")
    print(f"  平均重建 MSE: {np.mean(all_recon_errors):.6f}")
    print(f"  输出保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
