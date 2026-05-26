import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Simple 3D convolutional VAE for latent diffusion.

    - 输入:  (B, in_channels, D, H, W)，通常 in_channels=1，数值在 [-1, 1]
    - 潜变量: (B, latent_channels, D/4, H/4, W/4)
    - 输出:  (B, in_channels, D, H, W)，同输入尺寸

    说明:
    - 使用两次 stride=2 的 3D 卷积下采样，再用转置卷积上采样回来。
    - 不强绑定具体体素尺寸，只要求 D,H,W 能被 4 整除。
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 2,
        hidden_channels: int = 64,
        num_res_blocks: int = 2,
        num_downsamples: int = 2,
        dims: int = 3,
    ):
        super().__init__()
        assert dims == 3, "This VAE implementation is for 3D volumes (dims=3)."

        self.num_downsamples = num_downsamples

        # Encoder: dynamic downsample
        self.enc_down_convs = nn.ModuleList()
        in_ch = in_channels
        out_ch = hidden_channels
        for i in range(num_downsamples):
            self.enc_down_convs.append(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )
            in_ch = out_ch
            out_ch = min(hidden_channels * 2, out_ch * 2)

        enc_final_ch = in_ch

        # 额外的残差块
        res_blocks_enc = []
        for _ in range(max(0, num_res_blocks - 1)):
            res_blocks_enc.append(
                nn.Sequential(
                    nn.Conv3d(enc_final_ch, enc_final_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(enc_final_ch, enc_final_ch, kernel_size=3, padding=1),
                )
            )
        self.enc_res_blocks = nn.ModuleList(res_blocks_enc)

        # 输出均值和 log-variance
        self.to_mu_logvar = nn.Conv3d(
            enc_final_ch, latent_channels * 2, kernel_size=3, padding=1
        )

        # Decoder: input projection
        self.dec_conv_in = nn.Conv3d(
            latent_channels, enc_final_ch, kernel_size=3, padding=1
        )

        res_blocks_dec = []
        for _ in range(max(0, num_res_blocks - 1)):
            res_blocks_dec.append(
                nn.Sequential(
                    nn.Conv3d(enc_final_ch, enc_final_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(enc_final_ch, enc_final_ch, kernel_size=3, padding=1),
                )
            )
        self.dec_res_blocks = nn.ModuleList(res_blocks_dec)

        self.dec_up_convs = nn.ModuleList()
        in_ch = enc_final_ch
        for i in range(num_downsamples):
            # 最后一步恢复成输入通道数，其余减半
            out_ch = in_ch // 2 if in_ch > hidden_channels else hidden_channels
            if i == num_downsamples - 1:
                out_ch = in_channels
            self.dec_up_convs.append(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            in_ch = out_ch

    def encode(self, x):
        """
        返回:
          mu, logvar: 均值和对数方差
        """
        h = x
        for conv in self.enc_down_convs:
            h = F.relu(conv(h))

        for block in self.enc_res_blocks:
            h = h + block(h)

        h = self.to_mu_logvar(h)
        c = h.shape[1] // 2
        mu, logvar = h[:, :c], h[:, c:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        标准 reparameterization: z = mu + sigma * eps
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def decode(self, z):
        """
        从潜变量重建体积。
        输出形状与 encode 前输入相同 (D,H,W)。
        """
        h = F.relu(self.dec_conv_in(z))

        for block in self.dec_res_blocks:
            h = h + block(h)

        for i, up_conv in enumerate(self.dec_up_convs):
            h = up_conv(h)
            if i < len(self.dec_up_convs) - 1:
                h = F.relu(h)
        # 输入通常在 [-1,1]，这里用 tanh 保持同一范围
        return torch.tanh(h)

    def forward(self, x):
        """
        VAE 前向:
          返回 recon, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def kl_loss(mu, logvar):
    """
    标准 VAE KL 散度损失:
      KL(q(z|x) || N(0, I))
    对所有空间位置和通道求平均。
    """
    # 0.5 * sum( mu^2 + exp(logvar) - 1 - logvar )
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kld.mean()

