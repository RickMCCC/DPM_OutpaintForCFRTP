"""
Outpainting data preparer: from 600x600 inputs produce 256x256 training triplets.

For each 600x600 image in --input_dir:
  - Crop four 256x256 patches at corners as target
  - Build init (four corner blocks kept, others zero)
  - Build full_mask (unknown=1, four corners=0)
Outputs are saved under --out_dir/{init,full_mask,target}
"""

import argparse
import os

import numpy as np
from PIL import Image
import torch as th
import torch.distributed as dist

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import add_dict_to_argparser


def load_grayscale_image(path: str) -> Image.Image:
    img = Image.open(path).convert("L")
    return img


def to_model_tensor(img_gray: Image.Image) -> th.Tensor:
    arr = np.asarray(img_gray, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # 0..1 -> -1..1
    tensor = th.from_numpy(arr)[None, None, ...]  # (1,1,H,W)
    return tensor


def load_grayscale_8bit(src) -> Image.Image:
    """Load image and map 16-bit grayscale to 8-bit via min-max scaling."""
    img = Image.open(src) if isinstance(src, (str, bytes, os.PathLike)) else src
    if img.mode == "I;16":
        arr16 = np.array(img, dtype=np.int32)
        mn, mx = int(arr16.min()), int(arr16.max())
        if mx == mn:
            arr8 = np.full(arr16.shape, 128, dtype=np.uint8)
        else:
            arr8 = ((arr16 - mn) / max(1, (mx - mn)) * 255.0).astype(np.uint8)
        return Image.fromarray(arr8, mode="L")
    return img.convert("L")


def save_tensor_image(x: th.Tensor, path: str):
    x = x.clamp(-1, 1)
    x = ((x + 1) * 127.5).to(th.uint8)
    x = x[0, 0].cpu().numpy()
    img = Image.fromarray(x, mode="L")
    img.save(path)


def create_argparser():
    defaults = dict(
        input_dir="F:/CFRTP XCT/nonwoven", # F:/CFRTP XCT/1.576, F:\CFRTP XCT\nonwoven
        image_size=256,
        mask_size=50,
        out_dir="samples_outpaint",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # create scale-specific subfolder: samples_outpaint/size{H}_mask{sub}/
    image_size = int(args.image_size)
    sub = int(args.mask_size)
    input_dir = str(args.input_dir)
    source_tag = os.path.basename(os.path.normpath(input_dir))
    scale_root = os.path.join(args.out_dir, f"size{image_size}_mask{sub}_source{source_tag}")
    init_dir = os.path.join(scale_root, "init")
    mask_dir = os.path.join(scale_root, "full_mask")
    target_dir = os.path.join(scale_root, "target")
    os.makedirs(scale_root, exist_ok=True)
    os.makedirs(init_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    #dist_util.setup_dist()
    logger.configure()

    logger.log("preparing outpainting training triplets...")

    # iterate all images in input_dir
    from pathlib import Path
    src = Path(args.input_dir)
    files = sorted(list(src.glob("*.tif"))) + sorted(list(src.glob("*.png")))
    if not files:
        raise AssertionError(f"no images found in {src}")

    # image size and mask size already set above
    H = W = image_size
    sx = 600 - W
    sy = 600 - H
    crop_boxes = [
        (0, 0, W, H),
        (sx, 0, 600, H),
        (0, sy, W, 600),
        (sx, sy, 600, 600),
    ]

    total = 0
    for fp in files:
        img = load_grayscale_8bit(fp)
        if img.size != (600, 600):
            continue
        patches = [load_grayscale_8bit(img.crop(b)) for b in crop_boxes]
        # save target
        stem = fp.stem
        for i, p in enumerate(patches):
            p.save(os.path.join(target_dir, f"{stem}_corner{i}_c{W}_s{sub}.png"), "PNG")
        # build and save init/full_mask
        # prepare tensors to construct masks from target pixels
        t_tensors = [to_model_tensor(p) for p in patches]
        t = th.cat(t_tensors, dim=0)
        init = th.zeros_like(t)
        mask = th.ones_like(t)
        corners = [
            (0, 0, sub, sub),
            (W - sub, 0, W, sub),
            (0, H - sub, sub, H),
            (W - sub, H - sub, W, H),
        ]
        for (x0, y0, x1, y1) in corners:
            init[:, :, y0:y1, x0:x1] = t[:, :, y0:y1, x0:x1]
            mask[:, :, y0:y1, x0:x1] = 0.0
        # save
        for i in range(t.shape[0]):
            init_img = ((init[i:i+1].clamp(-1,1)+1)*127.5).to(th.uint8)[0,0].cpu().numpy()
            mask_img = (mask[i:i+1].clamp(0,1)*255).to(th.uint8)[0,0].cpu().numpy()
            Image.fromarray(init_img, mode="L").save(os.path.join(init_dir, f"{stem}_corner{i}_c{W}_s{sub}.png"))
            Image.fromarray(mask_img, mode="L").save(os.path.join(mask_dir, f"{stem}_corner{i}_c{W}_s{sub}.png"))
        total += 4

    logger.log(f"prepared {total} triplets (per patch: init/full_mask/target)")

    #dist.barrier()


if __name__ == "__main__":
    main()


