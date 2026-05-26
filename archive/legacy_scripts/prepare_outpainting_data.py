"""
Prepare outpainting dataset from 600x600 grayscale slices.

For each selected slice (by stride/offset), generate four corner canvases
of size canvas_size x canvas_size (default 300), and for each canvas,
save the center region of size mask_size x mask_size (default 100) as mask.

Outputs are saved into two folders under output_dir: canvas/ and mask/.

Example:
  python IDP_outpainting/prepare_outpainting_data.py \
    --input_dir E:/your/600x600_slices \
    --output_dir E:/your/outpainting_dataset \
    --canvas_size 300 --mask_size 100 \
    --slice_stride 2 --slice_offset 0
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm 

def parse_args():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--input_dir", type=str, help="输入目录")
    # 输入文件夹路径: F:/CFRTP XCT/1.576, F:\CFRTP XCT\l2-1.485
    parser.add_argument("--output_dir", type=str, help="输出目录")
    # 输出文件夹路径: E:/MyPython/improved-diffusion/IDP_outpainting/training_data
    parser.add_argument("--canvas_size", type=int, help="画布尺寸")
    parser.add_argument("--mask_size", type=int, help="掩码尺寸")
    parser.add_argument("--slice_stride", type=int, help="步长")
    parser.add_argument("--slice_offset", type=int, help="偏移")
    # 新增：在300×300画布内对四角再采样，用于外绘制训练
    parser.add_argument("--sub_size", type=int, help="画布内四角子块尺寸（作为已知区域），默认100")
    parser.add_argument("--feather", type=int, help="full_mask边缘羽化像素，默认0")
    parser.add_argument("--export_triplets", type=int, help="是否导出三元组 init/full_mask/target (1/0)，默认1")
    parser.set_defaults(
        input_dir="F:/CFRTP XCT/1.576",
        output_dir="E:/MyPython/improved-diffusion/IDP_outpainting/training_data",
        canvas_size=300,
        mask_size=100,
        slice_stride=2,
        slice_offset=0,
        sub_size=100,
        feather=0,
        export_triplets=1,
    )
    return parser.parse_args()


def load_grayscale_8bit(img: Image.Image) -> Image.Image:
    """Convert 16-bit grayscale to 8-bit with min-max scaling; others to L."""
    if img.mode == "I;16":
        arr16 = np.array(img, dtype=np.int32)
        min_val, max_val = int(arr16.min()), int(arr16.max())
        if max_val == min_val:
            arr8 = np.full(arr16.shape, 128, dtype=np.uint8)
        else:
            scaled = (arr16 - min_val) / max(1, (max_val - min_val))
            arr8 = (scaled * 255.0).astype(np.uint8)
        return Image.fromarray(arr8, mode="L")
    return img.convert("L")


def corner_boxes(w: int, h: int, size: int) -> List[Tuple[int, int, int, int]]:
    assert size <= w and size <= h
    return [
        (0, 0, size, size),
        (w - size, 0, w, size),
        (0, h - size, size, h),
        (w - size, h - size, w, h),
    ]


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    canvas_size = int(args.canvas_size)
    mask_size = int(args.mask_size)
    slice_stride = max(1, int(args.slice_stride))
    slice_offset = int(args.slice_offset) % slice_stride

    canvas_dir = output_dir / "canvas"
    mask_dir = output_dir / "mask"
    canvas_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    # 三元组输出目录
    init_dir = output_dir / "init"
    full_mask_dir = output_dir / "full_mask"
    target_dir = output_dir / "target"
    if int(args.export_triplets):
        init_dir.mkdir(parents=True, exist_ok=True)
        full_mask_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(list(input_dir.glob("*.tif")))
    if not tif_files:
        print(f"错误：在 {input_dir} 未找到 .tif 文件")
        return

    selected_files = [p for idx, p in enumerate(tif_files) if (idx - slice_offset) % slice_stride == 0]
    print(f"输入: {input_dir.resolve()}  输出: {output_dir.resolve()}")
    print(f"总切片 {len(tif_files)}，采样 stride={slice_stride}, offset={slice_offset} -> {len(selected_files)} 张")

    total_canvas = 0
    total_mask = 0
    total_triplets = 0

    for file_path in tqdm(selected_files, desc="处理图像中"):
        try:
            with Image.open(file_path) as img:
                img8 = load_grayscale_8bit(img)
        except Exception as e:
            print(f"读取失败 {file_path.name}: {e}")
            continue

        if img8.size != (600, 600):
            print(f"跳过 {file_path.name}：尺寸为 {img8.size}，需要 600x600")
            continue

        w, h = img8.size
        if canvas_size > w or canvas_size > h:
            print(f"跳过 {file_path.name}：canvas_size={canvas_size} 超出图像尺寸")
            continue

        cx0 = (canvas_size - mask_size) // 2
        cy0 = (canvas_size - mask_size) // 2
        if cx0 < 0 or cy0 < 0:
            print(f"跳过 {file_path.name}：mask_size={mask_size} 大于 canvas_size={canvas_size}")
            continue

        for ci, box in enumerate(corner_boxes(w, h, canvas_size)):
            canvas_img = img8.crop(box)
            mask_box = (cx0, cy0, cx0 + mask_size, cy0 + mask_size)
            mask_img = canvas_img.crop(mask_box)

            stem = file_path.stem
            canvas_name = f"{stem}_corner{ci}_c{canvas_size}.png"
            mask_name = f"{stem}_corner{ci}_m{mask_size}.png"

            try:
                canvas_img.save(canvas_dir / canvas_name, "PNG")
                mask_img.save(mask_dir / mask_name, "PNG")
                total_canvas += 1
                total_mask += 1
            except Exception as e:
                print(f"保存失败 {file_path.name}: {e}")

            # 额外：为每个300×300画布生成四个同分辨率掩码（对应四角子块，尺寸为 sub_size）
            # 掩码语义：未知区域=255，需要生成；已知子块=0
            sub = int(args.sub_size)
            if sub > 0 and sub <= canvas_size:
                sub_boxes = corner_boxes(canvas_size, canvas_size, sub)
                for si, sbox in enumerate(sub_boxes):
                    x0, y0, x1, y1 = sbox
                    mask_arr_full = np.ones((canvas_size, canvas_size), dtype=np.float32)
                    mask_arr_full[y0:y1, x0:x1] = 0.0
                    if int(args.feather) > 0:
                        from PIL import ImageFilter
                        m_full = Image.fromarray((mask_arr_full * 255).astype(np.uint8), mode="L")
                        m_full = m_full.filter(ImageFilter.GaussianBlur(radius=int(args.feather)))
                        mask_arr_full = np.asarray(m_full, dtype=np.uint8) / 255.0
                    full_mask_img = Image.fromarray((mask_arr_full * 255).astype(np.uint8), mode="L")
                    full_mask_name = f"{stem}_corner{ci}_sub{si}_c{canvas_size}_m{sub}.png"
                    try:
                        full_mask_img.save(mask_dir / full_mask_name, "PNG")
                        total_mask += 1
                    except Exception as e:
                        print(f"保存四角掩码失败 {file_path.name}: {e}")

            # 生成三元组：在画布内再从四角采样 sub_size 子块，作为已知区域，目标是还原整张画布
            if int(args.export_triplets):
                sub = int(args.sub_size)
                if sub <= 0 or sub > canvas_size:
                    print(f"跳过三元组：sub_size={sub} 非法")
                else:
                    sub_boxes = corner_boxes(canvas_size, canvas_size, sub)
                    for si, sbox in enumerate(sub_boxes):
                        x0, y0, x1, y1 = sbox
                        sub_patch = canvas_img.crop(sbox)
                        # init：同尺寸黑底，仅粘贴子块
                        init_img = Image.new("L", (canvas_size, canvas_size), color=0)
                        init_img.paste(sub_patch, (x0, y0))
                        # full_mask：未知区=255，已知区=0，可选羽化
                        mask_arr = np.ones((canvas_size, canvas_size), dtype=np.float32)
                        mask_arr[y0:y1, x0:x1] = 0.0
                        if int(args.feather) > 0:
                            from PIL import ImageFilter
                            m_img = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                            m_img = m_img.filter(ImageFilter.GaussianBlur(radius=int(args.feather)))
                            mask_arr = np.asarray(m_img, dtype=np.uint8) / 255.0
                        full_mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                        # target：原始画布
                        target_img = canvas_img
                        triplet_stem = f"{stem}_corner{ci}_sub{si}_c{canvas_size}_s{sub}"
                        try:
                            init_img.save(init_dir / f"{triplet_stem}.png", "PNG")
                            full_mask_img.save(full_mask_dir / f"{triplet_stem}.png", "PNG")
                            target_img.save(target_dir / f"{triplet_stem}.png", "PNG")
                            total_triplets += 1
                        except Exception as e:
                            print(f"保存三元组失败 {file_path.name}: {e}")

    if int(args.export_triplets):
        groups = total_triplets  # 组数：每个组包含 init/full_mask/target 三个文件
        files = total_triplets * 3
        print(f"完成：canvas {total_canvas} 张，mask {total_mask} 张，triplet 组数 {groups}，三元组文件 {files} 个。")
    else:
        print(f"完成：canvas {total_canvas} 张，mask {total_mask} 张。")


if __name__ == "__main__":
    main()


