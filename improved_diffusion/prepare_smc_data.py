import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从16位TIFF切片准备SMC训练数据（随机裁剪）")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="输入目录，包含 .tif 切片文件")
    parser.add_argument("--output_dir", type=str, default="./training_data",
                        help="裁剪后训练数据的输出目录，默认 ./training_data")
    parser.add_argument("--crop_size", type=int, default=64,
                        help="裁剪尺寸，默认 64")
    parser.add_argument("--crops_per_image", type=int, default=5,
                        help="每张图的裁剪数量，默认 5")
    return parser.parse_args()


def prepare_smc_dataset(input_dir: Path, output_dir: Path, crop_size: int, crops_per_image: int):
    output_dir.mkdir(exist_ok=True)
    print(f"输入文件夹: {input_dir.resolve()}")
    print(f"输出文件夹: {output_dir.resolve()}")

    tif_files = list(input_dir.glob("*.tif"))
    if not tif_files:
        print(f"错误：在文件夹 {input_dir} 中没有找到任何 .tif 文件。请检查路径。")
        return

    print(f"共找到 {len(tif_files)} 个 .tif 文件，开始处理...")

    total_crops_saved = 0
    for file_path in tqdm(tif_files, desc="处理图像中"):
        try:
            filename_stem = file_path.stem
            image_number_str = filename_stem.split('-')[-1]
            image_number = int(image_number_str[1:])
            if image_number % 2 == 0:
                continue
        except (ValueError, IndexError):
            print(f"\n警告：文件名 '{file_path.name}' 格式不符合预期，已跳过。")
            continue

        try:
            with Image.open(file_path) as img:
                if img.mode == 'I;16':
                    arr16 = np.array(img, dtype=np.int32)
                    min_val, max_val = arr16.min(), arr16.max()
                    if max_val == min_val:
                        arr8 = np.full(arr16.shape, 128, dtype=np.uint8)
                    else:
                        arr8 = 255 * (arr16 - min_val) / (max_val - min_val)
                        arr8 = arr8.astype(np.uint8)
                    processed_img = Image.fromarray(arr8)
                else:
                    processed_img = img.convert("L")

                if processed_img.size != (600, 600):
                    print(f"\n警告：图像 '{file_path.name}' 尺寸为 {processed_img.size}，非600x600，已跳过。")
                    continue

                width, height = processed_img.size

                for i in range(crops_per_image):
                    max_x = width - crop_size
                    max_y = height - crop_size

                    left = random.randint(0, max_x)
                    top = random.randint(0, max_y)

                    box = (left, top, left + crop_size, top + crop_size)
                    cropped_img = processed_img.crop(box)

                    output_filename = f"{filename_stem}_crop_{i}.png"
                    output_path = output_dir / output_filename
                    cropped_img.save(output_path, "PNG")
                    total_crops_saved += 1

        except Exception as e:
            print(f"\n处理文件 '{file_path.name}' 时发生错误: {e}")

    print(f"\n处理完成！总共保存了 {total_crops_saved} 个裁剪后的训练样本。")


if __name__ == "__main__":
    try:
        import PIL  # noqa: F401
        import tqdm  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        print("必需的库未安装。请运行: pip install Pillow tqdm numpy")
    else:
        args = parse_args()
        prepare_smc_dataset(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            crop_size=args.crop_size,
            crops_per_image=args.crops_per_image,
        )
