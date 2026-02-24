import os
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np  # 确保导入 numpy

# --- 1. 配置参数 ---
# !!! 请根据您的实际情况修改这两个路径 !!!
INPUT_DIR = Path(r"F:/CFRTP XCT/1.576")  # 输入文件夹路径: F:/CFRTP XCT/1.576, F:\CFRTP XCT\l2-1.485
OUTPUT_DIR = Path("./training_data")  # 保存裁剪后训练数据的输出文件夹路径



# 裁剪参数
CROP_SIZE = 64
CROPS_PER_IMAGE = 5  # 建议增加此值以扩充数据集


def prepare_smc_dataset():
    """
    遍历输入目录，对16位灰度图进行手动范围缩放，然后进行随机裁剪并保存结果。
    """
    # --- 2. 准备工作 ---
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"输入文件夹: {INPUT_DIR.resolve()}")
    print(f"输出文件夹: {OUTPUT_DIR.resolve()}")

    tif_files = list(INPUT_DIR.glob("*.tif"))
    if not tif_files:
        print(f"错误：在文件夹 {INPUT_DIR} 中没有找到任何 .tif 文件。请检查路径。")
        return

    print(f"共找到 {len(tif_files)} 个 .tif 文件，开始处理...")

    total_crops_saved = 0
    for file_path in tqdm.tqdm(tif_files, desc="处理图像中"):
        # --- 3. 解析文件名并筛选奇数编号 (此部分逻辑根据您的代码保留) ---
        try:
            filename_stem = file_path.stem
            image_number_str = filename_stem.split('-')[-1]
            image_number = int(image_number_str[1:])  # 假设您的编号前有一个非数字字符
            if image_number % 2 == 0:
                continue
        except (ValueError, IndexError):
            print(f"\n警告：文件名 '{file_path.name}' 格式不符合预期，已跳过。")
            continue

        # --- 4. 打开图像并进行处理 ---
        try:
            with Image.open(file_path) as img:
                # ==================== 新增：处理16位图像的核心逻辑 ====================
                if img.mode == 'I;16':
                    # 1. 将16位图像数据完整读入Numpy数组
                    arr16 = np.array(img, dtype=np.int32)

                    # 2. 找到实际的最小和最大值
                    min_val, max_val = arr16.min(), arr16.max()

                    # 3. 避免图像是纯色导致除以零
                    if max_val == min_val:
                        arr8 = np.full(arr16.shape, 128, dtype=np.uint8)
                    else:
                        # 4. 手动将 [min_val, max_val] 范围线性映射到 [0, 255]
                        arr8 = 255 * (arr16 - min_val) / (max_val - min_val)
                        arr8 = arr8.astype(np.uint8)

                    # 5. 从处理好的8位Numpy数组创建新的Pillow图像
                    processed_img = Image.fromarray(arr8)
                else:
                    # 如果不是16位图，按常规方式转为8位灰度图
                    processed_img = img.convert("L")
                # =====================================================================

                # 验证图像尺寸
                if processed_img.size != (600, 600):
                    print(f"\n警告：图像 '{file_path.name}' 尺寸为 {processed_img.size}，非600x600，已跳过。")
                    continue

                width, height = processed_img.size

                # 对每张图片进行指定次数的裁剪
                for i in range(CROPS_PER_IMAGE):
                    max_x = width - CROP_SIZE
                    max_y = height - CROP_SIZE

                    left = random.randint(0, max_x)
                    top = random.randint(0, max_y)

                    box = (left, top, left + CROP_SIZE, top + CROP_SIZE)

                    # 使用处理后的图像进行裁剪
                    cropped_img = processed_img.crop(box)

                    # --- 5. 保存裁剪后的图像 ---
                    output_filename = f"{filename_stem}_crop_{i}.png"
                    output_path = OUTPUT_DIR / output_filename

                    cropped_img.save(output_path, "PNG")
                    total_crops_saved += 1

        except Exception as e:
            print(f"\n处理文件 '{file_path.name}' 时发生错误: {e}")

    print(f"\n处理完成！总共保存了 {total_crops_saved} 个裁剪后的训练样本。")


if __name__ == "__main__":
    try:
        import PIL
        import tqdm
        import numpy
    except ImportError:
        print("必需的库未安装。请运行: pip install Pillow tqdm numpy")
    else:
        prepare_smc_dataset()