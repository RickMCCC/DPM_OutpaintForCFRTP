# prepare_outpainting_data.py (2D) vs prepare_outpainting_3d_data.py (3D) 对比

## 主要差异总结

### 1. **数据输入方式**

**2D版本**:
- 输入：单张600×600的.tif图像文件
- 处理方式：逐个文件处理
- 文件选择：通过`slice_stride`和`slice_offset`筛选文件

**3D版本**:
- 输入：连续的.tif切片序列（整个体数据）
- 处理方式：一次性加载所有切片，组装成3D体数据
- 文件选择：加载目录中所有.tif文件

### 2. **采样策略**

**2D版本**:
- 从每张图像裁剪**4个角落**的画布（canvas_size × canvas_size）
- 每个画布内再提取**4个角落**的子块
- 总共：每张图像 → 4个画布 → 16个三元组

**3D版本**:
- 从体数据中裁剪立方体块（volume_size³）
- 每个立方体有**8个顶角**
- 每个顶角提取一个子块
- 总共：每个立方体块 → 8个三元组

### 3. **输出文件夹结构**

**2D版本**:
```
output_dir/
├── canvas/       # 四个角落的画布（256×256）
├── mask/         # 中心掩码区域（可选）
├── init/         # 初始图像
├── full_mask/    # 完整掩码（三元组）
└── target/       # 目标图像
```

**3D版本** (当前状态，应该修改):
```
output_dir/
├── canvas/       # ❌ 应该删除
├── mask/         # ❌ 应该删除（或改为full_mask）
├── init/         # ✅ 初始图像（每个切片单独文件）
├── full_mask/    # ✅ 完整掩码（每个切片单独文件）
└── target/       # ✅ 目标图像（每个切片单独文件）
```

**3D版本** (理想状态):
```
output_dir/
├── mask/         # 或 full_mask/（完整掩码）
├── init/         # 初始图像
└── target/       # 目标图像
```

### 4. **文件保存方式**

**2D版本**:
- 每个三元组保存为**单张PNG文件**
- 文件名：`{stem}_corner{ci}_sub{si}_c{canvas_size}_s{sub_size}.png`
- 示例：`image_001_corner0_sub0_c300_s100.png`

**3D版本**:
- 每个三元组的每个Z切片保存为**单独的PNG文件**
- 文件名：`{block_id}_corner{ci}_c{volume_size}_s{sub_size}_z{zzz}.png`
- 示例：`z0000_y0000_x0000_corner0_c128_s32_z000.png`（共128个文件）

### 5. **命名格式差异**

**2D版本**:
```
{stem}_corner{ci}_sub{si}_c{canvas_size}_s{sub_size}.png
```
- `stem`: 原始文件名（不含扩展名）
- `corner{ci}`: 画布的角落索引（0-3）
- `sub{si}`: 画布内子块的索引（0-3）
- `c{canvas_size}`: 画布尺寸
- `s{sub_size}`: 子块尺寸

**3D版本**:
```
{block_id}_corner{ci}_c{volume_size}_s{sub_size}_z{zzz}.png
```
- `block_id`: 立方体块标识（`z{zzzz}_y{yyyy}_x{xxxx}`）
- `corner{ci}`: 立方体的顶角索引（0-7）
- `c{volume_size}`: 立方体尺寸
- `s{sub_size}`: 子块尺寸
- `z{zzz}`: 切片索引（000-127）

**关键差异**：
- 2D有`sub{si}`（画布内的4个子块），3D没有（直接是8个顶角）
- 3D有`z{zzz}`后缀（切片索引），2D没有

### 6. **参数差异**

**2D版本特有参数**:
- `--slice_stride`: 切片采样步长（筛选输入文件）
- `--slice_offset`: 切片偏移（筛选输入文件）
- `--mask_size`: 中心掩码尺寸（用于生成mask文件夹）

**3D版本特有参数**:
- `--stride_z/--stride_y/--stride_x`: 立方体采样步长（三个方向）
- `--limit_blocks`: 调试用，限制处理的立方体数量
- `--mask_size`: ❌ 应该删除（不再需要）

**共同参数**:
- `--input_dir`: 输入目录
- `--output_dir`: 输出目录
- `--canvas_size`/`--volume_size`: 画布/立方体尺寸
- `--sub_size`: 角落子块尺寸
- `--feather`: 掩码羽化
- `--export_triplets`: 是否导出三元组

### 7. **数据处理逻辑差异**

**2D版本**:
```python
# 1. 从图像裁剪4个角落画布
for ci, box in enumerate(corner_boxes(w, h, canvas_size)):
    canvas_img = img8.crop(box)
    
    # 2. 从画布提取4个角落子块
    for si, sbox in enumerate(corner_boxes(canvas_size, canvas_size, sub_size)):
        # 3. 生成三元组
        init = 黑底 + 子块
        mask = 未知区域=255，已知区域=0
        target = 完整画布
```

**3D版本**:
```python
# 1. 从体数据裁剪立方体块
for (z0, y0, x0), block in iter_blocks(volume, volume_size, ...):
    # 2. 从立方体提取8个顶角子块
    for ci, (z1, y1, x1, z2, y2, x2) in enumerate(corner_boxes_3d(...)):
        # 3. 生成三元组
        init = 黑底 + 子块（3D）
        mask = 未知区域=255，已知区域=0（3D）
        target = 完整立方体（3D）
        
        # 4. 每个Z切片单独保存
        for z_idx in range(volume_size):
            save_slice(init[z_idx], mask[z_idx], target[z_idx])
```

### 8. **角落提取方式**

**2D版本**:
```python
# 2D: 4个角落（2×2网格）
corner_boxes = [
    (0, 0, size, size),           # 左上
    (w-size, 0, w, size),         # 右上
    (0, h-size, size, h),         # 左下
    (w-size, h-size, w, h),       # 右下
]
```

**3D版本**:
```python
# 3D: 8个顶角（2×2×2立方体）
corner_boxes = [
    (0, 0, 0, sub, sub, sub),                    # 前-上-左
    (size-sub, 0, 0, size, sub, sub),           # 前-上-右
    (0, size-sub, 0, sub, size, sub),           # 前-下-左
    (size-sub, size-sub, 0, size, size, sub),   # 前-下-右
    (0, 0, size-sub, sub, sub, size),           # 后-上-左
    (size-sub, 0, size-sub, size, sub, size),   # 后-上-右
    (0, size-sub, size-sub, sub, size, size),   # 后-下-左
    (size-sub, size-sub, size-sub, size, size, size), # 后-下-右
]
```

### 9. **掩码生成差异**

**2D版本**:
- 使用PIL的`ImageFilter.GaussianBlur`进行羽化
- 掩码范围：0-255（uint8）

**3D版本**:
- 使用`scipy.ndimage.gaussian_filter`进行3D羽化
- 掩码范围：先0-1（float32），后转换为0-255（uint8）

### 10. **数据量差异**

**2D版本**:
- 每张图像生成：4个画布 × 4个子块 = 16个三元组
- 每个三元组：3个文件（init/full_mask/target）
- 总计：16 × 3 = 48个文件/图像

**3D版本**:
- 每个立方体生成：8个顶角 = 8个三元组
- 每个三元组：3个类型 × volume_size个切片 = 3 × 128 = 384个文件
- 总计：8 × 384 = 3072个文件/立方体

## 需要修复的问题

### 3D版本当前存在的问题：

1. **❌ 仍保存canvas和mask文件夹**（应该删除）
2. **❌ 文件夹命名不一致**（应该使用`full_mask`而不是`mask`，与数据集加载器兼容）
3. **❌ mask_size参数仍存在**（应该删除，不再需要中心掩码）
4. **❌ 命名格式缺少`sub{si}`**（虽然3D是8个顶角，但命名应该更清晰）

## 建议的修改

1. **删除canvas和mask文件夹的保存逻辑**
2. **将mask文件夹改为full_mask**（与`OutpaintingTripletDataset`兼容）
3. **删除mask_size参数及相关代码**
4. **确保命名格式与2D版本风格一致**

