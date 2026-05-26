import os
import numpy as np
import cv2
from tqdm import tqdm
from scipy.linalg import sqrtm

# ========== 设置路径 ==========
recon_folder = 'C:/Users/TERONG/AppData/Local/Temp/openai-2025-07-30-21-51-02-176661/extracted_samples'  # 替换为你的推理样本路径
train_folder = 'E:/MyPython/improved-diffusion/improved_diffusion/training_data'  # 替换为你的训练样本路径


# ========== 加载图像 ==========
def load_images_from_folder(folder, size=(64, 64)):
    images = []
    for filename in sorted(os.listdir(folder)):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, size)
            img = img.astype(np.float32) / 255.0  # 归一化
            images.append(img)
    return np.stack(images)


recon_images = load_images_from_folder(recon_folder)
train_images = load_images_from_folder(train_folder)

print(f"Loaded {recon_images.shape[0]} recon images, {train_images.shape[0]} train images")


# ========== Helper Functions ==========

def mse(a, b):
    return np.mean((a - b) ** 2)


def compute_gram_matrix(image):
    vec = image.flatten()
    return np.outer(vec, vec)


def spatial_three_point_correlation(image):
    offsets = [(0, 0), (1, 1), (2, 2)]
    H, W = image.shape
    vals = []
    for i in range(H - 2):
        for j in range(W - 2):
            triplet = [image[i + dx, j + dy] for dx, dy in offsets]
            vals.append(np.prod(triplet))
    return np.array(vals)


def extract_features(images):
    flat = images.reshape(images.shape[0], -1)
    mu = np.mean(flat, axis=0)
    sigma = np.cov(flat, rowvar=False)
    return mu, sigma


# ========== Metric Functions ==========

def compute_E_px(recon_images, train_images):
    errors = []
    for recon in tqdm(recon_images, desc="E_px"):
        dists = [mse(recon, train) for train in train_images]
        errors.append(min(dists))
    return np.mean(errors)


def compute_E_S(recon_images, train_images):
    errors = []
    for recon in tqdm(recon_images, desc="E_S"):
        S_rec = compute_gram_matrix(recon)
        dists = [mse(S_rec, compute_gram_matrix(train)) for train in train_images]
        errors.append(min(dists))
    return np.mean(errors)


def compute_E_G(recon_images, train_images):
    errors = []
    for recon in tqdm(recon_images, desc="E_G"):
        G_rec = spatial_three_point_correlation(recon)
        dists = [mse(G_rec, spatial_three_point_correlation(train)) for train in train_images]
        errors.append(min(dists))
    return np.mean(errors)


def compute_fid(recon_images, train_images):
    mu1, sigma1 = extract_features(recon_images)
    mu2, sigma2 = extract_features(train_images)

    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# ========== Run and Print Results ==========

E_px = compute_E_px(recon_images, train_images)
E_S = compute_E_S(recon_images, train_images)
E_G = compute_E_G(recon_images, train_images)
FID = compute_fid(recon_images, train_images)

print(f"\nPixel space error (E_px): {E_px:.6f}")
print(f"Gram matrix error (E_S): {E_S:.6f}")
print(f"3-point correlation error (E_G): {E_G:.6f}")
print(f"FID: {FID:.6f}")
