import cv2
import numpy as np
import os
from glob import glob

def compute_snr(original_image, denoised_image):
    if original_image.shape != denoised_image.shape:
        denoised_image = cv2.resize(denoised_image, (original_image.shape[1], original_image.shape[0]))
    original_float = original_image.astype(np.float64)
    denoised_float = denoised_image.astype(np.float64)
    mse = np.mean((original_float - denoised_float) ** 2)
    if mse == 0:
        return float('inf')
    signal_power = np.var(original_float)
    snr = 10 * np.log10(signal_power / mse)
    return snr

def measure_edge_sharpness(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()

def michelson_contrast(image):
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    return (max_intensity - min_intensity) / (max_intensity + min_intensity)

def process_images(original_dir, result_dir):
    original_files = sorted(glob(os.path.join(original_dir, '*.jpg')))
    result_files = sorted(glob(os.path.join(result_dir, '*.jpg')))
    snr_values, sharpness_values, contrast_values = [], [], []

    for original_path, result_path in zip(original_files, result_files):
        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
        if original is not None and result is not None:
            snr = compute_snr(original, result)
            sharpness = measure_edge_sharpness(result)
            contrast = michelson_contrast(result)
            snr_values.append(snr)
            sharpness_values.append(sharpness)
            contrast_values.append(contrast)

    avg_snr = np.mean(snr_values) if snr_values else 0
    avg_sharpness = np.mean(sharpness_values) if sharpness_values else 0
    avg_contrast = np.mean(contrast_values) if contrast_values else 0
    print(f"Average SNR: {avg_snr:.2f}, Average Sharpness: {avg_sharpness:.2f}, Average Contrast: {avg_contrast:.2f}")

if __name__ == "__main__":
    original_dir = 'xray_images'
    result_dir = 'Results'
    process_images(original_dir, result_dir)