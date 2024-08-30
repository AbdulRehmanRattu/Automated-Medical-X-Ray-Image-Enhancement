import cv2
import numpy as np
import os
import argparse
from glob import glob

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.5, amount=1.5):
    """Apply unsharp mask to the image to reduce blurriness and enhance edges."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened

def estimate_noise(image):
    """Estimate the noise level of an image using the Median Absolute Deviation."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    median = np.median(gray_image)
    mad = np.median(np.abs(gray_image - median))
    noise_estimate = mad * 1.4826
    noise_factor = 0.5
    return noise_estimate * noise_factor

def adaptive_denoise(image, h_for_luminance=8, h_for_color=8):
    """Apply adaptive Non-Local Means Denoising based on the estimated noise level."""
    noise_sigma = estimate_noise(image)
    scale_factor = 15
    h_scaled_luminance = max(3, h_for_luminance * noise_sigma / scale_factor)
    h_scaled_color = max(3, h_for_color * noise_sigma / scale_factor)

    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h=h_scaled_luminance, hColor=h_scaled_color)
    bilateral_filtered = cv2.bilateralFilter(denoised_image, d=8, sigmaColor=50, sigmaSpace=50)
    return bilateral_filtered

def correct_tilt(image):
    """Correct a leftward tilt in the image using an affine transformation."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    angle = -3
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return corrected_image

def remove_black_boundaries(image):
    """Detect and remove black boundaries by cropping the largest non-black rectangle."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def adjust_contrast_brightness(image):
    """Enhance local contrast using CLAHE without compromising low-light details."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    updated_lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)
    return image

def correct_warping(image):
    """Correct minor perspective distortions with minimal alteration to overall geometry."""
    height, width = image.shape[:2]
    src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    dst_points = np.float32([[0, 0], [width - 1, 0], [int(0.02 * width), height - 1], [width - int(0.02 * width), height - 1]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    image = cv2.warpPerspective(image, matrix, (width, height))
    return image

def fill_black_spots(image):
    """Fill black spots using inpainting and enhance color intensity post-inpainting."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY_INV)
    inpaint_radius = max(3, int(min(image.shape[:2]) / 50))
    inpainted_image = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)

    hsv_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.add(s, 80)
    s = np.clip(s, 0, 255)
    v = cv2.add(v, 50)
    v = np.clip(v, 0, 255)
    enhanced_hsv_image = cv2.merge([h, s, v])
    enhanced_image = cv2.cvtColor(enhanced_hsv_image, cv2.COLOR_HSV2BGR)
    return enhanced_image

def process_images(image_dir, output_dir):
    """Process all images in the given directory and save the enhanced versions."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_path in glob(os.path.join(image_dir, '*.jpg')):
        image = cv2.imread(img_path)
        if image is None:
            continue

       
        image = correct_tilt(image)
        image = remove_black_boundaries(image)
        image = correct_warping(image)
        
        image = fill_black_spots(image)
        image = adaptive_denoise(image)
        image = adjust_contrast_brightness(image)
        image = unsharp_mask(image)

        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), image)

def main():
    parser = argparse.ArgumentParser(description="Enhance X-ray images with detail preservation.")
    parser.add_argument('image_dir', type=str, help="Directory containing X-ray images")
    args = parser.parse_args()
    process_images(args.image_dir, "Results")

if __name__ == "__main__":
    main()