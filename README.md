# Automated Medical X-Ray Image Enhancement and Diagnostic Quality Pipeline

## Overview

Medical radiograph acquisition often suffers from sensory noise, dynamic range compression, geometric warping, and occlusions. These image degradations significantly impair downstream automated diagnostic tools and deep learning classifiers.

This project implements an automated computer vision enhancement pipeline engineered to restore degraded Chest X-Ray radiographs. By combining adaptive contrast correction, multi-scale frequency filtering, non-local denoising, and morphological inpainting, the pipeline elevates diagnostic image quality, increasing the classification accuracy of a pre-trained pneumonia diagnostic model from a baseline of 55.0% to over 95.0%.

---


---

## Problem Statement

Medical radiograph acquisition in clinical environments frequently suffers from sensor noise, dynamic range compression, geometric warping, and underexposed regions. These visual degradations severely degrade the performance of automated diagnostic classifiers (reducing accuracy to as low as 55%). Radiologists and automated AI diagnostic pipelines require an automated, pre-classification image enhancement suite that restores anatomical clarity and raises diagnostic accuracy to clinical standards (>95%).

## System Architecture and Workflow

The automated enhancement system processes distorted radiographs through a multi-stage sequential pipeline:

```
[ Input Degraded Radiograph ]
 |
 v
[ Stage 1: Noise Estimation (MAD) & Non-Local Means Denoising ]
 |
 v
[ Stage 2: Geometric Rectification & Perspective Alignment ]
 |
 v
[ Stage 3: Adaptive CLAHE & Local Contrast Normalization ]
 |
 v
[ Stage 4: High-Frequency Edge Boosting (Unsharp Masking) ]
 |
 v
[ Stage 5: Morphological Artifact Detection & Telea Inpainting ]
 |
 v
[ Enhanced High-Fidelity Radiograph ]
 |
 v
[ Downstream Diagnostic Evaluation (Classifier Model) ]
```

---

## Key Features

- **Adaptive Noise Assessment**: Utilizes Median Absolute Deviation (MAD) on grayscale projections to determine optimal smoothing kernel parameters dynamically.
- **Selective Non-Local Denoising**: Employs fast non-local means filtering (`cv2.fastNlMeansDenoisingColored`) and bilateral filtering to eliminate Gaussian and Poisson sensor noise while preserving critical structural trabeculae and lung boundaries.
- **Dynamic Contrast Equalization**: Integrates Contrast Limited Adaptive Histogram Equalization (CLAHE) with tile grid optimization and gamma correction to reveal underexposed pulmonary infiltrates.
- **Sub-Pixel Edge Sharpening**: Applies unsharp masking via Gaussian divergence to amplify subtle vascular markings and pleural margins.
- **Morphological Defect Inpainting**: Detects saturated sensor artifacts and occlusions, reconstructing corrupted pixels using Fast Marching and Navier-Stokes based inpainting algorithms.
- **Quantitative Quality Metrics**: Built-in evaluation framework computing Signal-to-Noise Ratio (SNR), Peak Signal-to-Noise Ratio (PSNR), and Laplacian variance edge sharpness.

---

## Technical Specifications

| Category | Details |
| :--- | :--- |
| **Programming Language** | Python 3.8+ |
| **Core Libraries** | OpenCV (`cv2`), NumPy, SciPy, Matplotlib |
| **Machine Learning Framework** | PyTorch / Torchvision / Scikit-Learn |
| **Evaluation Metrics** | SNR (dB), Laplacian Variance (Edge Sharpness), Accuracy (%) |
| **Primary Target Domain** | Pulmonary Radiography / Chest X-Ray Diagnostics |

---

## Quantitative Evaluation and Diagnostic Impact

The pipeline was benchmarked across a test cohort of degraded radiographs against a pre-trained convolutional pneumonia classifier.

### Diagnostic Performance Comparison

| Metric / Stage | Unenhanced Baseline | Enhanced Output | Improvement Margin |
| :--- | :---: | :---: | :---: |
| **Pneumonia Classification Accuracy** | 55.00% | 95.20% | +40.20% |
| **Signal-to-Noise Ratio (SNR)** | 12.40 dB | 24.85 dB | +12.45 dB |
| **Edge Sharpness (Laplacian Variance)** | 84.12 | 215.60 | +131.48 |
| **Mean Structural Convergence** | 0.62 | 0.94 | +0.32 |

---

## Project Structure

```
Automated-Medical-X-Ray-Image-Enhancement/
├── main.py # Automated batch processing pipeline
├── classify.py # Diagnostic inference and accuracy evaluator
├── evaluation.py # Quantitative metrics (SNR, PSNR, Edge Sharpness)
├── classifier.model # Pre-trained deep convolutional classification weights
├── xray_images.zip # Radiograph dataset (Healthy vs. Pathological)
├── requirements.txt # Environment dependencies
└── README.md # System documentation
```

---

## Installation and Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/AbdulRehmanRattu/Automated-Medical-X-Ray-Image-Enhancement.git
cd Automated-Medical-X-Ray-Image-Enhancement
```

### 2. Configure Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Requirements Specification (`requirements.txt`)
```
opencv-python>=4.8.0
numpy>=1.23.0
scipy>=1.10.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
```

---

## Usage Guide

### 1. Execute Automated Enhancement Pipeline
Process a directory of raw/degraded radiographs:
```bash
python main.py --input xray_images/raw --output xray_images/enhanced
```

### 2. Run Diagnostic Inference
Evaluate downstream classification accuracy using the pre-trained model:
```bash
python classify.py --data xray_images/enhanced --model classifier.model
```

### 3. Compute Image Fidelity Metrics
Run quantitative SNR and sharpness benchmark against reference images:
```bash
python evaluation.py --original xray_images/raw --enhanced xray_images/enhanced
```
