# A Hybrid Approach Based on YOLO and U-Net for Advanced Detection of Dangerous Objects in Airline Baggage

> Master's Thesis — Intelligent Systems Engineering  
> University Mustapha Stambouli, Mascara — 2025/2026

## Overview

This project presents a hybrid deep learning approach for detecting prohibited and dangerous objects in airline baggage X-ray images.

The proposed framework combines:

- **YOLOv11n** for fast object detection and localization.
- **U-Net** for pixel-level segmentation.
- **Hybrid fusion strategies** that use segmentation information to refine YOLO detections.

The main objective is to combine the speed and detection capabilities of YOLO with the spatial precision provided by U-Net, particularly in complex and cluttered X-ray baggage images.

The proposed hybrid architectures were evaluated on the **PIDray** X-ray baggage dataset. The experiments show that carefully designed YOLO–U-Net fusion can improve detection behavior compared with standalone YOLO, while also revealing an important precision–recall trade-off.

## Key Features

- PIDray X-ray baggage dataset preparation
- YOLO-format annotation conversion
- Segmentation mask generation and preprocessing
- YOLOv11n object detection
- U-Net binary segmentation
- Detection and segmentation evaluation
- Multiple YOLO–U-Net fusion architectures
- Precision, recall, mAP and IoU-based evaluation
- Dice coefficient and segmentation analysis
- Visualization of predictions and segmentation results
- Demonstration application

## Dataset

The experiments use the **PIDray** dataset, which contains X-ray images of prohibited objects.

The project uses 12 object categories:

| ID | Class |
|---:|---|
| 0 | Baton |
| 1 | Bullet |
| 2 | Gun |
| 3 | Hammer |
| 4 | Powerbank |
| 5 | Wrench |
| 6 | Handcuffs |
| 7 | Knife |
| 8 | Lighter |
| 9 | Pliers |
| 10 | Scissors |
| 11 | Sprayer |

The thesis experiments used:

- **56,000** training images
- **14,001** validation images
- **47,573** test images

The dataset itself is not included in this repository.

## Methodology

### 1. Data Preparation

The preprocessing pipeline prepares the dataset for both detection and segmentation.

For YOLO:

- Dataset organization
- Annotation conversion
- YOLO-format label generation
- `data.yaml` configuration

For U-Net:

- Image/mask alignment
- Image resizing to **512 × 512**
- BGR → RGB conversion
- Binary mask generation using thresholding
- ImageNet normalization
- Conversion to PyTorch tensors
- Training-time augmentation

### 2. YOLOv11n

YOLOv11n is used as the main object detector.

It provides:

- Bounding-box localization
- Object classification
- Confidence scores
- Fast inference

The YOLO training pipeline uses the Ultralytics framework.

### 3. U-Net

U-Net provides pixel-level segmentation to identify object regions within X-ray images.

The segmentation pipeline uses a U-Net model with a **ResNet34 encoder initialized with ImageNet pretrained weights**.

The segmentation output is then used as complementary information for the detection system.

### 4. Hybrid YOLO–U-Net Framework

The main contribution of the project is the integration of YOLO and U-Net.

Multiple fusion architectures were implemented and evaluated to study how segmentation information can improve YOLO predictions.

The experiments investigate different strategies for:

- Refining existing detections
- Recovering missed objects
- Reducing false positives
- Improving localization
- Balancing precision and recall

The results indicate that U-Net works best as a **complementary guidance module** rather than replacing YOLO's detection decisions.

## Evaluation

The models are evaluated using standard computer vision metrics.

### Object Detection

- Precision
- Recall
- mAP
- F1-score
- Per-class performance

### Segmentation

- IoU
- Dice coefficient
- Segmentation quality
- Prediction visualizations

### Hybrid System

The hybrid architectures are compared according to their impact on:

- Detection precision
- Detection recall
- False positives
- Missed detections
- Overall detection stability

## Results

The experiments demonstrate that hybrid YOLO–U-Net architectures can improve dangerous-object detection compared with the standalone YOLO baseline.

The experiments also reveal a precision–recall trade-off:

- Some architectures primarily improve **precision** by reducing false positives.
- Others improve **recall** by recovering additional detections.
- Controlled fusion strategies provide a better overall balance.

The study shows that **moderate and controlled integration of U-Net segmentation information is more effective than aggressive fusion**.

## Project Structure

```text
My_Code/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── preprocessing/
│   ├── convert_coco_masks_to_binary_masks.py
│   ├── convert_format_test.py
│   ├── convert_json_format_to_yolo_format.py
│   ├── create_empty_annotation_files_to_empty_images.py
│   ├── create_emty_annotation_files_test.py
│   ├── create masks for test set.py
│   └── split_dataset_into_train_and_validation.py
│
├── YOLO/
│   ├── train results/
│   └── evalresults/
│
├── UNET/
│   ├── train results/
│   └── evalresults/
│
├── hybrid/
│   └── final-hybrid-architectures.ipynb
│
├── evaluation/
│   ├── yolo_final_evaluation.py
│   └── unet_evaluation.py
│
├── notebooks/
│   ├── yolo-training.ipynb
│   └── train-unet-on-full.ipynb
│
├── demo/
│   └── app full.py
│
├── models/
│   ├── best.pt
│   └── best_unet.pth
│
├── results/
│
└── data/
    └── data.yaml
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd My_Code
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data preprocessing

The scripts in `preprocessing/` prepare and convert annotations and dataset files required for training.

### Train YOLO

The YOLO training workflow is available in:

```text
notebooks/yolo-training.ipynb
```

### Train U-Net

The U-Net training workflow is available in:

```text
notebooks/train-unet-on-full.ipynb
```

### Evaluate models

YOLO evaluation:

```bash
python evaluation/yolo_final_evaluation.py
```

U-Net evaluation:

```bash
python evaluation/unet_evaluation.py
```

### Hybrid experiments

The different YOLO–U-Net fusion architectures are implemented and evaluated in:

```text
hybrid/final-hybrid-architectures.ipynb
```

## Experimental Environment

Training and testing were performed using the Kaggle GPU environment with NVIDIA T4 GPUs.

The implementation used:

- Python
- PyTorch
- Ultralytics YOLO
- segmentation-models-pytorch
- OpenCV
- Albumentations
- NumPy
- Matplotlib

The computational environment was limited to approximately 30 GPU hours per week, so experiments were performed under controlled and consistent settings.

## Limitations and Future Work

The experiments highlight several possible directions for future improvement:

- More advanced fusion strategies
- Adaptive confidence calibration
- Adaptive thresholding
- Dedicated classification refinement for newly injected detections
- Evaluation on larger and more diverse X-ray datasets
- Improved robustness and generalization
- Optimization for real-time deployment

## Thesis

This repository contains the implementation associated with the Master's thesis:

**"A Hybrid Approach Based on YOLO and U-Net for Advanced Detection of Dangerous Objects in Airline Baggage"**

**Author:** Belhadjadji Douaa Houda  
**Specialty:** Intelligent Systems Engineering  
**University:** University Mustapha Stambouli, Mascara  
**Academic Year:** 2025/2026

## Technologies

```text
Python
PyTorch
YOLOv11
U-Net
OpenCV
Albumentations
NumPy
Matplotlib
Jupyter Notebook
Git
GitHub
```
## Documentation

- [Master's Thesis](docs/thesis.pdf)