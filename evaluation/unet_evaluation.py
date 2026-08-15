# Loads the trained U-Net model and evaluates it on the test set.
# Performs preprocessing, predicts masks, computes metrics
# (Dice, IoU, Precision, Recall, F1, Pixel Accuracy),
# analyzes multiple thresholds, generates ROC curve,
# saves predicted masks, overlays, visualizations,
# and exports results as CSV and JSON reports.
import os
import cv2
import csv
import json
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_curve, auc

# paths 
TEST_IMAGES_DIR = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\test\images"
TEST_MASKS_DIR = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\test\test_masks"
MODEL_PATH = "best_unet_fullimage.pth"
OUTPUT_DIR = "unet_fullimage_test_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# image size and evaluation parameters
IMG_SIZE = 512
THRESHOLD = 0.5
THRESHOLDS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
NUM_VIS = 12
EPS = 1e-7

# device  gpu or cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load unet model
model = smp.Unet(
#ResNet34 provides a good balance between performance and computational cost.
#it extracts rich features while remaining lightweight enough for my experiments.    
    encoder_name="resnet34",
    encoder_weights=None,#because i loaded my own trained weights afterward using load_state_dict().
    in_channels=3,
    classes=1,
    activation=None,#The model outputs logits. I apply sigmoid afterward during evaluation to obtain probabilities.
).to(device)

# load trained weights
state = torch.load(MODEL_PATH, map_location=device)
if isinstance(state, dict) and 'model' in state:
    state = state['model']
model.load_state_dict(state)#a dictionary containing all learned weights and biases of the neural network.
model.eval()
print("Model loaded successfully.")

# preprocessing transformations
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(#Normalization scales pixel values to a standard distribution to improve training stability and convergence
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2(),#PyTorch models operate on tensors rather than NumPy arrays.
])

# get all test images
extensions = ('.jpg', '.jpeg', '.png', '.bmp')

image_files = sorted([
    f for f in os.listdir(TEST_IMAGES_DIR)
    if f.lower().endswith(extensions)
])
print(f"Found {len(image_files)} test images")

# storage for metrics and results
per_image = []

all_probs_small = []
all_targets_small = []

vis_images = []
vis_gt = []
vis_pred = []
vis_names = []

# folders for saving outputs
pred_masks_dir = os.path.join(OUTPUT_DIR, "predicted_masks")
overlay_dir = os.path.join(OUTPUT_DIR, "overlays")

os.makedirs(pred_masks_dir, exist_ok=True)
os.makedirs(overlay_dir, exist_ok=True)

# evaluation loop over test images
for idx, fname in enumerate(tqdm(image_files)):

    stem = os.path.splitext(fname)[0]

    image_path = os.path.join(TEST_IMAGES_DIR, fname)

    mask_path = os.path.join(TEST_MASKS_DIR, stem + ".png")

    # read image
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    original_h, original_w = image_rgb.shape[:2]

    # read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        mask = np.zeros((original_h, original_w), dtype=np.uint8)

    # resize ground truth mask
    gt = cv2.resize(
        mask,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    gt = (gt > 127).astype(np.float32)

    # apply preprocessing
    aug = transform(image=image_rgb)

    image_tensor = aug['image'].unsqueeze(0).float().to(device)

    # model inference
    with torch.no_grad():
        logits = model(image_tensor)
#sigmoid converts logits to probabilities between 0 and 1 which will then be thresholded to binary masks
    probs = torch.sigmoid(logits)[0,0].cpu().numpy()

    # sample pixels for roc curve(Using all pixels would require a large amount of memory because images contain millions of pixels. 
    # Random sampling provides an accurate ROC estimate while reducing memory usage.)
    sample_idx = np.random.choice(
        probs.size,
        min(5000, probs.size),
        replace=False
    )

    all_probs_small.extend(probs.flatten()[sample_idx])
    all_targets_small.extend(gt.flatten()[sample_idx])

    # create predicted mask
    pred_mask = (probs > THRESHOLD).astype(np.uint8) * 255

    # resize to original size
    pred_mask_original = cv2.resize(
        pred_mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    # save predicted mask
    cv2.imwrite(
        os.path.join(pred_masks_dir, fname),
        pred_mask_original
    )

    # create overlay image
    overlay = image_rgb.copy()

    overlay[pred_mask_original > 127] = (
        overlay[pred_mask_original > 127] * 0.5
        + np.array([0,255,0]) * 0.5
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(overlay_dir, fname),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    # store images for visualization
    if len(vis_images) < NUM_VIS:
        vis_images.append(image_rgb)
        vis_gt.append(gt)
        vis_pred.append(probs)
        vis_names.append(fname)

    # compute metrics for different thresholds
    row = {
        'filename': fname,
        'positive_pixels': int(gt.sum())
    }

    for t in THRESHOLDS:

        pred = (probs > t).astype(np.float32)

        TP = float((pred * gt).sum())
        FP = float((pred * (1 - gt)).sum())
        FN = float(((1 - pred) * gt).sum())
        TN = float(((1 - pred) * (1 - gt)).sum())

        dice = (2*TP + EPS) / (2*TP + FP + FN + EPS)

        iou = (TP + EPS) / (TP + FP + FN + EPS)

        precision = (TP + EPS) / (TP + FP + EPS)

        recall = (TP + EPS) / (TP + FN + EPS)

        f1 = 2 * precision * recall / (
            precision + recall + EPS
        )

        pixel_acc = (
            TP + TN + EPS
        ) / (
            TP + TN + FP + FN + EPS
        )

        bg_pred = float(
            (probs * (1 - gt)).mean()
        )

        row[f't{t}_dice'] = dice
        row[f't{t}_iou'] = iou
        row[f't{t}_precision'] = precision
        row[f't{t}_recall'] = recall
        row[f't{t}_f1'] = f1
        row[f't{t}_pixel_acc'] = pixel_acc
        row[f't{t}_bg_pred'] = bg_pred

    per_image.append(row)

# convert results to dataframe
df = pd.DataFrame(per_image)

# save per image metrics
csv_path = os.path.join(
    OUTPUT_DIR,
    "per_image_metrics.csv"
)

df.to_csv(csv_path, index=False)

print("Saved metrics CSV")

# compute global metrics for each threshold
global_metrics = {}

for t in THRESHOLDS:

    metrics = {}

    for metric in [
        'dice',
        'iou',
        'precision',
        'recall',
        'f1',
        'pixel_acc'
    ]:

        values = df[f't{t}_{metric}']

        metrics[metric] = float(values.mean())

    global_metrics[str(t)] = metrics

# save summary json
with open(
    os.path.join(OUTPUT_DIR, "summary.json"),
    "w"
) as f:

    json.dump(global_metrics, f, indent=4)

# compute roc curve
fpr, tpr, _ = roc_curve(
    all_targets_small,
    all_probs_small
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f'AUC = {roc_auc:.4f}'
)

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve.png"),
    dpi=150
)

plt.close()

# threshold analysis curves
f1_scores = []
precisions = []
recalls = []
dices = []

for t in THRESHOLDS:

    f1_scores.append(global_metrics[str(t)]['f1'])
    precisions.append(global_metrics[str(t)]['precision'])
    recalls.append(global_metrics[str(t)]['recall'])
    dices.append(global_metrics[str(t)]['dice'])

plt.figure(figsize=(8,5))

plt.plot(THRESHOLDS, f1_scores, marker='o', label='F1')
plt.plot(THRESHOLDS, precisions, marker='s', label='Precision')
plt.plot(THRESHOLDS, recalls, marker='^', label='Recall')
plt.plot(THRESHOLDS, dices, marker='D', label='Dice')

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Sweep Analysis")
plt.legend()

plt.savefig(
    os.path.join(OUTPUT_DIR, "threshold_sweep.png"),
    dpi=150
)

plt.close()

# histogram of dice scores
dice_values = df[f't{THRESHOLD}_dice']

plt.figure(figsize=(8,5))

plt.hist(dice_values, bins=30)

plt.xlabel("Dice Score")
plt.ylabel("Frequency")
plt.title("Dice Score Distribution")

plt.savefig(
    os.path.join(OUTPUT_DIR, "dice_histogram.png"),
    dpi=150
)

plt.close()

# qualitative visualization
rows = min(NUM_VIS, len(vis_images))

fig, axes = plt.subplots(rows, 3, figsize=(12, 4*rows))

if rows == 1:
    axes = np.expand_dims(axes, axis=0)

for i in range(rows):

    pred_bin = (vis_pred[i] > THRESHOLD).astype(np.uint8)

    axes[i,0].imshow(vis_images[i])
    axes[i,0].set_title("Original")

    axes[i,1].imshow(vis_gt[i], cmap='gray')
    axes[i,1].set_title("Ground Truth")

    axes[i,2].imshow(pred_bin, cmap='gray')
    axes[i,2].set_title("Prediction")

    for j in range(3):
        axes[i,j].axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "qualitative_results.png"),
    dpi=150
)

plt.close()

# final report
best_threshold = THRESHOLDS[np.argmax(f1_scores)]

print("\n" + "="*60)
print("U-NET TEST EVALUATION FINISHED")
print("="*60)

print(f"\nROC AUC: {roc_auc:.4f}")

print(f"\nBest threshold by F1: {best_threshold}")

print("\nMetrics at threshold 0.5:")

main_metrics = global_metrics[str(THRESHOLD)]

for k, v in main_metrics.items():
    print(f"{k}: {v:.4f}")

print(f"\nOutputs saved in: {OUTPUT_DIR}")

print("="*60) 