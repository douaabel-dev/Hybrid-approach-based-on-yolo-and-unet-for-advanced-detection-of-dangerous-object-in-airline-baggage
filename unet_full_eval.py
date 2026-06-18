import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# PATHS
# =========================================================

TEST_IMAGES_DIR = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\test\images"
TEST_MASKS_DIR  = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\test\test_masks"

MODEL_PATH = 'best_unet_fullimage.pth'


SAVE_DIR = "results"
PRED_MASK_DIR = os.path.join(SAVE_DIR, "pred_masks")
VIS_DIR = os.path.join(SAVE_DIR, "visualizations")

os.makedirs(PRED_MASK_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
])

# =========================================================
# DATASET
# =========================================================

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path  = os.path.join(self.mask_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = Image.fromarray(image)
        mask  = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)

        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)

        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask, image_name

# =========================================================
# LOAD DATA
# =========================================================

test_dataset = SegmentationDataset(
    TEST_IMAGES_DIR,
    TEST_MASKS_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

# =========================================================
# LOAD MODEL
# =========================================================

# Replace with your UNet class
from model import UNet

model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# =========================================================
# METRIC FUNCTIONS
# =========================================================

SMOOTH = 1e-6

def dice_score(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()

    return (2. * intersection + SMOOTH) / (
        pred.sum() + target.sum() + SMOOTH
    )

def iou_score(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + SMOOTH) / (union + SMOOTH)

# =========================================================
# METRICS STORAGE
# =========================================================

all_dice = []
all_iou = []
all_precision = []
all_recall = []
all_f1 = []
all_accuracy = []
all_specificity = []

records = []

# =========================================================
# EVALUATION LOOP
# =========================================================

with torch.no_grad():

    for images, masks, names in tqdm(test_loader):

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        # =========================================
        # METRICS
        # =========================================

        pred_np = preds.cpu().numpy().astype(np.uint8).flatten()
        mask_np = masks.cpu().numpy().astype(np.uint8).flatten()

        tn, fp, fn, tp = confusion_matrix(
            mask_np,
            pred_np,
            labels=[0, 1]
        ).ravel()

        precision = tp / (tp + fp + SMOOTH)
        recall    = tp / (tp + fn + SMOOTH)
        f1        = 2 * precision * recall / (precision + recall + SMOOTH)

        accuracy = (tp + tn) / (tp + tn + fp + fn + SMOOTH)

        specificity = tn / (tn + fp + SMOOTH)

        dice = dice_score(preds, masks).item()
        iou  = iou_score(preds, masks).item()

        # =========================================
        # STORE
        # =========================================

        all_dice.append(dice)
        all_iou.append(iou)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_accuracy.append(accuracy)
        all_specificity.append(specificity)

        records.append({
            "image": names[0],
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "specificity": specificity
        })

        # =========================================
        # SAVE MASK
        # =========================================

        pred_mask = preds.squeeze().cpu().numpy() * 255
        pred_mask = pred_mask.astype(np.uint8)

        cv2.imwrite(
            os.path.join(PRED_MASK_DIR, names[0]),
            pred_mask
        )

        # =========================================
        # VISUALIZATION
        # =========================================

        original = images.squeeze().cpu().permute(1,2,0).numpy()

        gt = masks.squeeze().cpu().numpy()

        fig, ax = plt.subplots(1,3, figsize=(15,5))

        ax[0].imshow(original)
        ax[0].set_title("Original")

        ax[1].imshow(gt, cmap='gray')
        ax[1].set_title("Ground Truth")

        ax[2].imshow(pred_mask, cmap='gray')
        ax[2].set_title("Prediction")

        for a in ax:
            a.axis("off")

        plt.tight_layout()

        plt.savefig(
            os.path.join(VIS_DIR, names[0])
        )

        plt.close()

# =========================================================
# SAVE CSV
# =========================================================

df = pd.DataFrame(records)

csv_path = os.path.join(SAVE_DIR, "metrics.csv")

df.to_csv(csv_path, index=False)

# =========================================================
# FINAL REPORT
# =========================================================

report = f"""
=============================
U-NET TEST RESULTS
=============================

Mean Dice Score      : {np.mean(all_dice):.4f}
Mean IoU             : {np.mean(all_iou):.4f}
Mean Precision       : {np.mean(all_precision):.4f}
Mean Recall          : {np.mean(all_recall):.4f}
Mean F1 Score        : {np.mean(all_f1):.4f}
Mean Accuracy        : {np.mean(all_accuracy):.4f}
Mean Specificity     : {np.mean(all_specificity):.4f}
"""

print(report)

with open(os.path.join(SAVE_DIR, "final_report.txt"), "w") as f:
    f.write(report)

# =========================================================
# PLOTS
# =========================================================

plt.figure(figsize=(8,5))
plt.hist(all_dice, bins=20)
plt.title("Dice Score Distribution")
plt.xlabel("Dice")
plt.ylabel("Frequency")
plt.savefig(os.path.join(SAVE_DIR, "dice_distribution.png"))
plt.close()

plt.figure(figsize=(8,5))
plt.hist(all_iou, bins=20)
plt.title("IoU Distribution")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.savefig(os.path.join(SAVE_DIR, "iou_distribution.png"))
plt.close()

plt.figure(figsize=(8,5))
plt.boxplot([all_dice, all_iou, all_f1],
            labels=["Dice", "IoU", "F1"])
plt.title("Metric Comparison")
plt.savefig(os.path.join(SAVE_DIR, "metric_boxplot.png"))
plt.close()

print("Evaluation completed successfully.")