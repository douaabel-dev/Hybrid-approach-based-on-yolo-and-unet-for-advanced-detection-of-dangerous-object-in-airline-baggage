import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# --- 1. PATHS ---
coco_json = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\annotations_json\test\test.json"
images_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\test\images"
masks_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\test\test_masks"

os.makedirs(masks_dir, exist_ok=True)

# --- 2. LOAD JSON ---
with open(coco_json, 'r') as f:
    data = json.load(f)

# --- 3. MAP IMAGE IDS ---
image_map = {img['id']: img for img in data['images']}

# --- 4. INITIALIZE EMPTY MASKS FOR ALL IMAGES ---
mask_dict = {}

for img in data['images']:
    h = img['height']
    w = img['width']
    image_id = img['id']
    mask_dict[image_id] = np.zeros((h, w), dtype=np.uint8)

# --- 5. DRAW POLYGONS ON MASKS ---
print("Generating Masks...")

for ann in tqdm(data['annotations']):
    image_id = ann['image_id']

    if image_id not in mask_dict:
        continue

    for seg in ann['segmentation']:
        poly = np.array(seg).reshape((-1, 2)).astype(np.int32)

        cv2.fillPoly(mask_dict[image_id], [poly], 255)

# --- 6. SAVE ALL MASKS ---
print("Saving Masks...")

for image_id, mask in tqdm(mask_dict.items()):
    file_name = image_map[image_id]['file_name']

    mask_name = os.path.splitext(file_name)[0] + ".png"
    save_path = os.path.join(masks_dir, mask_name)

    cv2.imwrite(save_path, mask)

print(f"Done! {len(mask_dict)} masks saved in {masks_dir}")