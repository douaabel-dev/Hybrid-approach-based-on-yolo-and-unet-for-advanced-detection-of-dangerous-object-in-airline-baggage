import json
import os


coco_json_path = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\annotations\train\train.json"
images_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\train_images"
yolo_labels_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\labels\train"

os.makedirs(yolo_labels_dir, exist_ok=True)

# Load COCO JSON
with open(coco_json_path) as f:
    coco = json.load(f)

# Map image_id to file name
image_id_to_name = {img['id']: img['file_name'] for img in coco['images']}


for ann in coco['annotations']:
    image_id = ann['image_id']
    file_name = image_id_to_name[image_id]

 # Get bbox in COCO format 
    bbox = ann['bbox']
    x, y, w, h = bbox

 # Image size
    img_info = next(img for img in coco['images'] if img['id'] == image_id)
    img_w, img_h = img_info['width'], img_info['height']

# Convert to YOLO format (normalized center x, center y, width, height)
    x_center = (x + w/2) / img_w
    y_center = (y + h/2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

# bcs(COCO categories usually start from 1 but yolo requires starting from 0)
    class_id = ann['category_id'] - 1 

 # label file path
    label_file = os.path.join(yolo_labels_dir, os.path.splitext(file_name)[0] + ".txt")

# Append (if multiple boxe per image)
    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")





