import os
import shutil
import random

images_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\train\images"
labels_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\train\labels"

output_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\split_dataset"

val_img_dir = os.path.join(output_dir, "val", "images")
val_lbl_dir = os.path.join(output_dir, "val", "labels")

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# Get all images
images = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

print(f"Total images found: {len(images)}")

# Shuffle
random.shuffle(images)

# Keep only validation split (10%)
val_split = 0.1
val_images = images[:int(len(images) * val_split)]

def copy_files(file_list, img_dest, lbl_dest):
    for img_file in file_list:
        name, _ = os.path.splitext(img_file)
        label_file = name + ".txt"

        # copy image
        shutil.copy2(
            os.path.join(images_dir, img_file),
            os.path.join(img_dest, img_file)
        )

        # copy label if exists
        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(lbl_dest, label_file))
        else:
            print(f"WARNING: No label found for {img_file}")

# ONLY VAL COPY
copy_files(val_images, val_img_dir, val_lbl_dir)

print("\nDone!")
print(f"Validation set: {len(val_images)} images")