import os
import shutil
import random



images_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\train\images"
labels_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\train\labels"
output_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\split_dataset"



train_img_dir = os.path.join(output_dir, "train", "images")
train_lbl_dir = os.path.join(output_dir, "train", "labels")
val_img_dir = os.path.join(output_dir, "val", "images")
val_lbl_dir = os.path.join(output_dir, "val", "labels")

for folder in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(folder, exist_ok=True)



images = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

print(f"Total images found: {len(images)}")

# Shuffle images
random.shuffle(images)

# SPLIT 90/10

split_index = int(len(images) * 0.9)
train_images = images[:split_index]
val_images = images[split_index:]
    


def copy_files(file_list, img_dest, lbl_dest):
    for img_file in file_list:
        name, ext = os.path.splitext(img_file)
        label_file = name + ".txt"

        # Copy image
        shutil.copy2(
            os.path.join(images_dir, img_file),
            os.path.join(img_dest, img_file)
        )

        # Copy corresponding label
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(lbl_dest, label_file))
        else:
            print(f"WARNING: No label found for {img_file}")


#  SPLIT


copy_files(train_images, train_img_dir, train_lbl_dir)
copy_files(val_images, val_img_dir, val_lbl_dir)

print(f"\nSplit completed:")
print(f"Train: {len(train_images)} images")
print(f"Val: {len(val_images)} images")
