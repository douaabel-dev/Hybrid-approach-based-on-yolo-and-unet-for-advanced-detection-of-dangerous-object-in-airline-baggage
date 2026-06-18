import os
import random
import shutil

images_dir = "C:\\Users\\Douaa Belhadjadji\\Desktop\\Douaa_Study\\M2_isi_2025\\PROJECT\\Datasets\\PIDray\\PIDray\\unet_dataset\\images"
masks_dir = "C:\\Users\\Douaa Belhadjadji\\Desktop\\Douaa_Study\\M2_isi_2025\\PROJECT\\Datasets\\PIDray\\PIDray\\unet_dataset\\masks"

test_images_dir = "C:\\Users\\Douaa Belhadjadji\\Desktop\\Douaa_Study\\M2_isi_2025\\PROJECT\\Datasets\\PIDray\\PIDray\\unet_dataset\\test\\images"
test_masks_dir  = "C:\\Users\\Douaa Belhadjadji\\Desktop\\Douaa_Study\\M2_isi_2025\\PROJECT\\Datasets\\PIDray\\PIDray\\unet_dataset\\test\\masks"

os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)

images = os.listdir(images_dir)
test_samples = random.sample(images, int(0.15 * len(images)))  # 15%

for img in test_samples:
    shutil.move(os.path.join(images_dir, img), os.path.join(test_images_dir, img))
    shutil.move(os.path.join(masks_dir, img), os.path.join(test_masks_dir, img))

print("Test split done")