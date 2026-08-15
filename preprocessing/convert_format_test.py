import json
import os


coco_json_path = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\annotations_json\test\test.json"
images_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\test\images"
yolo_labels_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\test\labels"

os.makedirs(yolo_labels_dir, exist_ok=True)


with open(coco_json_path) as f:
    coco = json.load(f)


image_id_to_name = {img['id']: img['file_name'] for img in coco['images']}


for ann in coco['annotations']:
    image_id = ann['image_id']
    file_name = image_id_to_name[image_id]


    bbox = ann['bbox']
    x, y, w, h = bbox

    img_info = next(img for img in coco['images'] if img['id'] == image_id)
    img_w, img_h = img_info['width'], img_info['height']

    x_center = (x + w/2) / img_w
    y_center = (y + h/2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # vlass id
    class_id = ann['category_id'] - 1  


    label_file = os.path.join(yolo_labels_dir, os.path.splitext(file_name)[0] + ".txt")

    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")





