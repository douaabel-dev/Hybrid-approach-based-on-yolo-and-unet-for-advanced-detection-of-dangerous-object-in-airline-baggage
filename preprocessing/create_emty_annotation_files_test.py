import os

images_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\test\images"
labels_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\test\labels"

for img_file in os.listdir(images_dir):
    if img_file.endswith((".jpg", ".png")):
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_file)
        if not os.path.exists(txt_path):
            open(txt_path, "w").close()  
