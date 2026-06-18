import os

images_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\images\train"
labels_dir = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\labels\train"

for img_file in os.listdir(images_dir):
    if img_file.endswith((".jpg", ".png")):
#create corresponding label file name
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_file)
        if not os.path.exists(txt_path):
            open(txt_path, "w").close()  
