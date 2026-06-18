
from ultralytics import YOLO
import random
import cv2
from pathlib import Path

# =========================================================
# 1. LOAD YOLO MODEL
# =========================================================
model = YOLO("best.pt")

# =========================================================
# 2. TEST IMAGES PATH
# =========================================================
test_images_root = r"C:\Users\Douaa Belhadjadji\Desktop\Douaa_Study\M2_isi_2025\PROJECT\Datasets\PIDray\PIDray\PIDray_YOLO\test\images"

# =========================================================
# 3. EVALUATE MODEL ON TEST SET
# =========================================================
metrics = model.val(
    data="data.yaml",
    split="test",
    verbose=False
)

print("\n========== TEST METRICS ==========")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print("==================================\n")


# =========================================================
# 4. RANDOMLY SAMPLE AND SAVE TEST IMAGES
# =========================================================
def sample_and_save_test_images(
    image_root,
    num_samples=1000,
    output_dir="sampled_test_images"
):

    # -----------------------------------------------------
    # Create output directories
    # -----------------------------------------------------
    output_path = Path(output_dir)

    original_dir = output_path / "original"
    annotated_dir = output_path / "annotated"

    original_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Find all images recursively
    # -----------------------------------------------------
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]

    all_images = []

    for ext in image_extensions:
        all_images.extend(Path(image_root).rglob(ext))

    print(f"Found {len(all_images)} images in test set")

    # -----------------------------------------------------
    # Check if images exist
    # -----------------------------------------------------
    if len(all_images) == 0:
        print("ERROR: No images found!")
        return

    # -----------------------------------------------------
    # Random sampling
    # -----------------------------------------------------
    num_to_sample = min(num_samples, len(all_images))

    sampled_images = random.sample(all_images, num_to_sample)

    print(f"Randomly selected {num_to_sample} images")

    # -----------------------------------------------------
    # Process images
    # -----------------------------------------------------
    for idx, img_path in enumerate(sampled_images):

        # Read image
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        # YOLO prediction
        results = model(img, verbose=False)

        # -------------------------------------------------
        # Save original image
        # -------------------------------------------------
        original_save_path = original_dir / img_path.name

        cv2.imwrite(str(original_save_path), img)

        # -------------------------------------------------
        # Save annotated image
        # -------------------------------------------------
        annotated_img = results[0].plot()

        annotated_save_path = annotated_dir / img_path.name

        cv2.imwrite(str(annotated_save_path), annotated_img)

        # -------------------------------------------------
        # Progress display
        # -------------------------------------------------
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{num_to_sample} images")

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------
    print("\n==================================")
    print("Sampling completed successfully!")
    print(f"Original images saved in : {original_dir}")
    print(f"Annotated images saved in: {annotated_dir}")
    print("==================================")


# =========================================================
# 5. RUN SAMPLING
# =========================================================
sample_and_save_test_images(
    image_root=test_images_root,
    num_samples=1000,
    output_dir="sampled_test_images"
)
