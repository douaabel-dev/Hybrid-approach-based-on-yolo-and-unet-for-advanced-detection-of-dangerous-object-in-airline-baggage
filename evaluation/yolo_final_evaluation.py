from ultralytics import YOLO

# 1 Load the  YOLO model
model = YOLO("best.pt")

# 2. Evaluate the model on the entire test set
metrics = model.val(
    data="data.yaml",
    split="test",
    verbose=False
)
    
# 3evaluation metrics
print("\n===== TEST SET METRICS =====")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")