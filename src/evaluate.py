from data_loader import load_data
from config import MODEL_PATH

import numpy as np
from ultralytics import YOLO

model = YOLO("/Users/hamza/Library/CloudStorage/OneDrive-AL-HussienbinAbdullahTechnicalUniversity/GoldenEye/models/best.pt")
_, val_ds = load_data()

loss, acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {acc:.2f}")

