from tensorflow.keras.models import load_model
from data_loader import load_data
from config import MODEL_PATH

model = load_model(MODEL_PATH)
_, val_ds = load_data()

loss, acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {acc:.2f}")
