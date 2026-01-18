from data_loader import load_data
from model import build_model
from config import EPOCHS, MODEL_PATH

model = build_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

train_ds, val_ds = load_data()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

model.save(MODEL_PATH)
