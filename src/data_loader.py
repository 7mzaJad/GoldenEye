import tensorflow as tf
from config import IMG_SIZE, BATCH_SIZE, RAW_DATA_DIR

def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        RAW_DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        RAW_DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    return train_ds, val_ds
