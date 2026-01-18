import tensorflow as tf
import cv2
from config import IMG_SIZE, MODEL_PATH

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = img.reshape(1, *IMG_SIZE, 3)

    pred = model.predict(img)
    return pred.argmax()
