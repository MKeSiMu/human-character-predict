from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


TARGET_SIZE = (200, 156)
THRESHOLD = 0.5


def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=TARGET_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def classifier(model, img_path):
    processed_image = load_and_preprocess_image(img_path)

    prediction = model.predict(processed_image)

    predicted_prob = prediction[0][0] * 100

    predicted_label = (
        f"'Savory': {100 - predicted_prob:.2f}% & 'Unsavory': {predicted_prob:.2f}%"
    )

    return predicted_label
