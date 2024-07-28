import numpy as np
import tensorflow as tf
from PIL import Image

TARGET_SIZE = (156, 200)


def load_and_preprocess_image(img_path, target_size):
    # Load image
    with Image.open(img_path).convert("RGB") as image:
        # Resize image to the target size expected by the model
        image = image.resize(target_size)
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        # Add batch dimension
        processed_image = np.expand_dims(image_array, axis=0).astype(np.float32)
    return processed_image


def classifier(tflite_model_path, img_path):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape and size
    input_shape = input_details[0]["shape"]
    target_size = (input_shape[2], input_shape[1])  # Adjusted to match the expected shape

    # Preprocess the image to match the input shape
    processed_image = load_and_preprocess_image(img_path, target_size)

    # Set the tensor to point to the input data
    interpreter.set_tensor(input_details[0]["index"], processed_image)

    # Run the inference
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])

    predicted_prob = prediction[0][0] * 100

    predicted_label = (
        f"'Savory': {100 - predicted_prob:.2f}% & 'Unsavory': {predicted_prob:.2f}%"
    )

    return predicted_label
