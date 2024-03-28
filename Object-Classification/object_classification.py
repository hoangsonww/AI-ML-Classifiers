import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def classify_image(image_path):
    # Load the model
    model = MobileNetV2(weights='imagenet')

    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array)
    results = decode_predictions(predictions, top=1)[0]

    return results, image

def annotate_image(image, results):
    draw = ImageDraw.Draw(image)
    for i, (id, label, prob) in enumerate(results):
        text = f"{label} ({prob * 100:.2f}%)"
        draw.text((10, 10 + i * 30), text, fill="red")

    return image


if __name__ == "__main__":
    image_path = 'OIP.jpg'
    results, image = classify_image(image_path)
    annotated_image = annotate_image(image, results)

    print("Predictions:")
    for i, (id, label, prob) in enumerate(results):
        print(f"{i + 1}: {label} ({prob * 100:.2f}%)")

    annotated_image.show()
