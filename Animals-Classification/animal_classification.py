import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def classify_animal(image_path):
    # Load the MobileNetV2 model pre-trained on ImageNet
    model = MobileNetV2(weights='imagenet')

    # Load and preprocess the input image
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Perform classification
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    return decoded_predictions, image

def annotate_image(image, predictions):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_y = 10

    for i, (id, label, prob) in enumerate(predictions):
        text = f"{label} ({prob * 100:.2f}%)"
        draw.text((10, text_y), text, fill="red", font=font)
        text_y += 20

    return image

if __name__ == "__main__":
    image_path = 'ox.jpg'  # Path to the animal image you want to classify
    predictions, image = classify_animal(image_path)

    annotated_image = annotate_image(image, predictions)

    annotated_image.show()
