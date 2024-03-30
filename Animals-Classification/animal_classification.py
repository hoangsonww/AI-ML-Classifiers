import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def load_model():
    return MobileNetV2(weights='imagenet')


def classify_image(model, image):
    resized_image = image.resize((224, 224))
    image_array = img_to_array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    return decoded_predictions


def annotate_image(image, predictions):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_y = 10

    for i, (id, label, prob) in enumerate(predictions):
        text = f"{label} ({prob * 100:.2f}%)"
        draw.text((10, text_y), text, fill="red", font=font)
        text_y += 20

    return image


def process_input(source, model):
    if source == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        predictions = classify_image(model, image)
        annotated_image = annotate_image(image.copy(), predictions)

        cv2_image = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", cv2_image)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:  # ESC or Q key to exit
            break
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:  # Check if the window is closed
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("You may see some errors due to font issues. It is totally OK and can be ignored.")

    model = load_model()

    choice = input("Enter 'image', 'video', or 'webcam': ").lower()
    if choice == 'image':
        image_path = input("Enter the image path: ")
        image = Image.open(image_path)
        predictions = classify_image(model, image)
        annotated_image = annotate_image(image, predictions)
        annotated_image.show()
    elif choice in ['video', 'webcam']:
        source = 0 if choice == 'webcam' else input("Enter the video path: ")
        process_input(source, model)
