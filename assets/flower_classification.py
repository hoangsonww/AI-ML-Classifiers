import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def load_model():
    return MobileNetV2(weights='imagenet')


def classify_image(model, image):
    image_resized = image.resize((224, 224))
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    return decoded_predictions


def annotate_image(image, predictions):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        print("Arial font not found, using default font.")
        font = ImageFont.load_default()

    text_y = 10

    for i, (id, label, prob) in enumerate(predictions):
        text = f"{label} ({prob * 100:.2f}%)"
        draw.text((10, text_y), text, fill="red", font=font)

        if hasattr(font, 'getsize'):
            text_size = font.getsize(text)
        else:
            text_size = (len(text) * font.size, font.size)
        text_y += text_size[1] + 75

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

        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        predictions = classify_image(model, image)
        annotated_image = annotate_image(image.copy(), predictions)

        # Convert PIL image back to array and show
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
        print("Check the popup window for the results.")
        image = Image.open(image_path)
        predictions = classify_image(model, image)
        annotated_image = annotate_image(image, predictions)
        annotated_image.show()
    elif choice in ['video', 'webcam']:
        source = 0 if choice == 'webcam' else input("Enter the video path: ")
        print("Check the popup window for the results.")
        process_input(source, model)
    else:
        print("Invalid choice. Please enter 'image', 'video', or 'webcam'.")
