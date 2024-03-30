import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw
import numpy as np


def load_model():
    return MobileNetV2(weights='imagenet')


def classify_image(model, image_path):
    original_image = Image.open(image_path)
    resized_image = original_image.resize((224, 224))

    image_array = img_to_array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array)
    results = decode_predictions(predictions, top=1)[0]

    return results, original_image


def classify_frame(model, frame):
    # Convert the color space from BGR (OpenCV) to RGB (PIL)
    color_corrected_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_corrected_frame)
    resized_image = pil_image.resize((224, 224))

    image_array = img_to_array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array)
    results = decode_predictions(predictions, top=1)[0]

    return results, pil_image


def annotate_image(image, results):
    draw = ImageDraw.Draw(image)
    for i, (id, label, prob) in enumerate(results):
        text = f"{label} ({prob * 100:.2f}%)"
        draw.text((10, 10 + i * 30), text, fill="red")

    return image


def process_video(model, video_source):
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results, _ = classify_frame(model, frame)

        for id, label, prob in results:
            cv2.putText(frame, f"{label} ({prob * 100:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) == 27:  # ESC key
            break
        if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:  # Check if the window is closed
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("You may see some errors due to font issues. It is totally OK and can be ignored.")

    model = load_model()

    choice = input("Choose 'image', 'video', or 'webcam': ").lower()
    if choice == 'image':
        image_path = input("Enter the image path: ")

        print("Check the popup window for the results.")

        results, image = classify_image(model, image_path)
        annotated_image = annotate_image(image, results)
        annotated_image.show()

        print("Predictions:")
        for i, (id, label, prob) in enumerate(results):
            print(f"{i + 1}: {label} ({prob * 100:.2f}%)")
    elif choice in ['video', 'webcam']:
        video_source = 0 if choice == 'webcam' else input("Enter the video path: ")
        process_video(model, video_source)


if __name__ == "__main__":
    main()
