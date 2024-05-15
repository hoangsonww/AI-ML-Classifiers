import cv2
import pytesseract

# Configure the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def perform_ocr_on_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ocr_result = pytesseract.image_to_string(gray_image)
    annotated_image = annotate_image(image)

    return ocr_result, annotated_image


def annotate_image(image):
    boxes = pytesseract.image_to_boxes(image)
    annotated_image = image.copy()
    h, _ = image.shape[:2]

    for box in boxes.splitlines():
        b = box.split(' ')
        char = b[0]
        x, y, width, height = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
        cv2.rectangle(annotated_image, (x, y), (width, height), (0, 255, 0), 2)
        cv2.putText(annotated_image, char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return annotated_image


def process_input(source):
    if source == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, annotated_image = perform_ocr_on_image(frame)

        cv2.imshow('Annotated Image', annotated_image)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:  # ESC or Q key to exit
            break
        if cv2.getWindowProperty("Annotated Image", cv2.WND_PROP_VISIBLE) < 1:  # Check if the window is closed
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    choice = input("Enter 'image', 'video', or 'webcam': ").lower()

    if choice == 'image':
        image_path = input("Enter the image path: ")
        print("Check the popup window for detailed results.")
        image = cv2.imread(image_path)
        _, annotated_image = perform_ocr_on_image(image)
        cv2.imshow('Annotated Image', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif choice in ['video', 'webcam']:
        source = 0 if choice == 'webcam' else input("Enter the video path: ")
        print("Check the popup window for detailed results.")
        process_input(source)
    else:
        print("Invalid choice. Please enter 'image', 'video', or 'webcam'.")


if __name__ == "__main__":
    main()
