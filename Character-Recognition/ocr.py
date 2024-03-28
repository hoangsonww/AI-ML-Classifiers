import cv2
import pytesseract

# Configure the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def perform_ocr(image_path):
    # Load the image with OpenCV
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ocr_result = pytesseract.image_to_string(gray_image)
    annotated_image = annotate_image(image, ocr_result)

    cv2.imshow('Original Image', image)
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ocr_result


def annotate_image(image, ocr_result):
    boxes = pytesseract.image_to_boxes(image)
    annotated_image = image.copy()
    h, w = image.shape[:2]

    for box in boxes.splitlines():
        b = box.split(' ')
        char = b[0]
        x, y, width, height = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
        annotated_image = cv2.rectangle(annotated_image, (x, y), (width, height), (0, 255, 0), 2)
        cv2.putText(annotated_image, char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return annotated_image


def main():
    image_path = input("Enter the path to your image: ")
    detected_text = perform_ocr(image_path)
    print("Detected text:", detected_text)


if __name__ == "__main__":
    main()
