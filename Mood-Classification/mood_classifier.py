import cv2
from deepface import DeepFace


def analyze_and_show(image, show_results=True):
    try:
        analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)

        text_y = 20

        for result in analysis:
            if 'region' in result:
                region = result['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if 'emotion' in result:
                emotions = result['emotion']
                dominant_emotion = result['dominant_emotion']

                if show_results:
                    print(f"Dominant emotion: {dominant_emotion}")
                    print("See more detailed stats on the popup window.")

                for emotion, prob in emotions.items():
                    cv2.putText(image, f"{emotion}: {prob:.2f}%", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
                    text_y += 20

        cv2.imshow('Mood Detector', image)

    except Exception as e:
        print(f"Error in emotion detection: {e}")


def process_input(source):
    cap = cv2.VideoCapture(0 if source == 'webcam' else source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        show_results = True
        analyze_and_show(frame, show_results)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

        if cv2.getWindowProperty('Mood Detector', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = input("Enter 'image', 'video', or 'webcam': ").lower()

    if choice == 'image':
        image_path = input("Enter the image path: ")
        print("Check the popup window for the results.")
        image = cv2.imread(image_path)
        analyze_and_show(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif choice in ['video', 'webcam']:
        source = 'webcam' if choice == 'webcam' else input("Enter the video path: ")
        print("Check the popup window for the results.")
        process_input(source)
    else:
        print("Invalid choice. Please enter 'image', 'video', or 'webcam'.")
