import cv2
import numpy as np


def load_models():
    face_model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
    face_proto_path = 'deploy.prototxt.txt'
    age_model_path = 'age_net.caffemodel'
    age_proto_path = 'age_deploy.prototxt'
    gender_model_path = 'gender_net.caffemodel'
    gender_proto_path = 'gender_deploy.prototxt'

    face_net = cv2.dnn.readNetFromCaffe(face_proto_path, face_model_path)
    age_net = cv2.dnn.readNetFromCaffe(age_proto_path, age_model_path)
    gender_net = cv2.dnn.readNetFromCaffe(gender_proto_path, gender_model_path)

    return face_net, age_net, gender_net


def predict_age_and_gender(face, age_net, gender_net):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = 'Male' if gender_preds[0].argmax() == 0 else 'Female'

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'][age_preds[0].argmax()]

    return age, gender


def annotate_video(video_source, face_net, age_net, gender_net, use_webcam=False):
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_source)

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

            face_net.setInput(blob)
            detections = face_net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.6:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    if startX >= 0 and startY >= 0 and endX <= w and endY <= h:
                        face = frame[startY:endY, startX:endX]
                        age, gender = predict_age_and_gender(face, age_net, gender_net)

                        label = f"{gender}, {age}"
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key & 0xFF in [ord('q'), 27]:
            break
        elif key & 0xFF == ord(' '):
            paused = not paused

        if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def classify_image(image_path, face_net, age_net, gender_net):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            age, gender = predict_age_and_gender(face, age_net, gender_net)

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"{gender}, {age}"
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    face_net, age_net, gender_net = load_models()

    choice = input(
        "Do you want to use the webcam, classify a video, or classify an image? (webcam/video/image): ").strip().lower()

    print_statement = "Check the popup window for the results."

    if choice == 'webcam':
        print(print_statement)
        annotate_video(None, face_net, age_net, gender_net, use_webcam=True)
    elif choice == 'video':
        video_path = input("Enter the path to the video file: ")
        print(print_statement)
        annotate_video(video_path, face_net, age_net, gender_net)
    elif choice == 'image':
        image_path = input("Enter the path to the image file: ")
        print(print_statement)
        classify_image(image_path, face_net, age_net, gender_net)
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
