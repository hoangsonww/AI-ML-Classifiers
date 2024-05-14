import cv2
import numpy as np


def load_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, classes, output_layers


def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)


def load_model_and_classes():
    model, classes, output_layers = load_model()
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return model, classes, output_layers, colors


def process_webcam(cap, model, classes, output_layers, colors):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame, model, classes, output_layers, colors)
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
            break


def process_image(source, model, classes, output_layers, colors):
    frame = cv2.imread(source)
    if frame is None:
        print(f"Failed to load image at {source}")
        return
    process_frame(frame, model, classes, output_layers, colors)
    cv2.waitKey(0)


def process_video(cap, model, classes, output_layers, colors):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame, model, classes, output_layers, colors)
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
            break


def process_input(source):
    model, classes, output_layers, colors = load_model_and_classes()

    if source == 'webcam':
        cap = cv2.VideoCapture(0)
        process_webcam(cap, model, classes, output_layers, colors)
        cap.release()
    elif source.endswith(('.png', '.jpg', '.jpeg')):
        process_image(source, model, classes, output_layers, colors)
    else:
        cap = cv2.VideoCapture(source)
        process_video(cap, model, classes, output_layers, colors)
        cap.release()

    cv2.destroyAllWindows()


def process_frame(frame, model, classes, output_layers, colors):
    height, width, channels = frame.shape
    outputs = detect_objects(frame, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, frame)
    cv2.imshow("Video", frame)


if __name__ == "__main__":
    choice = input("Enter 'image', 'video', or 'webcam': ").lower()

    print_statement = "Check the popup window for the results."

    if choice == 'image':
        image_path = input("Enter the image path: ")
        print(print_statement)
        process_input(image_path)
    elif choice == 'video':
        video_path = input("Enter the video path: ")
        print(print_statement)
        process_input(video_path)
    elif choice == 'webcam':
        print(print_statement)
        process_input('webcam')
