# AI Classifiers

Created by [Son Nguyen](https://github.com/hoangsonww) in 2024, this repository contains Python scripts for vehicle classification and object classification using pre-trained deep learning models. The vehicle classification logic uses the YOLOv3 model for vehicle detection and classification, while the object classification logic uses a pre-trained model for object classification. These scripts can be used to classify vehicles in videos and objects in images, respectively.

This repository contains six sub-directories: one for vehicle classification logic, one for human face classification logic, one for flower classification logic, one for object classification logic, one for character classification logic, and one for animal classification logic, namely `Vehicle-Classification`, `Human-Face-Classification`, `Flowers-Classification`, `Object-Classification`, `Character-Recognition`, and `Animal-Classification`. Refer to the information below for details on each classifier.

## Table of Contents

- [Vehicle Classification](#vehicle-classification)
  - [Files Included](#files-included)
  - [Getting Started](#getting-started)
  - [Output](#output)
  - [License](#license)
- [Human Face Classification](#face-classification)
  - [Files Included](#files-included-1)
  - [Getting Started](#getting-started-1)
  - [Output](#output-1)
- [Flower Classification](#flower-classification)
  - [Files Included](#files-included-2)
  - [Getting Started](#getting-started-2)
  - [Output](#output-2)
- [Object Classification](#object-classification)
  - [Files Included](#files-included-3)
  - [Getting Started](#getting-started-3)
  - [Output](#output-3)
- [Character Classification (OCR)](#character-classification)
  - [Files Included](#files-included-4)
  - [Getting Started](#getting-started-4)
  - [Output](#output-4)
- [Animal Classification](#animal-classification)
  - [Files Included](#files-included-5)
  - [Getting Started](#getting-started-5)
  - [Output](#output-5)
- [Contact Information](#contact-information)

---

## Vehicle Classification

### Files Included
- `coco.names`: Class names used for vehicle detection.
- `traffic.mp4`: Sample video for vehicle detection.
- `yolov3.cfg`: YOLOv3 model configuration file.
- `yolov3.weights`: Pre-trained YOLOv3 model weights.
- `vehicle_detection.py`: Python script for vehicle detection and classification.

### Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hoangsonww/AI-Classification.git
   cd AI-Classification/Vehicle-Classification
   ```

2. **Download Model Weights**
   Download the pre-trained YOLOv3 model weights (`yolov3.weights`) from the official YOLO website or another trusted source and place it in the `Vehicle-Classification` directory.

3. **Install Dependencies**
   Install the required Python dependencies.
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Vehicle Detection**
   Replace `<video_path>` in the `vehicle_detection.py` script with the path to your video file (`traffic.mp4` or another video).
   ```bash
   python vehicle_detection.py
   ```
   
The script will then process the video frame by frame, detect vehicles, and classify them based on the detected classes. The output video will be saved as `output.avi` in the `vehicle_classification` directory.

Feel free to change the video path, output video name, and other parameters in the script to suit your needs.

### Output

The output video will display the detected vehicles along with their class labels. The class labels are based on the COCO dataset, which includes various classes such as car, truck, bus, motorcycle, and bicycle.

Example output:

<p align="center">
  <img src="Vehicle-Classification/vehicle-classi.png" alt="Vehicle Classification Output" width="100%">
</p>

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Face Classification

### Files Included
- `deploy.prototxt`: Model configuration file for the face detector.
- `res10_300x300_ssd_iter_140000.caffemodel`: Pre-trained model weights for face detection.
- `age_deploy.prototxt`: Model configuration file for age prediction.
- `age_net.caffemodel`: Pre-trained model weights for age prediction.
- `gender_deploy.prototxt`: Model configuration file for gender prediction.
- `gender_net.caffemodel`: Pre-trained model weights for gender prediction.
- `face_classification.py`: Python script for face detection, age, and gender classification.
- `woman-30.mp4`: Sample video for face classification

### Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hoangsonww/AI-Classification.git
   cd AI-Classification/Face-Classification
   ```

2. **Download Model Weights**: Ensure you have the model weights (`res10_300x300_ssd_iter_140000.caffemodel`, `age_net.caffemodel`, `gender_net.caffemodel`) in the `Human-Face-Classification` directory.

3. **Install Dependencies**: Install the required Python dependencies.

    ```
    pip install -r requirements.txt
    ```

4. **Run Face Classification**: Execute the face_classification.py script.

    ```
    python face_classification.py
    ```
    The script will now ask for the video file path. You can provide the path to the sample video file (`woman-30.mp4`) or another video file.

    ```
    woman-40.mp4
    ```
   
    The script will then process the video file, detect faces, predict ages and genders, and annotate the video with this information.

### Output

The output will be a video displaying the detected faces along with their estimated age and gender.

Example output:

<p align="center">
  <img src="Human-Face-Classification/face-classi.png" alt="Face Classification Output" width="100%">
</p>

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Character Classification

### Files Included
- `ocr.py`: Python script for character classification.
- `OIP.jpg`: Sample JPEG image for character classification.

### Getting Started

1. **Clone the Repository**
   ```bash
   git clone
   cd AI-Classification/Character-Recognition
    ```
2. **Install the required Python dependencies.**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Character Classification**
    ```bash
    python ocr.py
    ```
    The script will then process the image, detect characters, and classify them based on the detected classes.
    It will return 2 images: one is the original image and the other is the image with the detected characters.

### Output

The output will display the class labels of the characters detected in the image along with the confidence scores.

<p align="center">
  <img src="Character-Recognition/character-classi.png" alt="Character Classification Output" width="350">
</p>

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Flower Classification

### Files Included
- `flower_classification.py`: Python script for flower classification.
- `daisy.jpg`: Sample JPEG image for flower classification (Daisy).
- `marigold.jpg`: Sample JPEG image for flower classification (Marigold).

### Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hoangsonww/AI-Classification.git
   cd AI-Classification/Flowers-Classification
   ```

2. **Install Dependencies**
   Install the required Python dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Object Classification**
   Replace `<image_path>` in the `flower_classification.py` script with the path to your image file (`objects.jpg`, `objects.png`, or another image).
   ```bash
   python flower_classification.py
   ```

### Output

The output will display the class label of the flower detected in the image along with the confidence score.

Example output: Here are the sample image of Daisy flowers.

<p align="center">
  <img src="Flowers-Classification/flower-classi.png" alt="Flower Classification Output" width="350">
</p>

---

## Object Classification

### Files Included
- `object_classification.py`: Python script for object classification.
- `objects.jpg`: Sample JPEG image for object classification.
- `objects.png`: Sample PNG image for object classification.

### Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hoangsonww/AI-Classification.git
   cd AI-Classification/object_classification
   ```

2. **Install Dependencies**
   Install the required Python dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Object Classification**
   Replace `<image_path>` in the `object_classification.py` script with the path to your image file (`objects.jpg`, `objects.png`, or another image).
   ```bash
   python object_classification.py
   ```
   
The script will then classify the objects in the image and display the class labels along with the confidence scores.

Feel free to change the image path and other parameters in the script to suit your needs.

### Output

The output will display the class labels of the objects detected in the image along with the confidence scores.

<p align="center">
  <img src="Object-Classification/object-classi.png" alt="Object Classification Output" width="350">
</p>

---

## Animal Classification

### Files Included
- `animal_classification.py`: Python script for animal classification.
- `cow.jpg`: Sample JPEG image for animal classification (Cow).
- `ox.jpg`: Sample JPEG image for animal classification (Ox).

### Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hoangsonww/AI-Classification.git
   cd AI-Classification/Animal-Classification
   ```

2. **Install Dependencies**
   Install the required Python dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Object Classification**
   Replace `<image_path>` in the `animal_classification.py` script with the path to your image file (`objects.jpg`, `objects.png`, or another image).
   ```bash
   python animal_classification.py
   ```

### Output

The output will display the class labels of the animals detected in the image along with the confidence scores.

<p align="center">
  <img src="Animals-Classification/animal-classi.png" alt="Animal Classification Output" width="350">
</p>

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact Information

For any questions or issues, please contact:
- Name: Son Nguyen
- Email: [info@movie-verse.com](mailto:info@movie-verse.com)

---