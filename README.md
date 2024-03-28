# AI Classifiers

This repository contains Python scripts for vehicle classification and object classification using pre-trained deep learning models. The vehicle classification logic uses the YOLOv3 model for vehicle detection and classification, while the object classification logic uses a pre-trained model for object classification. These scripts can be used to classify vehicles in videos and objects in images, respectively.

This repository contains two directories: one for vehicle classification logic and another for object classification logic, namely `vehicle_classification` and `object_classification`. Each directory contains the necessary files and instructions to run the respective classification logic.

## Table of Contents

- [Vehicle Classification](#vehicle-classification)
  - [Files Included](#files-included)
  - [Getting Started](#getting-started)
  - [Output](#output)
  - [Contact Information](#contact-information)
  - [License](#license)
- [Flower Classification](#flower-classification)
  - [Files Included](#files-included-1)
  - [Getting Started](#getting-started-1)
  - [Output](#output-1)
- [Object Classification](#object-classification)
  - [Files Included](#files-included-1)
  - [Getting Started](#getting-started-1)
  - [Output](#output-1)
- [Animal Classification](#animal-classification)
  - [Files Included](#files-included-2)
  - [Getting Started](#getting-started-2)
  - [Output](#output-2)

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

## Flower Classification

### Files Included
- `flower_classification.py`: Python script for flower classification.
- `daisy.jpg`: Sample JPEG image for flower classification (Daisy).
- `marigold.jpg`: Sample JPEG image for flower classification (Marigold).

### Getting Started

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

## Contact Information

For any questions or issues, please contact:
- Name: Son Nguyen
- Email: [info@movie-verse.com](mailto:info@movie-verse.com)

---
