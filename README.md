# AI Multitask Classifiers: From Objects to Emotions

Created by [Son Nguyen](https://github.com/hoangsonww) in 2024, this repository contains Python scripts for various AI-powered classifiers. These classifiers can be used for object detection, face detection, character recognition, and more. The classifiers are built using popular deep learning frameworks such as `OpenCV`, `TensorFlow`, and `PyTorch`.

This repository contains **9** subdirectories for the **9** classifiers:

| Classifier                | Subdirectory Name           |
|:--------------------------|:----------------------------|
| Vehicle Classification    | `Vehicle-Classification`    |
| Human Face Classification | `Human-Face-Classification` |
| Mood Classification       | `Mood-Classification`       |
| Flower Classification     | `Flowers-Classification`    |
| Object Classification     | `Object-Classification`     |
| Character Recognition     | `Character-Recognition`     |
| Animal Classification     | `Animals-Classification`    |
| Speech Recognition        | `Speech-Recognition`        |
| Sentiment Analysis        | `Sentiment-Analysis`        |

Detailed information about each classifier can be found below.

What's even more interesting is that all these classifiers can use your **webcam** for live testing, **video** files, or **image** files!

Please read this README file carefully to understand how to use each classifier and how to run the main script to choose and run any of the classifiers. Happy classifying! 🚀

## Table of Contents

- [Before You Begin](#before-you-begin)
- [Main Script - Entry Point](#main-script)
- [Flask Web App](#flask-web-app)
- [Vehicle Classification](#vehicle-classification)
  - [Files Included](#files-included)
  - [Getting Started](#getting-started)
  - [Output](#output)
- [Human Face Classification](#face-classification)
  - [Files Included](#files-included-1)
  - [Getting Started](#getting-started-1)
  - [Output](#output-1)
- [Mood Classification](#mood-classification)
  - [Files Included](#files-included-2)
  - [Getting Started](#getting-started-2)
  - [Output](#output-2)
- [Flower Classification](#flower-classification)
  - [Files Included](#files-included-3)
  - [Getting Started](#getting-started-3)
  - [Output](#output-3)
- [Object Classification](#object-classification)
  - [Files Included](#files-included-4)
  - [Getting Started](#getting-started-4)
  - [Output](#output-4)
- [Character Classification (OCR)](#character-classification)
  - [Files Included](#files-included-5)
  - [Getting Started](#getting-started-5)
  - [Output](#output-5)
- [Animal Classification](#animal-classification)
  - [Files Included](#files-included-6)
  - [Getting Started](#getting-started-6)
  - [Output](#output-6)
- [Speech Recognition](#speech-recognition)
  - [Files Included](#files-included-7)
  - [Getting Started](#getting-started-7)
  - [Output](#output-7)
- [Special: Self-Trained Sentiment Classifier](#special-self-trained-sentiment-classifier)
  - [Files Included](#files-included-8)
  - [Getting Started](#getting-started-8)
  - [Output](#output-8)
- [Contact Information](#contact-information)
- [Future Work](#future-work)
- [License](#license)

---

## Before You Begin

Before you begin, ensure you have the following installed on your machine (run `pip install <requirement_name>` for each dependency or `pip install -r requirements.txt` to install all the required packages):

- Python 3.12 or higher (download from the [official Python website](https://www.python.org/))
- OpenCV
- TensorFlow
- PyTorch
- NumPy
- Matplotlib
- Tesseract OCR
- Pytesseract
- SpeechRecognition
- MoviePy
- PyDub
- PyAudio
- Scikit-learn
- Pandas
- NLTK
- tqdm
- Joblib
- YoloV3
- jQuery
- Git LFS (for downloading large model weights files)
- A webcam (if you want to use live testing)
- A microphone (if you want to use speech recognition)
- A video file or image file for testing the classifiers
- A stable internet connection (for downloading model weights and dependencies)
- A working speaker or headphones (for speech recognition)
- Additionally, if you would like to train the sentiment classifier, you will need:
  - A machine with sufficient computational resources
  - The large training data files (e.g. `training.1600000.processed.noemoticon.csv`) or the small dataset generated from it (`small_dataset.csv`)
- And if you would like to use the website version of this app, you will also need to install `Flask` and `Flask-SocketIO` as well.

It is also **recommended** to use a virtual environment to use these classifiers. You can create a virtual environment using `venv` or `conda`:

```bash
python -m venv env
source env/bin/activate
```

**Note: If you are unable to use Git LFS, you can download the necessary files from my Google Drive:**
- [yolov3.weights](https://drive.google.com/file/d/13Tf2j9blTPUnITUuUtvWNFxUImSz1TKe/view?usp=sharing)
- [training.1600000.processed.noemoticon.csv](https://drive.google.com/file/d/1VHdnu9pNPIp2Gu6y0mOW5GB4azaEPu8I/view?usp=sharing)
- [test.csv](https://drive.google.com/file/d/1NO46ZaztULg-oHImjf5E_Qwb3ht-GtMl/view?usp=sharing)
- [train.csv](https://drive.google.com/file/d/1gs1pjTGsDEgzXd6o-9fvroM0ezmP45pn/view?usp=sharing)
- [small_dataset.csv](https://drive.google.com/file/d/13r2bPO_dOPITn4UQh0_0wOWRfERD0UMj/view?usp=sharing)

Please feel free to let me know if you encounter any problems with any of the files, or with getting started with the project!

---

## Main Script

If you prefer not to navigate through the subdirectories, you can run the main script `main.py` to choose and run any of the classifiers. The main script will ask you to choose a classifier from the list of available classifiers. You can then select a classifier and run it.

To run the main script, use the following command:

```bash
python main.py
```

The main script will display a list of available classifiers. Enter the number corresponding to the classifier you want to run. The script will then run the selected classifier.

To stop the script, press `Q`, `ESC`, or otherwise close the window.

Alternatively, you can also run the individual scripts in each subdirectory below to run the classifiers directly.

---

## Flask Web App

If you would like to use the interactive website version of this app, you can run the Flask web app. The web app allows you to use the classifiers through a web interface. You can choose a classifier and the app will run the selected classifier.

To run the Flask web app, use the following command:

```bash
python app.py
```

The web app will start running on `http://127.0.0.1:5000/`. Open this URL in your web browser to access the web app. You can then choose a classifier from the list of available classifiers and run it. A pop-up window will display the output of the classifier - so be sure to allow pop-ups in your browser.

Here is what it looks like:

<p align="center">
  <img src="assets/flask-web-app.png" alt="Flask Web App" width="100%">
</p>

Note that the app has also been deployed to Heroku [at this link](https://ai-multipurpose-classifier-b1655f2a20d4.herokuapp.com/). However, due to changes in Heroku's free tier regarding available Dynos (and I'm a broke college student), the app may not work as expected. If you encounter any issues, please run the app locally using the instructions above.

---

## Vehicle Classification

### Files Included

- `coco.names`: Class names used for vehicle detection.
- `traffic.mp4`: Sample video for vehicle detection.
- `india.jpg`: Sample image for vehicle detection.
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

4. **Install and Pull Git LFS**
   Install Git LFS by following the instructions on the [official Git LFS website](https://git-lfs.github.com/). Then, pull the model weights using Git LFS.

   ```bash
   git lfs install
   git lfs pull
   ```

   Alternatively, you can download the weights file from the [official YOLO website](https://pjreddie.com/darknet/yolo/) and place it in the `Vehicle-Classification` directory. However, using Git LFS is recommended.

   This is crucial for pulling the large model weights file `yolov3.weights`. Without Git LFS, the weights file may not be downloaded correctly and the script may not work as expected.

   If you still encounter problems with Git LFS, you can download the weights file from my Google Drive, which is publicly available [here](https://drive.google.com/file/d/13Tf2j9blTPUnITUuUtvWNFxUImSz1TKe/view?usp=sharing).

5. **Run Vehicle Detection**
   ```bash
   python vehicle_detection.py
   ```

You will then be asked to choose your input type (image, video, or webcam). Enter `image` to classify the vehicles in the sample video provided (`traffic.mp4`), or enter `video` to classify vehicles in a video file. You can also use your webcam for live testing.

All our classifiers will only stop when you press `Q`, `ESC`, or otherwise close the window.

### Output

The output video will display the detected vehicles along with their class labels. The class labels are based on the COCO dataset, which includes various classes such as car, truck, bus, motorcycle, and bicycle.

Example output:

<p align="center">
  <img src="Vehicle-Classification/vehicle-classi.png" alt="Vehicle Classification Output" width="100%">
</p>

---

## Face Classification

### Files Included

- `deploy.prototxt`: Model configuration file for the face detector.
- `res10_300x300_ssd_iter_140000.caffemodel`: Pre-trained model weights for face detection.
- `age_deploy.prototxt`: Model configuration file for age prediction.
- `age_net.caffemodel`: Pre-trained model weights for age prediction.
- `gender_deploy.prototxt`: Model configuration file for gender prediction.
- `gender_net.caffemodel`: Pre-trained model weights for gender prediction.
- `faces_classification.py`: Python script for face detection, age, and gender classification.
- `woman-30.mp4`: Sample video for face classification
- `man.jpg`: Sample image for face classification.

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

You will then be asked to choose your input type (image, video, or webcam). Enter `image` to classify the faces in the sample image provided (`woman-30.mp4`), or enter `video` to classify faces in a video file. You can also use your webcam for live testing.

All our classifiers will only stop when you press `Q`, `ESC`, or otherwise close the window.

### Output

The output will be a video displaying the detected faces along with their estimated age and gender.

Example output:

<p align="center">
  <img src="Human-Face-Classification/face-classi.png" alt="Face Classification Output" width="100%">
</p>

---

## Mood Classification

### Files Included

- `mood_classifier.py`: Python script for mood classification.
- `angry.mp4`: Sample video for mood classification (angry).
- `surprised.jpg`: Sample image for mood classification (surprised).

### Getting Started

1. **Clone the Repository**
   ```bash
   git clone
   cd AI-Classification/Mood-Classification
   ```
2. **Install Dependencies**
   Install the required Python dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Mood Classification**
   ```bash
   python mood_classifier.py
   ```

You will then be asked to choose your input type (image, video, or webcam). Enter `image` to classify the mood in the sample image provided (`surprised.jpg`), or enter `video` to classify the mood in a video file. You can also use your webcam for live testing.

The script will then display the detected mood in the image, video, or webcam stream and in the console.

All our classifiers will only stop when you press `Q`, `ESC`, or otherwise close the window.

### Output

The output will be displayed the detected mood in the image, video, or webcam stream and in the console.

Example output:

<p align="center">
  <img src="Mood-Classification/mood-classi.png" alt="Mood Classification Output" width="100%">
</p>

---

## Character Classification

### Files Included

- `ocr.py`: Python script for character classification.
- `OIP.jpg`: Sample JPEG image for character classification.
- `chars.jpg`: Sample JPEG image for character classification.
- `chars.mp4`: Sample video for character classification.
- `letters.mp4`: Sample video for character classification.

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
3. **Install Tessaract OCR**
   - For Windows: Download and install the Tesseract OCR executable from the [official Tesseract OCR website](https://github.com/UB-Mannheim/tesseract/wiki).
   - For Linux: Install Tesseract OCR using the package manager.
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - For macOS: Install Tesseract OCR using Homebrew.
     ```bash
     brew install tesseract
     ```
   - This is required for the OCR functionality to work. Also, when you install, note down the installation path of the Tesseract OCR executable. Replace the path in the `pytesseract.pytesseract.tesseract_cmd` variable in the `ocr.py` script with yours.
   - For example, if you installed Tesseract OCR in the default location on Windows, the path would be:
     ```python
     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
     ```
4. **Run Character Classification**

   ```bash
   python ocr.py
   ```

   You will then be asked to choose your input type (image, video, or webcam). Enter `image` to classify the characters in the sample image provided (`OIP.jpg`), or enter `video` to classify characters in a video file. You can also use your webcam for live testing.

   ```
   image
   ```

The script will then display the detected characters in the image, video, or webcam stream.

All our classifiers will only stop when you press `Q`, `ESC`, or otherwise close the window.

### Output

The output will display the class labels of the characters detected in the image along with the confidence scores.

Example output:

<p align="center">
  <img src="Character-Recognition/character-classi.png" alt="Character Classification Output" width="350">
</p>

---

## Flower Classification

### Files Included

- `flower_classification.py`: Python script for flower classification.
- `daisy.jpg`: Sample JPEG image for flower classification (Daisy).
- `marigold.jpg`: Sample JPEG image for flower classification (Marigold).
- `rose.mp4`: Sample video for flower classification (Rose).

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
   ```bash
   python flower_classification.py
   ```

You will then be asked to choose your input type (image, video, or webcam). Enter `image` to classify the flowers in the sample image provided (`daisy.jpg`), or enter `video` to classify flowers in a video file. You can also use your webcam for live testing.

All our classifiers will only stop when you press `Q`, `ESC`, or otherwise close the window.

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
- `balls.mp4`: Sample video for object classification.
- `OIP.jpg`: Sample image for object classification.

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
   ```bash
   python object_classification.py
   ```

You will then be asked to choose your input type (image, video, or webcam). Enter `image` to classify the objects in the sample image provided (`objects.jpg`), or enter `video` to classify objects in a video file. You can also use your webcam for live testing.

Feel free to change the paths and other parameters in the script to suit your needs.

**Note:** All our classifiers will only stop when you press `Q`, `ESC`, or otherwise close the window.

### Output

The output will display the class labels of the objects detected in the image along with the confidence scores. Or, if you choose to use your webcam, the output will display the class labels of the objects detected in the video stream. If you choose to use a video file, the output will be a video displaying the detected objects along with their class labels.

Example output:

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
   cd AI-Classification/Animals-Classification
   ```

2. **Install Dependencies**
   Install the required Python dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Object Classification**
   ```bash
   python animal_classification.py
   ```

The script will then ask you to choose your input type (image, video, or webcam). Enter `image` to classify the animals in the sample image provided (`cow.jpg`), or enter `video` to classify animals in a video file. You can also use your webcam for live testing.

All our classifiers will only stop when you press `Q`, `ESC`, or otherwise close the window.

### Output

The output will display the class labels of the animals detected in the image along with the confidence scores.

Example output:

<p align="center">
  <img src="Animals-Classification/animal-classi.png" alt="Animal Classification Output" width="350">
</p>

---

## Speech Recognition

### Files Included

- `speech_classifier.py`: Python script for speech recognition.
- `speech.mp4`: Sample video file for speech recognition in a video context.
- `temp_audio.wav`: Temp audio file (used by our AI) for speech recognition.

### Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/hoangsonww/AI-Classification.git
   cd AI-Classification/Speech-Recognition
   ```

2. **Install Dependencies**
   Install the required Python dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Speech Recognition**
   ```bash
    python speech_classifier.py
   ```

You will then be asked to choose your preferred input method (microphone or video). Enter `microphone` to use your microphone for live speech recognition, or enter `video` to use a video file for speech recognition.

### Output

You will see the output of the speech recognition process in the console. The script will display the recognized speech from the audio input. The audio is processed in chunks and recognized in real-time. All our classifiers will stop when you press `Q`, `ESC`, or otherwise close the window.

Example output:

<p align="center">
  <img src="Speech-Recognition/speech-classi.png" alt="Speech Recognition Output" width="100%">
</p>

---

## Special: Self-Trained Sentiment Classifier

In addition to the other pre-trained classifiers, this repository includes a special sentiment classifier that you can train yourself. The sentiment classifier is trained on a large dataset of tweets and can classify the sentiment of a sentence as positive, negative, or neutral. This is excellent for educational purposes and for understanding how sentiment analysis works.

### Files Included

- `sentiment_classifier.py`: Python script for sentiment classification.
- `train_model.py`: Python script for training the sentiment classifier, which includes data preprocessing, model training, and evaluation.
- `sentiment_model.pkl`: Trained sentiment classifier model.
- `vectorizer.pkl`: Trained vectorizer for the sentiment classifier.
- `training.1600000.processed.noemoticon.csv`: Training data for the sentiment classifier (Large file).
- `testdata.manual.2009.06.14.csv`: Test data for the sentiment classifier.
- `test.csv`: Sample test data for the sentiment classifier.
- `train.csv`: Sample training data for the sentiment classifier.
- `generate_small_dataset.py`: Python script for generating a small dataset from the large training data.
- `small_dataset.csv`: Small dataset generated from the large training data.

### Getting Started

1. **Clone the Repository**
   ```bash
   git clone
   cd AI-Classification/Sentiment-Analysis
   ```
2. **Install Dependencies:**
   Install the required Python dependencies.
   ```bash
   pip install scikit-learn pandas numpy nltk tqdm joblib
   ```
3. **Pull the Large Training Data:**
   The sentiment classifier is trained on a large dataset of tweets. The large training data is stored in a CSV file named `training.1600000.processed.noemoticon.csv`. This file is stored using Git LFS due to its large size. To pull the large training data, use the following command:

   ```bash
   git lfs install
   git lfs pull
   ```

   - Alternatively, you can download the large training data from the [Sentiment140 dataset](http://help.sentiment140.com/for-students) website and place it in the `Sentiment-Classifier` directory. However, using Git LFS is **recommended.**
   - If you do not have Git LFS installed, remember to install it first. You can find instructions on how to install Git LFS on the [official Git LFS website](https://git-lfs.github.com/).
   - If you still encounter problems, you can download the large training data file from my Google Drive, which is publicly available here:
     - [training.1600000.processed.noemoticon.csv](https://drive.google.com/file/d/1VHdnu9pNPIp2Gu6y0mOW5GB4azaEPu8I/view?usp=sharing)
     - [test.csv](https://drive.google.com/file/d/1NO46ZaztULg-oHImjf5E_Qwb3ht-GtMl/view?usp=sharing)
     - [train.csv](https://drive.google.com/file/d/1gs1pjTGsDEgzXd6o-9fvroM0ezmP45pn/view?usp=sharing)
     - [small_dataset.csv](https://drive.google.com/file/d/13r2bPO_dOPITn4UQh0_0wOWRfERD0UMj/view?usp=sharing)
     - If you need any other files, please [let me know](mailto:hoangson091104@gmail.com).

4. **Train the Sentiment Classifier**
   Run the `train_model.py` script to train the sentiment classifier.

   ```bash
   python train_model.py
   ```

   - When running the script, you will be asked to choose the dataset size (small or large). Enter `small` to use the small dataset or `large` to use the large dataset. The script will then preprocess the training data, train the sentiment classifier, and save the trained model and vectorizer to disk.
   - However, if you choose `small`, the script will use the small dataset provided in the repository. In order to use it, be sure to run the `generate_small_dataset.py` script first to generate the small dataset from the large training data.

   ```bash
   python generate_small_dataset.py
   ```

   - **Note:** Training the sentiment classifier on the large dataset may take a long time and require significant computational resources. However, it is recommended since it provides better model accuracy.
   - **Once again, if you are patient and have a good machine, you are encouraged use the large dataset to get a higher accuracy. Otherwise, use the small dataset for faster training.**
   - This script will then preprocess the training data, train the sentiment classifier, and save the trained model and vectorizer to disk.

   - Additionally, it will output the expected accuracy, F1 score, and expected confidence level of the sentiment classifier. The higher these statistics are, the better the sentiment classifier will perform. Of course, this is highly dependent on the training dataset size and quality. Feel free to experiment with the training data and parameters to improve the sentiment classifier's performance.

5. **Run Sentiment Classification**
   ```bash
   python sentiment_classifier.py
   ```
   You will then be asked to enter a sentence for sentiment classification. Enter a sentence, and the script will classify the sentiment of the sentence as positive, negative, or neutral, with a level of confidence.

### Output

The output will display the sentiment classification of the input sentence. The sentiment classifier will classify the sentiment as positive, negative, or neutral.

**Training Output Example:**

<p align="center">
  <img src="Sentiment-Analysis/sentiment-train.png" alt="Sentiment Classifier Training Output" width="100%">
</p>

**Classification Output Example:**

<p align="center">
  <img src="Sentiment-Analysis/sentiment-classi.png" alt="Sentiment Classifier Classification Output" width="100%">
</p>

Feel free to experiment with the sentiment classifier and test it with your own sentences and explore how powerful sentiment analysis can be!

---

## Containerization

If you would like to containerize this application, you can use Docker to create a container image. The Dockerfile is included in the repository for this purpose. To build the Docker image, use the following command:

```bash
docker build -t ai-multitask-classifiers .
```

This command will build a Docker image named `ai-multitask-classifiers` based on the Dockerfile in the repository. You can then run the Docker container using the following command:

```bash
docker run -p 5000:5000 ai-multitask-classifiers
```

This command will run the Docker container and expose port 5000 for the Flask web app. You can access the web app by visiting `http://127.0.0.1:500`.

Note: Before containerizing the application, ensure you have Docker installed on your machine. You can download Docker from the [official Docker website](https://www.docker.com/). Then, ensure that Docker Desktop is running on your machine before building and running the Docker container.

---

## Contact Information

For any questions or issues, please refer to the contact information below:

- GitHub: [Son Nguyen](https://github.com/hoangsonww)
- Email: [info@movie-verse.com](mailto:info@movie-verse.com)
- LinkedIn: [Son Nguyen](https://www.linkedin.com/in/hoangsonw/)

## Future Work

- Add more classifiers for various tasks such as emotion recognition, sentiment analysis, and more.
- Refine existing classifiers and improve their accuracy and performance.
- Add more sample images and videos for testing the classifiers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Live Information Website

Feel free to visit the live demo and information website [here](https://hoangsonww.github.io/AI-ML-Classifiers/).

It is a simple website that provides information about the classifiers in this repository.

---

Created with ❤️ by [Son Nguyen](https://github.com/hoangsonww) in 2024.

Thank you for visiting! 🚀
