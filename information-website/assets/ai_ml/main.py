import subprocess
import os


def run_script(script_path):
    try:
        script_dir = os.path.dirname(script_path)
        subprocess.run(['python', script_path], check=True, cwd=script_dir)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_path}.")
        print(e)


def main():
    scripts = {
        '1': 'Vehicle-Classification/vehicle_detection.py',
        '2': 'Human-Face-Classification/faces_classification.py',
        '3': 'Mood-Classification/mood_classifier.py',
        '4': 'Flowers-Classification/flower_classification.py',
        '5': 'Object-Classification/object_classification.py',
        '6': 'Character-Recognition/ocr.py',
        '7': 'Animals-Classification/animal_classification.py',
        '8': 'Speech-Recognition/speech_classifier.py',
        '9': 'Sentiment-Analysis/sentiment_classifier.py'
    }

    print("""
    AI Multitask Classifier
    Select a task to perform:
    1) Vehicle Classification
    2) Face Classification
    3) Mood Classification
    4) Flower Classification
    5) Object Classification
    6) Character Recognition
    7) Animal Classification
    8) Speech Recognition
    9) Sentiment Analysis
    10) Exit
    """)

    choice = input("Enter the number of the task: ")
    script_path = scripts.get(choice)

    if script_path:
        full_path = os.path.join(os.getcwd(), script_path)
        run_script(full_path)
    elif choice == '10':
        print("Exiting.")
    else:
        print("Invalid selection, exiting.")


if __name__ == "__main__":
    main()
