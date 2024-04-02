from flask import Flask, request, render_template_string
import subprocess
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        choice = request.form.get('task')
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

        script_path = scripts.get(choice)
        if script_path:
            full_path = os.path.join(os.getcwd(), script_path)
            try:
                result = subprocess.run(['python', script_path], check=True, capture_output=True, cwd=os.path.dirname(full_path))
                return f"Script executed successfully! Output:\n{result.stdout.decode('utf-8')}"
            except subprocess.CalledProcessError as e:
                return f"An error occurred while running {script_path}. Error: {e.output.decode('utf-8')}"
        else:
            return "Invalid selection"

    # HTML for the form
    form_html = """
    <h1>AI Multitask Classifier</h1>
    <form method="POST">
        <select name="task">
            <option value="1">Vehicle Classification</option>
            <option value="2">Face Classification</option>
            <option value="3">Mood Classification</option>
            <option value="4">Flower Classification</option>
            <option value="5">Object Classification</option>
            <option value="6">Character Recognition</option>
            <option value="7">Animal Classification</option>
            <option value="8">Speech Recognition</option>
            <option value="9">Sentiment Analysis</option>
        </select>
        <button type="submit">Run</button>
    </form>
    """
    return render_template_string(form_html)

if __name__ == '__main__':
    app.run(debug=True)
