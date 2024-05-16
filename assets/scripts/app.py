from flask import Flask, request, send_from_directory
import subprocess
import os
import sys

app = Flask(__name__, static_folder='.', static_url_path='')


@app.route('/')
def index():
    return send_from_directory('..', 'app.html')


@app.route('/run_script', methods=['POST'])
def run_script():
    choice = request.form.get('task')
    text_input = request.form.get('textInput', '')

    scripts = {
        '1': 'Vehicle-Classification/vehicle_detection.py',
        '2': 'Human-Face-Classification/faces_classification.py',
        '3': 'Mood-Classification/mood_classifier.py',
        '4': 'Flowers-Classification/flower_classification.py',
        '5': 'Object-Classification/object_classification.py',
        '6': 'Character-Recognition/ocr.py',
        '7': 'Animals-Classification/animal_classification.py',
    }

    script_path = scripts.get(choice)
    if script_path:
        full_path = os.path.join(os.getcwd(), script_path)
        python_executable = sys.executable

        my_env = os.environ.copy()
        my_env["PYTHONIOENCODING"] = "utf-8"

        try:
            process = subprocess.Popen(
                [python_executable, full_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(full_path),
                env=my_env
            )

            input_data = text_input if choice == '8' else 'webcam\n'
            output, error = process.communicate(input=input_data)

            if process.returncode == 0:
                return {'status': 'success', 'output': output}, 200
            else:
                filtered_error = '\n'.join(line for line in error.splitlines()
                                           if "nltk_data" not in line and "EOFError" not in line)
                return {'status': 'error', 'error': filtered_error or "Error executing script"}, 400
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode('utf-8')
            return {'status': 'error', 'error': error_message}, 400
    else:
        return {'status': 'error', 'error': 'Invalid selection'}, 400


if __name__ == '__main__':
    app.run(debug=True)
