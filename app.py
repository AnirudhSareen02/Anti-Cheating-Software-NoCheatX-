from flask import Flask, render_template, request
import subprocess
import os
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    detection_type = request.args.get('type')

    scripts = {
        'multiple': ("1_Multiple_people_detection.py", "Multiple People Detection"),
        'phone': ("2_Phone_detection.py", "Phone Detection"),
        'lip_tracking': ("4_lip_tracking.py", "Lip Tracking"),
        'sound': ("5_sound_detection.py", "Sound Detection"),
        'signature': ("6_signature_match.py", "Signature Match"),
        'same_person': ("7_same_person.py", "Same Person Detection"),
    }

    if detection_type not in scripts:
        return render_template('results.html', result="Unknown detection type!")

    script, title = scripts[detection_type]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"  # Force UTF-8 output on Windows

    try:
        completed = subprocess.run(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=60
        )
        output = completed.stdout + "\n" + completed.stderr
    except Exception as e:
        output = f"Error occurred: {str(e)}"

    return render_template('results.html', result=f"<b>{title} Output:</b><br><pre>{output}</pre>")

if __name__ == '__main__':
    app.run(debug=True)
