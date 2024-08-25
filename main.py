# File: main.py
from flask import Flask, request, render_template, send_from_directory
import os
import modify_image  # Assuming the provided script is saved as modify_image.py
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create directories if they don't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['OUTPUT_FOLDER']):
    os.makedirs(app.config['OUTPUT_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle the form submission
        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prompt = request.form['prompt']
            strength = float(request.form['strength'])
            num_variations = int(request.form['num_variations'])
            resize_option = request.form['resize_option']
            custom_size = None
            if resize_option == 'custom':
                width = int(request.form['custom_width'])
                height = int(request.form['custom_height'])
                custom_size = (width, height)
            # Call the modify_image function
            modify_image.modify_image(file_path, prompt, strength, num_variations, resize_option, custom_size)
            # Return the results
            return render_template('results.html', num_variations=num_variations)
    else:
        return render_template('index.html')

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
from flask import jsonify
import threading

log = ""
model_building_complete = False

def build_model():
    global log, model_building_complete
    try:
        # Your model building code here
        log += "Model building started...\n"
        # ... model building logic ...
        log += "Model building completed.\n"
    finally:
        model_building_complete = True

@app.route('/start-model-building', methods=['POST'])
def start_model_building():
    global model_building_complete
    model_building_complete = False
    threading.Thread(target=build_model).start()
    return jsonify({'status': 'started'})

@app.route('/status')
def status():
    return jsonify({'log': log, 'complete': model_building_complete})

if __name__ == '__main__':
    app.run(debug=True)
