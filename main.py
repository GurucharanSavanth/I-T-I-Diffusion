from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import modify_image  # Assuming the provided script is saved as modify_image.py
from werkzeug.utils import secure_filename, send_file
import threading
import psutil  # For memory and CPU resource management
import zipfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16 MB file size

# Create directories if they don't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['OUTPUT_FOLDER']):
    os.makedirs(app.config['OUTPUT_FOLDER'])

# Global variables for model building status and logs
log = ""
model_building_complete = False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            prompt = request.form.get('prompt', '')
            strength = float(request.form.get('strength', 7.5))
            num_variations = int(request.form.get('num_variations', 1))
            resize_option = request.form.get('resize_option', 'default')
            custom_size = None
            batch_size = int(request.form.get('batch_size', 1))
            apply_custom_transforms = request.form.get('apply_custom_transforms') == 'on'

            if resize_option == 'custom':
                width = int(request.form['custom_width'])
                height = int(request.form['custom_height'])
                custom_size = (width, height)

            # Call the modify_image function
            modify_image.modify_image(file_path, prompt, strength, num_variations, resize_option, custom_size, batch_size, apply_custom_transforms)

            return render_template('results.html', num_variations=num_variations)
    else:
        return render_template('index.html')

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/status')
def status():
    return jsonify({'log': log, 'complete': model_building_complete})

def build_model():
    global log, model_building_complete
    try:
        log += "Model building started...\n"
        # Example: Load custom datasets or perform intensive tasks
        # ... model building logic ...
        log += "Model building completed.\n"
    except Exception as e:
        log += f"Model building failed: {e}\n"
    finally:
        model_building_complete = True

@app.route('/start-model-building', methods=['POST'])
def start_model_building():
    global model_building_complete
    model_building_complete = False
    threading.Thread(target=build_model).start()
    return jsonify({'status': 'started'})

# New feature: Download all results as a ZIP file
@app.route('/download-all')
def download_all():
    zipf = zipfile.ZipFile("outputs/all_variations.zip", 'w', zipfile.ZIP_DEFLATED)
    for root, _, files in os.walk(app.config['OUTPUT_FOLDER']):
        for file in files:
            zipf.write(os.path.join(root, file), file)
    zipf.close()
    return send_file("outputs/all_variations.zip", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
