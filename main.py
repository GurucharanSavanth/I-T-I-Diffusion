from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import modify_image
from werkzeug.utils import secure_filename
import threading
import zipfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16 MB file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variables for model building status and logs
log = ""
model_building_complete = False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
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
                width = int(request.form.get('custom_width', 224))
                height = int(request.form.get('custom_height', 224))
                custom_size = (width, height)

            # Call the modify_image function
            num_images = modify_image.modify_image(file_path, prompt, strength, num_variations, resize_option, custom_size, batch_size, apply_custom_transforms)

            return render_template('results.html', num_variations=num_images)
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

@app.route('/download-all')
def download_all():
    zip_path = os.path.join(app.config['OUTPUT_FOLDER'], "all_variations.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(app.config['OUTPUT_FOLDER']):
            for file in files:
                if file != "all_variations.zip":
                    zipf.write(os.path.join(root, file), file)
    return send_from_directory(app.config['OUTPUT_FOLDER'], "all_variations.zip", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
