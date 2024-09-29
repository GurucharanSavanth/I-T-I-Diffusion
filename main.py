# File: main.py
from flask import Flask, request, render_template, send_from_directory, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import threading
import zipfile
from modify_image import modify_image, load_models
import torch
import gc

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Max 32 MB file size

# Create directories if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Global variables for model building status and logs
log = ""
model_building_complete = False


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'images' not in request.files:
            return 'No file part'

        files = request.files.getlist('images')
        if len(files) == 0:
            return 'No selected file'

        prompts = request.form.getlist('prompt')
        num_variations = int(request.form.get('num_variations', 1))
        batch_size = int(request.form.get('batch_size', 1))

        results = []
        for i, file in enumerate(files):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            num_images = modify_image(
                file_path,
                prompts[i],
                num_variations,
                batch_size
            )
            results.append({
                'filename': filename,
                'num_variations': num_images
            })

        return render_template('results.html', results=results)
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
        load_models()  # This will cache all the models
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
    zipf = zipfile.ZipFile("outputs/all_variations.zip", 'w', zipfile.ZIP_DEFLATED)
    for root, _, files in os.walk(app.config['OUTPUT_FOLDER']):
        for file in files:
            zipf.write(os.path.join(root, file), file)
    zipf.close()
    return send_file("outputs/all_variations.zip", as_attachment=True)


@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()
    return jsonify({'status': 'Cache cleared'})


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
