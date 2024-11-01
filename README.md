
# Image Variation Generator

## Overview
This application utilizes a Flask-based web server to provide a user interface for generating image variations using the Stable Diffusion model. The application allows users to upload an image, specify parameters for the variation generation, and view the results directly on the webpage.

## Key Features
- Image upload functionality.
- Real-time progress updates during image processing.
- Display of generated image variations directly in the browser.

## Requirements
- Python 3.x
- Flask
- PyTorch (with or without CUDA support)
- PIL (Python Imaging Library)
- torchvision
- diffusers

## Installation

### Clone the Repository
```bash
git clone https://github.com/GurucharanSavanth/DeepLearning_Image-text-image.git
cd DeepLearning_Image-text-image
```

### Set Up a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # For Unix or MacOS
venv\Scripts\activate  # For Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
   ```bash
   flask run
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Use the web interface to upload an image, set parameters, and start the image variation generation process.

4. View the progress in real-time on the webpage and see the output images once processing is complete.

## Development Notes
- The application is set up for development and testing purposes. For production deployment, consider using a production WSGI server.
- The application can run on systems with or without a CUDA-enabled GPU. It defaults to CPU processing if CUDA is not available.

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to check [issues page](link-to-issues-page) if you want to contribute.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Project Link: [https://github.com/your-repo/image-variation-generator](https://github.comGurucharanSavanth/DeepLearning_Image-text-image)

