# 💻 Technologies & Stack

This project utilizes a modern stack of machine learning and web technologies to deliver state-of-the-art deepfake detection.

## Core Frameworks
*   **Python 3.8+**: The primary programming language.
*   **PyTorch**: The deep learning framework used for model definition, training, and inference.
*   **Torchaudio/Torchvision**: Used for specific vision utilities and transforms.

## Machine Learning Libraries
*   **NumPy**: Fundamental package for scientific computing and array manipulation.
*   **OpenCV (cv2)**: Used for image I/O, color space conversions, and heatmap generation.
*   **Albumentations**: A fast and flexible image augmentation library used for resizing, normalization, and robust training data generation.
*   **SafeTensors**: A safe and fast file format for storing tensors (model weights), preventing pickle-based security vulnerabilities.
*   **TIMM (PyTorch Image Models)**: Used for initializing the EfficientNet and Swin Transformer backbones.

## Architectures & Models
*   **EfficientNetV2-S**: A convolutional neural network optimized for speed and accuracy, used as the RGB backbone.
*   **Swin Transformer V2 (Tiny)**: A hierarchical vision transformer used for capturing global semantic information.
*   **Custom CNN**: A custom 3-layer Convolutional Neural Network designed for frequency domain analysis.

## Web & API
*   **Flask**: A lightweight WSGI web application framework used to serve the API and frontend.
*   **Flask-CORS**: Handles Cross-Origin Resource Sharing (CORS) to allow frontend-backend communication.
*   **Werkzeug**: Utilities for safe file handling and server operations.

## Frontend
*   **HTML5 / CSS3**: Standard web technologies for the user interface.
*   **JavaScript (Vanilla)**: Handling client-side logic, API calls, and UI updates without heavy frameworks.

## Development Tools
*   **Git**: Version control.
*   **Tqdm**: Progress bars for training and long-running processes.
