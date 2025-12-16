# 🛡️ DeepGuard: Deepfake Detection System

> A state-of-the-art Multi-Branch Deep Learning system designed to distinguish between real camera-captured images and AI-generated synthetic media (GANs, Diffusion Models).

---

## 📚 Documentation Index

We have initialized comprehensive documentation for every aspect of this project:

### 🚀 Getting Started
*   [**Quick Start Guide**](documentation/QUICK_START.md): Get up and running in 5 minutes.
*   [**Project Structure**](documentation/STRUCTURE.md): Understand the file organization.

### 🧠 Model & Technology
*   [**How It Works**](documentation/HOW_IT_WORKS.md): High-level explanation of the pipeline.
*   [**Model Architecture**](documentation/MODEL_ARCHITECTURE.md): Deep dive into the 4-Branch Neural Network.
*   [**Model Quick Reference**](documentation/MODEL_QUICK_REFERENCE.md): Inputs, Outputs, and Specs.
*   [**Training Guide**](documentation/model_training_guide.md): How to fine-tune the model on your own data.
*   [**Technologies Used**](documentation/TECHNOLOGIES.md): The full tech stack (PyTorch, Flask, etc.).

### 🎓 Educational Resources
*   [**Educational Booklet**](documentation/EDUCATIONAL_BOOKLET.md): What are deepfakes and why are dangerous?
*   [**Research Paper Draft**](documentation/Deepfake_Detection_Research_Paper.md): Academic abstract and methodology.

---

## ✨ Features

*   **Multi-Modal Detection**: Analyzes images using 4 distinct branches:
    *   **RGB** (Visual Artifacts)
    *   **Frequency** (Spectral/Grid Artifacts)
    *   **Patch** (Local Inconsistencies)
    *   **ViT** (Global Semantic Logic)
*   **Explainable AI**: Generates **Heatmaps** to show exactly *why* an image was flagged.
*   **Web Interface**: Clean, modern dashboard for easy dragging-and-dropping of images.
*   **API Ready**: REST API allows integration into other apps.

## 🛠️ Installation

```bash
# 1. Clone
git clone https://github.com/yourusername/morden-detections-system.git
cd morden-detections-system

# 2. Install
cd backend
pip install -r requirements_web.txt
cd ..
```

## 🏃 Usage

### Start the Web Dashboard
```bash
cd backend
python app.py
```
Visit `http://localhost:5001` in your browser.

### Command Line Inference
```bash
cd model
python -m src.inference --source test_images/image.jpg
```

---

## ⚠️ Disclaimer
While DeepGuard is trained on a vast dataset of real and fake images, no detection system is 100% perfect. This tool should be used as an *aid* for verification, not as absolute proof.
