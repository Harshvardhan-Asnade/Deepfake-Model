<p align="center">
  <img src="Documentation/logo.ico" alt="DeepGuard" width="120"/>
</p>

<h1 align="center">DeepGuard</h1>

<p align="center">
  A multi-branch deep learning system for detecting AI-generated and manipulated images.
</p>

<p align="center">
  <a href="https://github.com/Harshvardhan-Asnade/Deepfake-Model/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://huggingface.co/spaces/Harshasnade/Deepfake-Detection-Model"><img src="https://img.shields.io/badge/Live_Demo-Hugging_Face-FFD21E?logo=huggingface" alt="Demo"/></a>
  <img src="https://img.shields.io/badge/Accuracy-96.97%25-00C853" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Params-12.3M-FF6D00" alt="Parameters"/>
</p>

---

## What is DeepGuard?

Most deepfake detectors rely on a single neural network to make a judgment call. DeepGuard takes a different approach — it runs **four specialized analysis branches in parallel**, each examining a different forensic signal, and fuses their findings into a single verdict. The result is **96.97% accuracy** across 13 diverse datasets, including generators the model has never seen during training.

The system ships as a self-contained web application: a Flask backend that runs the model and a drag-and-drop frontend that visualizes results with Grad-CAM heatmaps. Everything runs locally on your machine — no images leave your device.

**[Try the live demo on Hugging Face →](https://huggingface.co/spaces/Harshasnade/Deepfake-Detection-Model)**

---

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Model Lineage](#model-lineage)

---

## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/Harshvardhan-Asnade/Deepfake-Model.git
cd Deepfake-Model

# Set up a virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies and launch
pip install -r backend/requirements_web.txt
python backend/app.py
```

Open **http://localhost:7860** — drop an image and get results in seconds.

> **Docker alternative:** `docker build -t deepguard . && docker run -p 7860:7860 deepguard`

**System requirements:** Python 3.9+, 8 GB RAM minimum. GPU optional but recommended (NVIDIA CUDA or Apple Silicon MPS).

---

## How It Works

DeepGuard processes every image through four branches simultaneously, each designed to catch what the others might miss:

| Branch | Backbone | What It Looks For |
|--------|----------|-------------------|
| **Spatial (RGB)** | EfficientNet-V2-S | Visual artifacts — unnatural textures, lighting inconsistencies, blurred edges |
| **Frequency** | FFT + CNN | Spectral fingerprints invisible to the human eye, such as GAN upsampling grid patterns |
| **Patch** | Lightweight CNN (64×64 grid) | Localized inconsistencies — blending boundaries, regions that are "too perfect" |
| **Global Semantic** | Swin Transformer V2-Tiny | Scene-level logic — impossible physics, inconsistent perspectives, semantic incoherence |

The four feature vectors (1280-d + 128-d + 64-d + 768-d) are concatenated and passed through a classification head that outputs a single probability: how likely the image is to be synthetic.

Before the neural network even runs, DeepGuard also checks for **C2PA content credentials**, **EXIF generation metadata**, and **invisible watermarks** embedded by Stable Diffusion — a layered defense that catches the easy cases fast.

---

## Performance

### Universal Benchmark (Mark-V, 100K images across 13 datasets)

| Metric | Score |
|--------|:-----:|
| Accuracy | **96.97%** |
| Precision | 97.26% |
| ROC-AUC | 0.9771 |
| Loss (BCE) | 0.0912 |

### Zero-Shot Generalization — Generators Never Seen in Training

| Generator | Accuracy | Precision | Recall |
|-----------|:--------:|:---------:|:------:|
| SDXL Turbo | 96.2% | 95.8% | 96.7% |
| Adobe Firefly | 97.9% | 97.5% | 98.3% |
| Bing Image Creator | 98.5% | 98.2% | 98.8% |
| StyleGAN-XL | 98.9% | 98.6% | 99.2% |

### Why Four Branches Matter — Ablation Study

| Configuration | Accuracy | Error Reduction vs. Baseline |
|---------------|:--------:|:----------------------------:|
| RGB only | 94.73% | — |
| + Frequency | 97.18% | −46% |
| + Patch | 98.52% | −71% |
| **+ ViT (Full model)** | **99.64%** | **−94.5%** |

DeepGuard achieves higher accuracy than a standalone ViT-B/16 (96.12%, 86.6M params) while using only **12.3M parameters** — a 7× reduction.

---

## Project Structure

```
DeepGuard/
├── backend/                  # Flask API server (app.py, database.py)
├── model/
│   ├── src/                  # Model architecture, training, inference code
│   ├── results/checkpoints/  # Model weights (Mark-V.safetensors)
│   └── visualizations/       # Evaluation plots and charts
├── frontend/                 # Web UI (HTML/CSS/JS, PWA, Three.js backgrounds)
├── extension/                # Chrome browser extension
├── Documentation/            # Full project documentation (23 files)
├── release/                  # Release notes
├── Dockerfile
└── LICENSE
```

---

## Model Lineage

| Version | Training Data | Accuracy | Status |
|---------|--------------|:--------:|--------|
| **Mark-V** | 1.3M images, 13 datasets (universal) | **96.97%** | Active — production model |
| Mark-II | 525K images, 4 datasets | 99.64%\* | Legacy — overfits to seen data |
| Mark-IV | 1.3M images (from scratch) | 77.96% | Research prototype |
| Mark-III | 191K images (FF++ only) | 85.35% | Deprecated |

\*Mark-II scores 99.64% on its training distribution but drops to 77% on universal benchmarks, which is why Mark-V replaced it.

---

<p align="center">
  <sub>Defending authenticity in the age of generative AI.</sub>
</p>
