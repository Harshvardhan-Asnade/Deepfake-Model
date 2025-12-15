# 📂 Project Structure

This document provides a high-level overview of the `open-deepfake-detection` repository organization.

```
Morden Detections system/
├── README.md                   # Primary project entry point
├── app.py                      # Flask backend API & Server entry point
├── requirements_web.txt        # Python dependencies for the web application
├── finetune_datasetB.py        # Example script for fine-tuning on new data
├── extensions/                 # Chrome extension source code
├── frontend/                   # Web user interface files
│   ├── index.html              # Main dashboard page
│   ├── style.css               # Styling
│   └── script.js               # Frontend logic
├── src/                        # Core source code
│   ├── config.py               # Global configuration (Hyperparameters, Paths)
│   ├── dataset.py              # Custom PyTorch Dataset & Data Loading
│   ├── models.py               # DeepGuard Model Architecture Definition
│   ├── inference.py            # Inference logic & Ensemble support
│   ├── train.py                # Main training loop
│   └── utils.py                # Helper functions (FFT, logging)
├── results/                    # Output directory
│   ├── checkpoints/            # Saved model weights (.safetensors)
│   └── logs/                   # Training logs
└── uploads/                    # Temporary storage for analyzed images
```

## Key Files Description

### `src/models.py`
Contains the `DeepfakeDetector` class, which defines the 4-branch architecture:
1.  **RGB Stream**: EfficientNetV2 encoder.
2.  **Frequency Stream**: FFT-based spectral analysis.
3.  **Patch Stream**: Local texture analysis.
4.  **ViT Stream**: Swin Transformer for global context.

### `app.py`
The web server that:
-   Initializes the model.
-   Exposes the `/api/predict` endpoint.
-   Handles image uploads and preprocessing.
-   Generates Explainability Heatmaps (Grad-CAM/Activation Maps).

### `src/dataset.py`
Handles data ingestion. It implements the `DeepfakeDataset` class which:
-   Reads images from directories.
-   Applies `Albumentations` augmentations (Resize, Normalize, Compression, Noise).
-   Computes the Frequency Transform on the fly.
