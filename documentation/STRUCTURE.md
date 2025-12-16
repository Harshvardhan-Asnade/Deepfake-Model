# 📂 Project Structure

This document provides a high-level overview of the `open-deepfake-detection` repository organization.

```
Morden Detections system/
├── README.md                   # Primary project entry point
├── backend/                    # Flask backend API & Server
│   ├── app.py                  # Server entry point
│   ├── database.py             # Database management
│   ├── requirements_web.txt    # Python dependencies
│   └── uploads/                # Temporary storage
├── frontend/                   # Web user interface
│   ├── index.html              # Main dashboard page
│   ├── style.css               # Styling
│   ├── script.js               # Frontend logic
│   └── history_uploads/        # Saved history images
├── model/                      # Deepfake Detection Logic
│   ├── src/                    # Core source code
│   │   ├── config.py           # Configuration
│   │   ├── dataset.py          # Data Loading
│   │   ├── models.py           # Model Architecture
│   │   ├── inference.py        # Inference logic
│   │   ├── train.py            # Training loop
│   │   └── utils.py            # Helper functions
│   ├── evaluate_models.py      # Evaluation scripts
│   ├── finetune_datasetB.py    # Fine-tuning script
│   └── results/                # Chekpoints and logs
├── extension/                  # Chrome extension source code
└── documentation/              # Project documentation
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
