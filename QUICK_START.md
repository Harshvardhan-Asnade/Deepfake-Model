# ⚡ Quick Start Guide

Get the **DeepGuard** detection system running on your local machine in under 5 minutes.

## Prerequisites
*   Python 3.8 or higher
*   pip (Python package manager)
*   (Optional) NVIDIA GPU for faster inference

## 1. Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/morden-detections-system.git
cd morden-detections-system

# Install dependencies
pip install -r requirements_web.txt
```

*Note: If you plan to train the model, you may need additional dependencies found in `requirements.txt` (if available) or install `tqdm` manually.*

## 2. Launching the Web App

The easiest way to use the system is via the Web UI.

1.  **Start the Server**:
    ```bash
    python app.py
    ```

2.  **Access the Dashboard**:
    Open your browser and navigate to:
    `http://localhost:5001` or `http://127.0.0.1:5001`

3.  **Analyze an Image**:
    *   Click "Upload Image" or drag & drop a file.
    *   Wait for the processing (approx. 1-2 seconds).
    *   View the **Real/Fake** probability and the **Heatmap**.

## 3. Using the Command Line (CLI)

You can run detection on a single image without the web server.

```bash
python src/inference.py --source /path/to/your/image.jpg
```

**Options:**
-   `--source`: Path to a single image or a directory of images.
-   `--checkpoints`: Path to a folder containing model checkpoints (for ensemble mode).

## 4. Troubleshooting

**"No checkpoint found" warning?**
The system will load a randomly initialized model for demonstration purposes if no weights are found. To get actual detection results, ensure you have placed `best_model.safetensors` in the `results/checkpoints/` directory.

**"CUDA not available"?**
The system automatically defaults to CPU if no GPU is detected. Inference will be slower but fully functional.
