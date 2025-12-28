# Installation & Setup Guide

Since DeepGuard is a complex system involving Python backends and Deep Learning libraries, please follow these steps carefully.

## Prerequisites
- **Python 3.8+** (Recommended: 3.10)
- **pip** (Python Package Manager)
- **Git**

## 1. Clone the Repository
```bash
git clone <your-repo-url>
cd morden-detections-system
```

## 2. Backend Setup
The backend requires specific ML libraries. We recommend using a Virtual Environment.

```bash
# Navigate to backend
cd backend

# Create virtual environment (Mac/Linux)
python3 -m venv venv
source venv/bin/activate

# Create virtual environment (Windows)
# python -m venv venv
# venv\Scripts\activate

# Install dependencies
pip install -r requirements_web.txt
```

> **Note**: If `requirements_web.txt` does not include the torch libraries, you may need to install them manually:
> `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (Adjust for your OS/CUDA version)

## 3. Model Setup
The backend looks for a checkpoint file in `model/results/checkpoints/`.
1.  Ensure you have `best_model.safetensors` (or similar) in that directory.
2.  If you trained the model yourself, the file should already be there.

## 4. Running the Application
```bash
# From the 'backend' directory, with venv activated:
python app.py
```
You should see:
```
Using device: cuda (or mps/cpu)
✅ Model loaded successfully!
🌐 Starting server on http://localhost:5001
```

## 5. Using the App
Open your web browser and navigate to `http://localhost:5001`.
You can now drag and drop images to test them!

## ❓ Troubleshooting

### Common Issues

#### 1. "Model file not found"
**Error**: `❌ CRITICAL ERROR: Model file not found at: .../best_model.safetensors`
**Fix**:
- Ensure you have downloaded or trained a model.
- Place the `.safetensors` file in `model/results/checkpoints/`.
- Rename it to `best_model.safetensors` or update `load_model()` in `backend/app.py`.

#### 2. "Safetensors not installed"
**Error**: `Warning: safetensors not installed.`
**Fix**: Run `pip install safetensors`.

#### 3. CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`
**Fix**:
- Open `model/src/config.py`.
- Lower `BATCH_SIZE` (e.g., from 32 to 16 or 8).

#### 4. Import Errors (e.g., `Module not found: albumentations`)
**Fix**: Ensure your virtual environment is activated and you ran `pip install -r requirements_web.txt`.

