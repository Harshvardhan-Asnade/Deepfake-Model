# 🎓 Model Training Guide

This guide explains how to fine-tune the DeepGuard model on new datasets using the provided scripts.

## 📂 Data Preparation

Your dataset should be organized into directory-based classes or a standard Train/Val structure.

**Expected Structure:**
```
Dataset/
├── Train/
│   ├── Real/      # Verified real images
│   └── Fake/      # AI Generated images
└── Validation/
    ├── Real/
    └── Fake/
```

## 🛠️ Configuration

## 🛠️ Configuration

Open `model/src/config.py` to adjust hyperparameters before training:

*   `BATCH_SIZE`: Set to 16 or 32 depending on your VRAM (32 needs ~8GB VRAM).
*   `LEARNING_RATE`: Default `1e-4` for training from scratch, `5e-6` for fine-tuning.
*   `NUM_EPOCHS`: Number of full passes through the dataset.

## 🚀 Running the Training Script

### Option 1: Fine-tuning on Dataset B (Script)
We provided a specialized script `finetune_datasetB.py` for fine-tuning.

```bash
cd model
python finetune_datasetB.py
```
**What this script does:**
1.  Loads the `best_model.safetensors` from a previous run (if available).
2.  Sets a very low learning rate (`5e-6`) to avoid destroying learned features.
3.  Freezes the bottom layers (optional/configurable) and updates weights based on the new dataset.
4.  Saves checkpoints as `finetuned_datasetB_epX.safetensors`.

### Option 2: Generic Training (src/train.py)
For training from scratch or main training loops:

```bash
cd model
python -m src.train
```
*Note: Ensure you update the paths in `src/config.py` to point to your data source first.*

## 💾 Checkpoints & Saving

The system uses **SafeTensors** by default for security and speed.
*   **Best Model**: Saved as `results/checkpoints/best_model.safetensors` (Highest Validation Accuracy).
*   **Periodic**: Saved every epoch.

## 📊 Monitoring

Training progress is displayed in the terminal using `tqdm` progress bars.
*   **Loss**: Should decrease over time.
*   **Acc (Accuracy)**: Should increase.

If `Train Acc` is high (99%) but `Val Acc` is notably lower (e.g., 85%), your model is **Overfitting**.
*   **Solution**: Increase `DROPOUT`, add more Data Augmentation, or use `EarlyStopping`.
