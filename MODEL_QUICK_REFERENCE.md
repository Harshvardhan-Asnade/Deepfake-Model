# 🧠 Model Quick Reference

**DeepGuard** is a hybrid deep learning model for deepfake detection.

## 📋 Specifications

| Feature | Details |
| :--- | :--- |
| **Input Resolution** | 256x256 pixels |
| **Color Space** | RGB (normalized) |
| **Batch Size** | 32 (Default) |
| **Output** | Single scalar logit (Sigmoid -> Probability) |
| **Classes** | 0: Real, 1: Fake |
| **Backbone 1** | EfficientNetV2-Small (Spatial) |
| **Backbone 2** | Swin Transformer V2-Tiny (Semantic) |
| **Backbone 3** | Custom FFT CNN (Frequency) |
| **Backbone 4** | Custom Patch CNN (Texture) |

## 🔢 Inputs & Outputs

### Input Tensor
*   **Shape**: `(B, 3, 256, 256)`
*   **Normalization**: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`

### Output Dictionary
When calling `inference.py` or the API, the model returns:
*   **`prediction`**: "REAL" or "FAKE" (Threshold > 0.5)
*   **`confidence`**: 0.5 - 1.0 (Certainty of the prediction)
*   **`fake_probability`**: 0.0 - 1.0 (Raw probability of being fake)
*   **`heatmap`**: Base64 encoded JPEG (Grad-CAM visualization overlay)

## ⚡ Performance Tips
*   **Inference Speed**: ~40ms/image on GPU (RTX 3060), ~300ms/image on CPU.
*   **Memory**: Requires ~2GB VRAM for inference.
