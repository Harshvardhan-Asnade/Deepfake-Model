# ⚙️ How It Works

This document explains the journey of an image through the **DeepGuard** detection system, from upload to final verdict.

## The Pipeline

1.  **Image Upload & Preprocessing**
    *   **User Action**: You upload an image (JPG/PNG/WEBP).
    *   **System Action**:
        *   The server reads the image using `OpenCV`.
        *   It is converted to RGB color space.
        *   **Standardization**: The image is resized to `256x256` pixels and normalized mathematically (subtracting mean, dividing by standard deviation) to match the format the AI was trained on.

2.  **Multi-Branch Analysis**
    The image is simultaneously fed into four different "Experts" (AI Brains):

    *   **👁️ The Eye (RGB Branch)**: Looks at the image normally. It detects visual artifacts like warped backgrounds, asymmetric eyes, or "plastic" skin textures.
    *   **📉 The Signal Analyst (Frequency Branch)**: Converts the image into a frequency spectrum (FFT). It looks for invisible "grid" patterns left behind by the upsampling layers of Generative AI.
    *   **🔍 The Inspector (Patch Branch)**: Chops the image into 16 small squares. It analyzes each square individually. If even *one* square looks fake (e.g., a glitchy finger), the system raises a flag.
    *   **🧠 The Contextualizer (ViT Branch)**: Looks at the whole image relationship. It checks if the shadows, lighting, and physics make sense globally.

3.  **The Verdict (Fusion)**
    *   The features from all four experts are combined.
    *   A final "Classifier" weighs the evidence.
    *   It outputs a score between 0 and 1.
        *   **0.0 - 0.5**: Real
        *   **0.5 - 1.0**: Fake

4.  **Explainability (Heatmap)**
    *   The system runs a process called **Grad-CAM**.
    *   It traces the decision back to the pixels that triggered it.
    *   **Result**: A heat overlay.
        *   **Red Areas**: The parts of the image that convinced the AI it was fake/real.
        *   **Blue Areas**: Irrelevant parts.

## Real-World Example
**Input**: An AI-generated photo of a person.
*   **RGB Branch**: Sees the skin is too smooth. (Score: 0.7)
*   **Frequency Branch**: Detects a checkerboard grid. (Score: 0.9)
*   **Patch Branch**: Notices the earrings don't match. (Score: 0.8)
*   **ViT Branch**: Notices the background blur is inconsistent. (Score: 0.6)

**Final Result**: **FAKE (92% Confidence)**.
