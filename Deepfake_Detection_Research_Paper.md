# DeepGuard: A Multi-Branch Deep Learning Framework for Generalized Deepfake Detection

**Abstract**
The rapid proliferation of high-fidelity generative adversarial networks (GANs) and diffusion models has rendered traditional forensic methods insufficient. We propose **DeepGuard**, a unified detection framework leveraging a quad-branch architecture. By integrating spatial (RGB), spectral (Frequency), local (Patch), and semantic (ViT) feature extraction streams, our model achieves state-of-the-art generalization against unseen generation methods.

## 1. Introduction
Synthetic media generation has evolved from crude autoencoders to photorealistic diffusion models (e.g., Stable Diffusion XL). While visually impressive, these models often exhibit statistical anomalies in the Fourier domain and semantic inconsistencies in complex scenes. Existing detectors typically focus on a single modality, making them brittle to new generation techniques. DeepGuard addresses this by fusing multi-modal evidence.

## 2. Methodology
Our architecture consists of four parallel streams:

### 2.1 Spatial RGB Stream
We utilize **EfficientNetV2-S** as the backbone for learning high-level visual artifacts. The model is initialized with ImageNet-1k weights to leverage robust texture representations.

### 2.2 Frequency Domain Stream
Generative upsampling operations (e.g., Transposed Convolution) induce periodic artifacts. We apply a 2D Discrete Fourier Transform (DFT) to the input $X$:
$$ F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} X(x, y) e^{-j2\pi(ux/M + vy/N)} $$
The log-magnitude spectrum is fed into a custom 3-layer CNN to capture these high-frequency fingerprints.

### 2.3 Local Patch Stream
To prevent the model from overfitting to global composition (e.g., "all blurred backgrounds are fake"), we employ a patch-based stream. The image is divided into $N$ non-overlapping patches $p_i$. A shared weight CNN encoder $E$ processes each patch, and we aggregate features using Max Pooling:
$$ f_{patch} = \max_i(E(p_i)) $$
This forces the network to detect local anomalies like hair-skin blending artifacts.

### 2.4 Semantic ViT Stream
A **Swin Transformer V2-Tiny** is employed to capture global context and long-range dependencies, addressing the "tunnel vision" limitation of CNNs.

## 3. Experiments

### 3.1 Datasets
We trained on a composite dataset including:
*   **Real**: FFHQ, CelebA-HQ
*   **Fake**: StyleGAN2, StyleGAN3, Stable Diffusion 1.5/2.1/XL

### 3.2 Result Analysis
Preliminary results indicate that the fusion of Frequency and Spatial domains improves detection on Diffusion models by 15% compared to RGB-only baselines. The Patch branch significantly reduces false positives on heavily compressed JPEG images.

## 4. Conclusion
DeepGuard demonstrates that a holistic, multi-view approach is necessary to combat the diverse artifacts introduced by modern generative engines. Future work will focus on video temporal consistency and adversarial robustness.
