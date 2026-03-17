<div align="center">

<br/>

```
██████╗ ███████╗███████╗██████╗  ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
██║  ██║█████╗  █████╗  ██████╔╝██║  ███╗██║   ██║███████║██████╔╝██║  ██║
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
██████╔╝███████╗███████╗██║     ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
╚═════╝ ╚══════╝╚══════╝╚═╝      ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
```

### AI-Powered Deepfake Detection — *See Through the Fake*

<br/>

[![Accuracy](https://img.shields.io/badge/Accuracy-96.97%25-00ff88?style=for-the-badge&logo=target&logoColor=black)](/)
[![Model](https://img.shields.io/badge/Model-Mark--V-0ea5e9?style=for-the-badge&logo=databricks&logoColor=white)](/)
[![Performance](https://img.shields.io/badge/~25_FPS-Real--Time-f97316?style=for-the-badge&logo=lightning&logoColor=white)](/)
[![Privacy](https://img.shields.io/badge/Privacy-Local_First-6366f1?style=for-the-badge&logo=shield&logoColor=white)](/)
[![License](https://img.shields.io/badge/License-MIT-e2e8f0?style=for-the-badge)](/)

</div>

---

## What is DeepGuard?

**DeepGuard** is a high-precision, privacy-first deepfake detection system that identifies AI-generated and manipulated media with **96.97% accuracy** — without ever sending your data to the cloud.

At its core is the **Mark-V Hybrid Multi-Branch Neural Network**, which fuses spatial, frequency, patch-level, and transformer-based signals to reliably expose manipulations produced by modern generative models including GANs, diffusion pipelines, and hybrid architectures.

<div align="center">
  <br/>
 
  <sub><i>Mark-V performance across detection dimensions vs. baseline models</i></sub>
  <br/><br/>
</div>

---

## Key Features

### 🧠 Hybrid Multi-Branch Architecture
Combines four complementary analysis branches — **RGB spatial**, **FFT frequency**, **patch-level**, and **Vision Transformer** — into a single unified detection pipeline. No single cue is enough; DeepGuard reasons across all of them simultaneously.

### 🛡️ Advanced Authenticity Checks
Goes beyond pixel analysis. Detects metadata inconsistencies, validates **C2PA provenance credentials**, and uncovers hidden digital watermarks such as diffusion-based steganographic embeddings.

### 🔒 Privacy by Design
Fully offline, local-first processing. Your images and videos never leave your machine. No telemetry, no cloud calls, no compromises.

### 📊 Scan History & Audit Trail
Maintains a local, structured log of every scan — timestamps, verdicts, confidence scores — for transparency, compliance, and personal traceability.

### ⚡ Real-Time Performance
GPU-accelerated inference at **~25 FPS**, making DeepGuard practical for both interactive image review and live video stream analysis on consumer hardware.

---

## Model Performance — Mark-V

| Metric | Score | Notes |
|---|---|---|
| **Accuracy** | **96.97%** | Evaluated on a diverse multi-source deepfake benchmark |
| **Generalization** | **High** | Handles GANs, diffusion models, and hybrid pipelines |
| **Inference Speed** | **~25 FPS** | GPU-accelerated; suitable for real-time workflows |
| **False Positive Rate** | **Low** | Tuned to minimize false alarms on authentic media |

> The Mark-V model was trained and evaluated on a curated dataset spanning multiple deepfake generation methods, ensuring robustness against both classical GAN artifacts and modern diffusion-based manipulations.

---

## Architecture at a Glance

```
Input Media
    │
    ├──► RGB Branch          (spatial texture & artifact analysis)
    ├──► FFT Branch           (frequency-domain anomaly detection)
    ├──► Patch Branch         (local consistency & boundary checks)
    └──► ViT Branch           (global context via Vision Transformer)
              │
              ▼
       Feature Fusion
              │
              ▼
    Authenticity Engine
    ├── C2PA credential check
    ├── Metadata integrity scan
    └── Watermark probe
              │
              ▼
    Verdict + Confidence Score
```



## Project Structure

```
DeepGuard/
│
├── backend/            # FastAPI server · detection pipeline · authenticity engine
├── frontend/           # Web UI (React / Vanilla JS)
├── extension/          # Browser extension for real-time in-page scanning
├── model/              # Mark-V weights, training configs, evaluation artifacts
│   └── visualizations/ # Benchmark charts, radar plots, confusion matrices
├── release/            # Versioned production builds
├── scripts/            # Automation, CI/CD, and data utilities
└── Documentation/      # Full technical and user documentation
```

## License

Released under the [MIT License](LICENSE). Free to use, modify, and distribute.

---

<div align="center">
  <sub>Built with precision. Runs locally. Detects the undetectable.</sub>
</div>
