# 🎨 AI Photo Colorizer & Restorer Pro Studio

A modern, high-performance desktop application built with Python, CustomTkinter, PyTorch, FastAI, and OpenCV to colorize and restore black-and-white photos using offline AI models.

---

## 🌟 Key Features
- **Multi-Model Support:** Includes Zhang (Caffe), DeOldify Artistic, and DeOldify Stable models.
- **Modern Dark UI:** Responsive interface with CustomTkinter and real-time live preview.
- **Hardware Acceleration:** Auto-detects CUDA GPU or falls back smoothly to CPU.
- **Post-Processing Enhancements:** Real-time controls for Saturation, Contrast, Brightness, Sharpness, and Gamma.

---

## 🚀 Quick Start (Windows)

1. **Clone or Download** this repository.
2. **Download Model Weights:** Download the 5 required model files from our [**Latest Release (v1.0.0)**](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/tag/v1.0.0) and place them inside the `models/` directory.
3. **Run the App:** Double-click **`run_app.bat`**. It will automatically set up the virtual environment, install dependencies, and launch the application!

---

## 📥 Required AI Model Weights

All required model files are hosted in the **Releases** section of this repository:

| Model File | Description | Download Link |
| :--- | :--- | :--- |
| `colorization_deploy_v2.prototxt` | Zhang Caffe Model Prototxt | [Download via Release](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/download/v1.0.0/colorization_deploy_v2.prototxt) |
| `pts_in_hull.npy` | Zhang Caffe Points | [Download via Release](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/download/v1.0.0/pts_in_hull.npy) |
| `colorization_release_v2.caffemodel` | Zhang Caffe Weights | [Download via Release](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/download/v1.0.0/colorization_release_v2.caffemodel) |
| `ColorizeArtistic_gen.pth` | DeOldify Artistic Weights | [Download via Release](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/download/v1.0.0/ColorizeArtistic_gen.pth) |
| `ColorizeStable_gen.pth` | DeOldify Stable Weights | [Download via Release](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/download/v1.0.0/ColorizeStable_gen.pth) |

---

## 📂 Project Structure

```text
AI-Photo-Colorizer-Pro/
├── models/                     <-- Place downloaded model files here
├── dummy/                      <-- Auto-created workspace folder
├── result_images/              <-- Auto-created output folder
├── app.py                      <-- Main CustomTkinter GUI code
├── run_app.bat                 <-- One-click Windows launcher script
├── requirements.txt            <-- Python dependencies list
└── README.md                   <-- Project documentation