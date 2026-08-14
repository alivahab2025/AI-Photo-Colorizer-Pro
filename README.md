# 🎨 AI Photo Colorizer & Restorer Pro Studio

A modern, high-performance desktop application built with Python, CustomTkinter, PyTorch, FastAI, and OpenCV to colorize and restore black-and-white photos using offline AI models.

---

## 🚀 Download & Quick Start (No Python Required)

If you just want to run the app on Windows without installing Python or dependencies:

1. 📦 [**Download Portable Executable Version (v1.0.0 .ZIP)**](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/download/v1.0.0/AI_Photo_Colorizer_Pro_v1.0.0_Portable.zip)
2. Extract the `.zip` archive on your computer.
3. Download the **5 required model weights** from the table below and place them inside the `models/` directory.
4. Launch `AI_Photo_Colorizer_Pro.exe` and enjoy!

---

## 🛠️ Running from Source Code

If you prefer running the app via Python source code:

1. **Clone or Download** this repository to your local machine.
2. Download the required model weights (listed below) and place them inside the `models/` directory.
3. Double-click **`run_app.bat`**. It will automatically set up the virtual environment, install dependencies, and launch the application!

---

## 🌟 Key Features
- **Multi-Model AI Engine:** Switch seamlessly between Zhang (Caffe), DeOldify Artistic, and DeOldify Stable models.
- **Modern Dark Interface:** Built with CustomTkinter featuring clean dark themes, rounded widgets, and real-time live preview.
- **Hardware Acceleration:** Auto-detects NVIDIA CUDA GPU acceleration or smoothly falls back to CPU.
- **Post-Processing Enhancements:** Real-time controls for Saturation, Contrast, Brightness, Sharpness, and Gamma.
- **Dynamic Controls:** Toggle enhancements on/off with visual status indicators.

---

## 📥 Required AI Model Weights

All required model files are hosted directly in the **Releases** section of this repository:

| Model File | Description | Download Link |
| :--- | :--- | :--- |
| `colorization_deploy_v2.prototxt` | Zhang Caffe Model Architecture | [Download](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/download/v1.0.0/colorization_deploy_v2.prototxt) |
| `pts_in_hull.npy` | Zhang Caffe Points Data | [Download](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/download/v1.0.0/pts_in_hull.npy) |
| `colorization_release_v2.caffemodel` | Zhang Caffe Weights | [Download](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/download/v1.0.0/colorization_release_v2.caffemodel) |
| `ColorizeArtistic_gen.pth` | DeOldify Artistic Weights | [Download](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/download/v1.0.0/ColorizeArtistic_gen.pth) |
| `ColorizeStable_gen.pth` | DeOldify Stable Weights | [Download](https://github.com/alivahab2025/AI-Photo-Colorizer-Pro/releases/download/v1.0.0/ColorizeStable_gen.pth) |

---

## 📂 Project Directory Structure

```text
AI-Photo-Colorizer-Pro/
├── models/                     <-- Place downloaded model weights here
├── dummy/                      <-- Auto-created workspace folder
├── result_images/              <-- Auto-created output folder
├── app.py                      <-- Main CustomTkinter GUI application
├── run_app.bat                 <-- One-click Windows launcher script
├── requirements.txt            <-- Python package dependencies list
└── README.md                   <-- Project documentation