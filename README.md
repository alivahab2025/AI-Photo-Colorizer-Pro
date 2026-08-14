\# 🎨 AI Photo Colorizer \& Restorer Pro Studio



A professional desktop application built with Python, CustomTkinter, PyTorch, FastAI, and OpenCV to colorize and restore black-and-white photos using offline AI models.



\---



\## 🚀 Quick Start (Windows)



1\. Clone or download this repository.

2\. Download the required model weights (listed below) and place them inside the `models/` directory.

3\. Double-click \*\*`run\_app.bat`\*\*. It will automatically set up the Python environment, install dependencies, and launch the application!



\---



\## 📥 Required AI Model Weights (Manual Download)



Download the following files and place them into the `models/` folder:



| Model File | Description | Download Link |

| :--- | :--- | :--- |

| `colorization\_deploy\_v2.prototxt` | Zhang Caffe Prototxt | \[Download](https://fastly.jsdelivr.net/gh/richzhang/colorization@master/models/colorization\_deploy\_v2.prototxt) |

| `pts\_in\_hull.npy` | Zhang Caffe Points | \[Download](https://fastly.jsdelivr.net/gh/richzhang/colorization@master/resources/pts\_in\_hull.npy) |

| `colorization\_release\_v2.caffemodel` | Zhang Caffe Weights | \[Download](https://huggingface.co/spaces/sczhou/CodeFormer/resolve/main/weights/caffe/colorization\_release\_v2.caffemodel) |

| `ColorizeArtistic\_gen.pth` | DeOldify Artistic Model | \[Download](https://data.deepai.org/deoldify/ColorizeArtistic\_gen.pth) |

| `ColorizeStable\_gen.pth` | DeOldify Stable Model | \[Download](https://data.deepai.org/deoldify/ColorizeStable\_gen.pth) |



\---



\## 📂 Project Directory Structure



```text

AiColorizePhoto/

├── models/                     <-- Place downloaded model weights here

├── dummy/                      <-- Created automatically for DeOldify

├── result\_images/              <-- Created automatically for output saves

├── app.py                      <-- Main GUI Python code

├── run\_app.bat                 <-- One-click Windows launcher

├── requirements.txt            <-- Dependencies list

└── README.md                   <-- Project documentation

