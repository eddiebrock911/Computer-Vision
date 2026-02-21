<div align="center">

# 👁️ Computer Vision with Python

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Computer%20Vision-AI%20Powered-blueviolet?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Real--Time-Detection-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-red?style=for-the-badge"/>
</p>

<p align="center">
  A Python-based <strong>Computer Vision project</strong> using OpenCV for real-time image and video analysis.<br/>
  Implements core CV techniques including object detection, image processing, and visual feature extraction.
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-contributing">Contributing</a>
</p>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Core Concepts](#-core-concepts)
- [Example Output](#-example-output)
- [Use Cases](#-use-cases)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

**Computer-Vision** is a Python project built around **OpenCV** — one of the most widely used libraries for real-time computer vision. This project demonstrates practical implementations of CV techniques that can be applied to images, video streams, and webcam feeds.

Whether you're a beginner exploring computer vision or a developer building AI-powered visual pipelines, this project provides a clean, modular foundation.

---

## ✨ Features

- 👁️ **Real-time video/webcam processing** via OpenCV
- 🎯 **Object & face detection** using Haar cascades / DNN
- 🖼️ **Image preprocessing** — grayscale, blur, threshold, edge detection
- 📐 **Contour detection & shape analysis**
- 🎨 **Color space conversions** — BGR, RGB, HSV, Gray
- 📦 **Bounding box drawing** with labels
- ⚡ **Lightweight** — pure Python, single script entry point

---

## ⚙️ How It Works

```
Input Source (Image / Video / Webcam)
              │
              ▼
┌─────────────────────────────────────┐
│           OpenCV Pipeline           │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  1. Frame Capture            │  │  ← cv2.VideoCapture / imread
│  └──────────────┬───────────────┘  │
│                 │                   │
│  ┌──────────────▼───────────────┐  │
│  │  2. Preprocessing            │  │  ← Resize, Grayscale, Blur,
│  │                              │  │     Normalize, Threshold
│  └──────────────┬───────────────┘  │
│                 │                   │
│  ┌──────────────▼───────────────┐  │
│  │  3. Feature Detection        │  │  ← Edge detection (Canny),
│  │                              │  │     Contours, Keypoints
│  └──────────────┬───────────────┘  │
│                 │                   │
│  ┌──────────────▼───────────────┐  │
│  │  4. Object / Face Detection  │  │  ← Haar Cascade / DNN Model
│  └──────────────┬───────────────┘  │
│                 │                   │
│  ┌──────────────▼───────────────┐  │
│  │  5. Annotation & Display     │  │  ← Draw bounding boxes,
│  │                              │  │     labels, contours
│  └──────────────┬───────────────┘  │
└─────────────────┼───────────────────┘
                  │
                  ▼
     Output Window / Saved Image/Video
```

---

## 🗂️ Project Structure

```
Computer-Vision/
│
├── 🐍 vision1.py       # Main computer vision script
├── 📄 LICENSE          # Apache 2.0
└── 📄 README.md        # You are here
```

> **Note:** Input images/video can be placed in the project root or passed as arguments to `vision1.py`.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `Python 3.8+` | Core programming language |
| `OpenCV (cv2)` | Image & video capture, processing, detection |
| `NumPy` | Array/matrix operations on pixel data |
| `Matplotlib` *(optional)* | Visualization & plotting results |

---

## 📦 Installation

**1. Clone the repository:**
```bash
git clone https://github.com/eddiebrock911/Computer-Vision.git
cd Computer-Vision
```

**2. Create & activate a virtual environment:**
```bash
# Create
python -m venv venv

# Activate — Linux/Mac
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install opencv-python numpy matplotlib
```

> For headless environments (servers without display):
```bash
pip install opencv-python-headless numpy
```

---

## ▶️ Usage

**Run the main script:**
```bash
python vision1.py
```

**Common OpenCV operations you can extend:**

```python
import cv2
import numpy as np

# --- Load an image ---
img = cv2.imread("input.jpg")

# --- Grayscale conversion ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Gaussian Blur ---
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Edge Detection (Canny) ---
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# --- Contour Detection ---
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# --- Face Detection (Haar Cascade) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# --- Display ---
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Real-time Webcam Feed:**
```python
import cv2

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Your processing here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Webcam Feed", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press Q to quit
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📚 Core Concepts

### Image Representation
```
Each image = NumPy array of shape (Height, Width, Channels)

BGR Image:  shape = (480, 640, 3)   ← OpenCV default color order
Gray Image: shape = (480, 640)      ← Single channel
```

### Key OpenCV Functions

| Function | Description |
|---|---|
| `cv2.imread()` | Load image from disk |
| `cv2.VideoCapture()` | Open camera or video file |
| `cv2.cvtColor()` | Convert color spaces (BGR ↔ Gray ↔ HSV) |
| `cv2.GaussianBlur()` | Smooth image to reduce noise |
| `cv2.Canny()` | Detect edges using gradient magnitude |
| `cv2.findContours()` | Detect object boundaries |
| `cv2.rectangle()` | Draw bounding boxes |
| `cv2.putText()` | Overlay text labels on frames |
| `cv2.imshow()` | Display image/frame in window |

### Color Spaces

```
BGR  →  Default in OpenCV
RGB  →  Standard (swap R and B from BGR)
GRAY →  Single channel, used for detection
HSV  →  Hue-Saturation-Value, great for color filtering
```

---

## 📋 Example Output

| Operation | Input | Output |
|---|---|---|
| Grayscale | Color image | Single-channel gray image |
| Edge Detection | Grayscale image | White edges on black background |
| Face Detection | Portrait photo | Face bounded by blue rectangle |
| Contour Detection | Binary image | Green contours drawn on objects |
| Webcam Feed | Live video | Real-time annotated frames |

---

## 💡 Use Cases

| Domain | Application |
|---|---|
| 🔐 **Security** | Real-time face detection & surveillance |
| 🏭 **Manufacturing** | Defect detection on production lines |
| 🚗 **Autonomous Vehicles** | Lane detection, obstacle recognition |
| 🏥 **Healthcare** | Medical image analysis |
| 📦 **Retail** | Product recognition & shelf monitoring |
| 🎮 **Gaming / AR** | Gesture control, augmented reality |
| 📸 **Photography** | Auto-enhancement, object segmentation |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python` |
| Camera not opening | Check `VideoCapture(0)` index; try `1` or `2` for external cams |
| Window not displaying | Ensure you have a display; use `opencv-python` not headless |
| `imshow` crashes on Linux | Install `python3-tk` or use `matplotlib` for display |
| Slow FPS on webcam | Reduce resolution: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` |
| Face not detected | Tune `scaleFactor` and `minNeighbors` in `detectMultiScale()` |

---

## 🚀 Future Enhancements

- [ ] Deep learning-based object detection (YOLOv8)
- [ ] Multi-face tracking across video frames
- [ ] Gesture recognition with MediaPipe
- [ ] OCR integration (Tesseract)
- [ ] Real-time emotion detection
- [ ] Streamlit web UI for live demo

---

## 🤝 Contributing

Contributions are welcome!

```bash
# 1. Fork the repo on GitHub

# 2. Clone your fork
git clone https://github.com/your-username/Computer-Vision.git

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make your changes & commit
git commit -m "feat: describe your change"

# 5. Push & open a Pull Request
git push origin feature/your-feature-name
```

**Ideas for contributions:**
- 🎯 Add YOLOv8 / MobileNet object detection
- 🖐️ Hand gesture recognition with MediaPipe
- 📊 Add FPS counter and performance metrics
- 🌐 Build a Streamlit/Gradio live demo UI

---

## 📄 License

This project is licensed under the **Apache 2.0 License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [eddiebrock911](https://github.com/eddiebrock911)

⭐ **Star this repo** if you found it useful!

</div>
