````markdown
# 🇮🇳⚡ AGASTYA — SIH ⚡🇮🇳

AGASTYA is our Smart India Hackathon (SIH) project.

This repository contains the current demonstration implementation, including real-time computer vision using YOLO, webcam input, GPU/CPU inference support, and voice-based alerts.

---

## 🚀 Features

- 🎥 Real-time webcam-based computer vision
- 🤖 YOLO object detection
- ⚡ NVIDIA GPU acceleration when CUDA is available
- 🖥️ CPU fallback when CUDA is unavailable
- 🔊 Voice alerts using pyttsx3
- 📊 Experiment/event logging
- 🧠 Configurable confidence threshold and image size
- 📷 Configurable camera resolution and FPS

---

## 🛠️ Tech Stack

- Python
- OpenCV
- PyTorch
- Ultralytics YOLO
- pyttsx3

---

## 📁 Project Structure

```text
AGASTYA-SIH/
│
├── final_sih_demo_project/
│   └── src/
│       └── demo.py
│
├── models/
│   └── bigbang.pt
│
├── requirements.txt
├── requirements-cpu.txt
├── requirements-nvidia.txt
├── README.md
└── .gitignore
````

---

## 💻 Requirements

* Python 3
* Webcam
* NVIDIA GPU is optional
* Internet connection for installing Python dependencies

The application automatically detects whether CUDA is available and selects NVIDIA GPU or CPU inference accordingly.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/chethan779/-AGASTYA-SIH-.git
cd AGASTYA-SIH
```

### 2. Create a virtual environment

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 🧩 Install Dependencies

Choose one of the following options depending on your hardware.

### 🟢 NVIDIA GPU

For systems with a compatible NVIDIA GPU:

```bash
pip install -r requirements-nvidia.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

This installs the NVIDIA CUDA 13.0 build of PyTorch together with the common AGASTYA dependencies.

#### Verify the installation

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

A working NVIDIA installation should report CUDA availability as:

```text
True
```

### 🔵 CPU / Non-NVIDIA

For systems without an NVIDIA GPU:

```bash
pip install -r requirements-cpu.txt --index-url https://download.pytorch.org/whl/cpu
```

The application will use CPU inference automatically.

---

## ▶️ Running AGASTYA

From the repository root:

### Linux / macOS

```bash
python3 final_sih_demo_project/src/demo.py
```

### Windows

```bash
python final_sih_demo_project/src/demo.py
```

---

## 📷 Camera Configuration

Camera settings can be configured directly in demo.py.

### Example

```python
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
```

The application reads back the actual camera configuration after initialization.

A resolution of 1280 × 720 is recommended for better compatibility across different webcams and systems.

---

## ⚙️ Inference Configuration

The YOLO configuration can be adjusted in demo.py.

### Example

```python
CONFIDENCE = 0.30
IMAGE_SIZE = 416
```

These values can be adjusted depending on the required detection behavior and system performance.

---

## 🧠 Model

The project uses the trained YOLO model:

```text
models/bigbang.pt
```

The model is loaded using a project-relative path, allowing the repository to be moved between systems without changing the model path.

---

## ⚡ GPU / CPU Behavior

AGASTYA automatically checks whether CUDA is available.

### If CUDA is available:

```text
Using NVIDIA GPU
```

### Otherwise:

```text
Using CPU
```

No source-code changes are required to switch between GPU and CPU inference.

---

## 📋 Direct Dependencies

The project directly uses the following Python packages:

* opencv-python
* pyttsx3
* torch
* ultralytics

The additional packages required internally by these libraries are installed automatically by pip.

---

## 🇮🇳 Smart India Hackathon

**Project:** AGASTYA
**Event:** Smart India Hackathon (SIH)

---

## 👥 Team

---

## 📜 License

No license has been specified for this project.

```
```
