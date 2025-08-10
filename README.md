# Sign Language Recognition App

An **American Sign Language Recognition** application built with **TensorFlow**, **MediaPipe**, and **OpenCV**.  
The app detects a hand in the webcam feed, processes it using MediaPipe, and classifies the gesture using a pre-trained deep learning model.

---

## ✨ Features

-  **Real-time webcam capture**
-  **Hand detection** with MediaPipe
-  **American Sign Language (ASL) classification** using a trained TensorFlow model
-  **Annotated video feed** with predicted labels
---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
bash
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip insteall -r requirements.txt
```

## Usage

Run the application
```bash
python -m app.py
```
## Training
The model was trained based on a Kaggle dataset (Muvezwa, 2019):
https://www.kaggle.com/datasets/kuzivakwashe/significant-asl-sign-language-alphabet-dataset
This dataset consists of over 70,000 coloured, RGB images with a resolution of 320x240 pixels. In this set of data, letters “J” and “Z” are excluded from our scope as they contain motion.

Please find the training process in the project report:
[📄 View Project Report](AML Group 15 Final Report.pdf)