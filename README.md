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

## 👥 Authors

- Lim Pek Liang Arnol: [@arnollim](https://github.com/arnollim)
- Wong Chun Keet Brian: [@Brian-Wong](https://github.com/Brian-Wong)
- Wang Zhifei Celia [@teammate2](https://github.com/teammate2)
- Tan Yan Ru [@teammate3](https://github.com/teammate3)
- Wong Sook Xian Sophia [@teammate4](https://github.com/teammate4)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/arnollim/ASL_Recognition.git
cd ASL_Recognition
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
pip install -r requirements.txt
```

### 4. Place the model file (.h5) and model weighs file (.json) into the /model directory
```sql
models/
├── final_model.json
└── final_model.h5
```

## Usage

Start the application by running app.py in the /ASL folder
```bash
cd ASL
python app.py
```
## Training
The model was trained based on a Kaggle dataset (Muvezwa, 2019):
https://www.kaggle.com/datasets/kuzivakwashe/significant-asl-sign-language-alphabet-dataset
This dataset consists of over 70,000 coloured, RGB images with a resolution of 320x240 pixels. In this set of data, letters “J” and “Z” are excluded from our scope as they contain motion.

Please find the training process in the project report:
[📄 View Project Report](AML Group 15 Final Report.pdf)