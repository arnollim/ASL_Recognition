# sign_lang_recog/detector.py

import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        """
        Detect hands and return preprocessed hand ROI(s) ready for model prediction.
        Also returns annotated frame with landmarks drawn.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return frame, []

