# sign_lang_recog/model.py
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

alpha_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N',
          14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}
class SignLanguageModel:
    def __init__(self, model_json_path='../models/final_model.json', weights_path='../models/final_model.h5'):
        self.model = self.load_model(model_json_path, weights_path)

    def load_model(self, model_json_path, weights_path):
        with open(model_json_path, 'r') as f:
            model_json = f.read()
        model = model_from_json(
            model_json,
            custom_objects={
                "Sequential": Sequential,
                "Conv2D": Conv2D,
                "MaxPooling2D": MaxPooling2D,
                "Flatten": Flatten,
                "Dense": Dense,
                "Dropout": Dropout,
            }
        )
        model.load_weights(weights_path)
        print("[INFO] Model loaded successfully.")
        return model

    def predict(self, crop_img):
        """
        Run prediction on preprocessed image.
        Assumes image shape matches model input (e.g., (1, 100, 100, 3))
        """
        #preds = self.model.predict(preprocessed_image)

        pred = alpha_dict[int(self.model.predict(np.expand_dims(crop_img, axis=0)).argmax(axis=1))]

        return pred