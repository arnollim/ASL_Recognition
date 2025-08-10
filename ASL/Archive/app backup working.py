from detector import HandDetector
from model import SignLanguageModel
import cv2
import mediapipe as mp
import numpy as np
from skimage.transform import resize

alpha_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N',
          14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles

size = 75
padding = 40
partsOfHand = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
ratio_of_hand_to_circle = 0.05

xmin = 0
xmax = 0
ymin = 0
ymax = 0

diff = 10

def run():
    model = SignLanguageModel('final_model.json', 'final_model.h5')
    detector = HandDetector()

    hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_hight, image_width, _ = image.shape

        if results.multi_hand_landmarks:
            # For feeding in black background with annotations
            black = np.zeros([image_hight, image_width, 3], dtype=np.uint8)
            black.fill(0)

            for hand_landmarks in results.multi_hand_landmarks:
                xmin = image_width - hand_landmarks.landmark[0].x * image_width  #
                xmax = image_width - hand_landmarks.landmark[0].x * image_width  # image_width -
                ymin = hand_landmarks.landmark[0].y * image_hight
                ymax = hand_landmarks.landmark[0].y * image_hight

                for i in range(1, len(partsOfHand)):
                    x = image_width - hand_landmarks.landmark[i].x * image_width  # image_width -
                    y = hand_landmarks.landmark[i].y * image_hight

                    if x < xmin:
                        xmin = x
                    elif x > xmax:
                        xmax = x
                    if y < ymin:
                        ymin = y
                    elif y > ymax:
                        ymax = y

                if ymax - ymin < xmax - xmin:
                    max = xmax - xmin
                    diff = xmax - xmin + padding
                else:
                    max = ymax - ymin
                    diff = ymax - ymin + padding

                xstart = int((xmax - xmin) / 2 + xmin - diff / 2)
                xstop = int((xmax - xmin) / 2 + xmin + diff / 2)
                ystart = int((ymax - ymin) / 2 + ymin - diff / 2)
                ystop = int((ymax - ymin) / 2 + ymin + diff / 2)

                if ystart < 0:
                    ystart = 0
                if xstart < 0:
                    xstart = 0
                if ystop > image_hight:
                    ystop = int(image_hight)
                if xstop > image_width:
                    xstop = int(image_width)

                # Change radius of circle according to ratio of hand
                for k, v in drawing_styles._HAND_LANDMARK_STYLE.items():
                    for landmark in k:
                        v.circle_radius = int(ratio_of_hand_to_circle * max)

                # Change thickness of hand connection
                for k, v in drawing_styles._HAND_CONNECTION_STYLE.items():
                    for connection in k:
                        # if k == drawing_styles._PALM_CONNECTIONS:
                        # v.thickness = int(0.02*max)
                        # else:
                        v.thickness = int(0.02 * max)
                        if v.thickness == 0:
                            v.thickness = 1

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    # drawing_styles.get_default_hand_landmark_style(),
                    # drawing_styles.get_default_hand_connection_style())
                )

                mp_drawing.draw_landmarks(
                    black, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    # drawing_styles.get_default_hand_landmark_style(),
                    # drawing_styles.get_default_hand_connection_style())
                )

            #START: IF YOU TAB THIS CHUNK, it will detect multiple hands

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            final_image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            final_black = cv2.cvtColor(cv2.flip(black, 1), cv2.COLOR_BGR2RGB)
            # final_image = cv2.flip(image, 1)
            # final_black = cv2.flip(black, 1)

            # Change final_image/final_black variable depending on which is used for model prediction
            crop_img = resize(final_black[ystart:ystop, xstart:xstop], (100, 100, 3))
            # plt.imshow(crop_img)
            # plt.show()

            image = cv2.rectangle(final_image, (xstart, ystart), (xstop, ystop), (36, 255, 12), 1)
            # Predict the alphabet and show the alphabet in the video
            cv2.putText(image, HandDetector.detect_hands(),
                        (xstart, ystart), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #END: IF YOU TAB THIS CHUNK, it will detect multiple hands


            # cv2.imshow('Sign Language Recogition', image)
        # if cv2.waitKey(5) & 0xFF == 27:
        cv2.imshow('Sign Language Recogition', image)  # 9 Aug 2025 - Arnol added this here instead of above. It works!
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()