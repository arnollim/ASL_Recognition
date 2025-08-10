import cv2

from annotator import Annotator
from detector import HandDetector
from model import SignLanguageModel

def run():
    detector = HandDetector()
    model = SignLanguageModel()
    annotator = Annotator(model)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read() #cv2.VideoCapture.read() method in OpenCV is used to read frames from a video stream
        # or a camera. It is a fundamental function for processing video data in real-time or from pre-recorded files.
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
        handsDetected = detector.detect_hands(image) #handsDetected was previously "results"

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)#convert back from RGB to BGR

        #multi_hand_landmarks: Collection of detected/tracked hands, where each hand is represented as a list of 21
        # hand landmarks and each landmark is composed of x, y and z.
        if handsDetected.multi_hand_landmarks:
            for hand_landmarks in handsDetected.multi_hand_landmarks:
                image = annotator.annotate(hand_landmarks, image)

        cv2.imshow('Sign Language Recogition', image)  # 9 Aug 2025 - Arnol added this here instead of above. It works!
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()