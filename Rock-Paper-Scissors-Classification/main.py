from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

import cv2
import numpy as np
import PILasOPENCV as Image
import PILasOPENCV as ImageDraw
import PILasOPENCV as ImageFont

from gameMain import Gesture
from gameMain import RockPaperScissors

from constants import x, y, w, h
from constants import model_path, model_weights_path
from constants import rectangle_color, text_color
from constants import computer_gestures
from constants import BG_path
from constants import font




class GestureModel:
    def __init__(self, model_path_, model_weights_path_):
        self.model = model_from_json(open(model_path_,"r").read())
        self.model.load_weights(model_weights_path_)
        self.gestures = ('empty', 'paper', 'rock', 'scissors')

    @classmethod
    def preprocess(cls, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray_frame[y:y + w, x:x + h]
        roi = cv2.resize(roi, (50, 50))
        img_pixels = img_to_array(roi)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0
        return img_pixels

    def predict(self, frame):
        img_pixels = self.preprocess(frame)
        prediction = self.model.predict(img_pixels)
        max_index = np.argmax(prediction[0])

        predicted_gesture = self.gestures[max_index]
        predict_percent = prediction[0][max_index]*100
        return predicted_gesture, round(predict_percent, 2)


class WebCam:
    model = GestureModel(model_path,
                         model_weights_path)

    @classmethod
    def create_rectangle(cls, frame):
        cv2.rectangle(img=frame,
                      pt1=(x, y), pt2=(x+w, y+h),
                      color=rectangle_color, thickness=5)

    @classmethod
    def create_text(cls, frame,
                    text, font_scale=2,
                    thickness=2, org=(x-w-10, y),
                    color=text_color, font=font):

        cv2.putText(img=frame, text=text,
                    org=org,
                    fontFace=font, fontScale=font_scale,
                    color=color, thickness=thickness)

    @classmethod
    def play_game(cls):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        computer_gesture = Gesture.generate_random()
        # scores = [0, 0, 0]
        hand_in_screen = 0
        hand_exited = 0
        result_ = ""
        frames_elapsed = 0
        rounds = 0
        rounds_ = 0

        while cap.isOpened():

            imgBG = cv2.imread(BG_path)
            ret, frame = cap.read()
            frame = cv2.resize(frame,(0,0),None,0.704, 0.704)

            gesture, percent = cls.model.predict(frame)
            cls.create_rectangle(frame)

            frame = cv2.flip(frame, 1)
            
            if not gesture == "empty":
                frames_elapsed += 1
                if frames_elapsed > 5:
                    if hand_exited == 0:
                        hand_exited += 1

                    person_gesture = Gesture(gesture)
                    image = cv2.imread(computer_gestures[computer_gesture.name])
                    image = cv2.resize(image, (400, 400))
                    x_offset, y_offset = (1360, 310)

                    imgBG[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image

                    result = RockPaperScissors.get_result(person_gesture, computer_gesture)

                    if hand_in_screen == 0:
                        # scores[result[0]] += 1
                        hand_in_screen += 1
                        result_ = result[1]
                        rounds_ += 1

                    else:
                        cls.create_text(imgBG, f"{result_}", org=(895, 900), color=(0, 128, 255), font_scale=2)
                        cls.create_text(frame, f"{gesture} {percent}%", org=(250, 80),font = cv2.FONT_HERSHEY_DUPLEX ,font_scale=1)

                    frames_elapsed += 1
            else:
                computer_gesture = Gesture.generate_random()
                hand_in_screen = 0
                frames_elapsed = 0
                rounds = rounds_

            cls.create_text(frame, f"frames: {frames_elapsed}", org=(10, 500),font = cv2.FONT_HERSHEY_DUPLEX ,color=(255, 255, 255),
                            font_scale=1, thickness=2)
            # cls.create_text(frame, f"Round: {rounds}", org=(150, 50), color=(255, 0, 0))
            # cls.create_text(frame, f"Person: {scores[0]}", org=(100, 310), color=(255, 0, 0))
            # cls.create_text(frame, f"Computer: {scores[1]}", org=(350, 310), color=(255, 0, 0))           
                      
            imgBG[268:775, 175:1076] = frame   # y1:y2 , x1:x2
            cv2.imshow('BG', imgBG)      
            # cv2.imshow('Rock Paper Scissors!', frame)
            
            if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
                break

    # @classmethod
    # def start(cls):
    #     cap = cv2.VideoCapture(0)
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             continue
    #         flipped_frame = cv2.flip(frame, 1)
    #         resized_frame = cv2.resizecv2.resize(flipped_frame,(0,0),None,0.458,0.458)
    #         cv2.imshow('Rock Paper Scissors!', resized_frame)
    #         if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
    #             break

    #     cap.release()
    #     cv2.destroyWindow()