import cv2
import cvzone
import numpy as np
import matplotlib.image as mpimg
import math

from method import Gesture
from method import RockPaperScissors

from constants import x, y, w, h
from constants import model_path, model_weights_path
from constants import rectangle_color, text_color
from constants import computer_gestures
from constants import BG_path
from constants import FRM_path
from constants import font

from tensorflow import keras
from keras.models import model_from_json
from keras.utils import img_to_array




# (1)1280×720 
# (2)1366×768 default
# (3)1600×900 
# (4)1920×1080 

with open('config.txt') as f:
    contents = f.readlines()
    con = contents[0].split(" ")
    numscale = int(con[4].replace("\n", ""))

if(numscale == 1):
    scale = 0.9375    
elif(numscale == 3):
    scale = 1.171875  
elif(numscale == 4):
    scale = 1.40625  
else:
    scale = 1

class GestureModel:
    def __init__(self, model_path_, model_weights_path_):
        self.model = model_from_json(open(model_path_,"r").read())
        self.model.load_weights(model_weights_path_)
        self.gestures = ('empty', 'paper', 'rock', 'scissors')

    @classmethod
    def preprocess(cls, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        handarea = gray_frame[math.ceil(y*scale):math.ceil((y + w)*scale), math.ceil(x*scale):math.ceil((x + h)*scale)]
        # cv2.imshow("test", handarea)
        handarea = cv2.resize(handarea, (50, 50))
        img_pixels = img_to_array(handarea)
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


class webCam:
    model = GestureModel(model_path,
                         model_weights_path)         

    @classmethod
    def create_rectangle(cls, frame):
        cv2.rectangle(img=frame,
                      pt1=(math.ceil(x*scale), math.ceil(y*scale)), pt2=(math.ceil((x+w)*scale), math.ceil((y+h)*scale)),
                      color=rectangle_color, thickness=5)

    @classmethod
    def create_text(cls, frame,
                    text, font_scale=2,
                    thickness=2, org=(x, y),
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
            imgBG = cv2.resize(imgBG, (math.ceil(1366*scale),math.ceil(768*scale)))

            imgFRM = cv2.imread(FRM_path, cv2.IMREAD_UNCHANGED)
            # imgFRM = cv2.resize(imgFRM, (732,412)) 
            imgFRM = cv2.resize(imgFRM, (math.ceil(695*scale),math.ceil(423*scale))) 

            ret, frame = cap.read()
            frame = cv2.resize(frame,(0,0),None,379*scale/720, 379*scale/720)
            379*scale

            gesture, percent = cls.model.predict(frame)
            cls.create_rectangle(frame)

            frame = cv2.flip(frame, 1)
            
            if not gesture == "empty":
                frames_elapsed += 1
                if frames_elapsed > 5:
                    if hand_exited == 0:
                        hand_exited += 1

                    person_gesture = Gesture(gesture)
                    image = cv2.imread(computer_gestures[computer_gesture.name], cv2.IMREAD_UNCHANGED)
                    image = cv2.resize(image, (math.ceil(284*scale), math.ceil(284*scale)))
                    imgBG = cvzone.overlayPNG(imgBG, image, (math.ceil(960*scale), math.ceil(220*scale)))

                    result = RockPaperScissors.get_result(person_gesture, computer_gesture)

                    if hand_in_screen == 0:
                        hand_in_screen += 1
                        result_ = result[1]

                    else:
                        cls.create_text(imgBG, f"{result_}", org=(math.ceil(620*scale), math.ceil(640*scale)), color=(0, 128, 255), font_scale=3*scale, thickness=5)
                        cls.create_text(frame, f"{gesture} {percent}%", org=(math.ceil(181*scale), math.ceil(57*scale)),font = cv2.FONT_HERSHEY_DUPLEX ,font_scale= 0.8*scale)

                    frames_elapsed += 1
            else:
                computer_gesture = Gesture.generate_random()
                hand_in_screen = 0
                frames_elapsed = 0
                rounds = rounds_

            cls.create_text(frame, f"frames: {frames_elapsed}", org=(math.ceil(25*scale), math.ceil(370*scale)),font = cv2.FONT_HERSHEY_DUPLEX ,color=(255, 255, 255),
                            font_scale=0.8*scale, thickness=2)                      
                             
            imgBG[math.ceil(182*scale):math.ceil(561*scale), math.ceil(107*scale):math.ceil(781*scale)] = frame   # y1:y2 , x1:x2
            imgBG = cvzone.overlayPNG(imgBG, imgFRM, (math.ceil(95*scale), math.ceil(165*scale)))      
            
            cv2.imshow('SAIG-Rock Paper Scissors!', imgBG)      
            # cv2.imshow('Rock Paper Scissors!', frame)
            
            if cv2.waitKey(10) == ord('q'):
                break