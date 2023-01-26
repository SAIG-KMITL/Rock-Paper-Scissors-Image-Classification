import cv2

x, y, w, h = 334, 71, 140, 140

model_path = "model/modelTrue/model2/model2.json"
model_weights_path = "model/modelTrue/model2/best_weights.h5"

BG_path = "src/BG.png"
FRM_path = "src/frame.png"

rectangle_color = (153, 76, 0)
text_color = (153, 76, 0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

computer_gestures = {
    "rock":     "src/rockk.png",
    "paper":    "src/paperr.png",
    "scissors": "src/scissorss.png"
}

stronger_gesture = {
    "rock":  "paper",
    "paper": "scissor",
    "scissors": "rock"
}