import cv2

x, y, w, h = 470, 100, 170, 170

model_path = "Testmodel/model2/model2.json"
model_weights_path = "Testmodel/model2/best_weights.h5"

BG_path = "src/BG.png"

rectangle_color = (153, 76, 0)
text_color = (153, 76, 0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

computer_gestures = {
    "rock":     "src/rock.png",
    "paper":    "src/paper.png",
    "scissors": "src/scissors.png"
}

stronger_gesture = {
    "rock":  "paper",
    "paper": "scissor",
    "scissors": "rock"
}