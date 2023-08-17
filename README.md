# Rock Paper Scissors Game with EfficientNet Image Classification

This project presents an interactive Rock Paper Scissors game that employs an image classification model based on the EfficientNet architecture. The game is enhanced with a graphical user interface (GUI) built using the cvzone library and OpenCV. Players can engage in a visually appealing experience as they challenge the computer's image classifier.

## Features
1. Graphical User Interface (GUI) powered by cvzone and OpenCV.
2. EfficientNet model for image classification to predict player's and computer's choices.
3. Real-time display of game results, including round outcomes and overall scores.
4. Intuitive hand gesture recognition using webcam feed.
5. Seamless integration of GUI elements for a captivating gaming experience

## Technologies Used
- Python programming language
- TensorFlow and Keras for building and training the EfficientNet model
- cvzone and OpenCV for creating the graphical user interface
- Command-line interface for player interaction
- Git version control for code management
- EfficientNet architecture for image classification

## How to Play
1. Clone this project: `https://github.com/MAOK-Yongsuk/Rock-Paper-Scissors-Image-Classification.git`
2. Install the required dependencies by running: `pip install -r requirements.txt`
3. Run the game script: `python Rock-Paper-Scissors-Game.py`
4. The GUI window will open, displaying game elements and webcam feed.
5. Make your hand gesture for rock, paper, or scissors in front of the webcam.
6. The computer will also make a choice using the image classification model.
7. Round results will be displayed on the GUI, updating the scores accordingly.
8. Keep playing rounds until you decide to exit the game.

## Model Training
The EfficientNet model utilized in this project was trained on a custom dataset of rock, paper, and scissors hand gesture images. The dataset was curated from diverse sources and underwent preprocessing to ensure uniform sizing and quality.
The model boasts an accuracy exceeding 99% on the validation set, establishing its reliability for accurately predicting player and computer choices during the game.

## Acknowledgments
We extend our gratitude to the creators of the EfficientNet architecture, the authors of the cvzone library, and the providers of the Rock Paper Scissors datasets used for training the image classification model.

## Future Enhancements
- Expansion of hand gesture classes to enhance game diversity.
- Integration of additional GUI elements to enrich the gaming interface.
- Implementation of adjustable difficulty levels for computer decision-making.
- Contributions to this project are welcomed – simply fork the repository and submit pull requests!

Note: The images and model employed in this project are for demonstrative purposes and may not yield peak accuracy in real-world scenarios.
