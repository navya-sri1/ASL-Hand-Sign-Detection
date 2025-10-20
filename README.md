🖐️ Hand Gesture Recognition using OpenCV and Mediapipe


📘 Overview


This project is a Hand Gesture Recognition System built using Python, OpenCV, mediapipe and CVZone.
It allows you to collect hand gesture images and then recognize gestures in real time using a trained deep learning model.

🚀 Features

Real-time hand detection using cvzone.HandTrackingModule

Image dataset collection with adjustable bounding box

Gesture classification using a pre-trained Keras model

Modular design for easy customization (add more gestures or retrain model)



🧩 Project Structure

├── Data/                  # Folder for collected gesture images

│   ├── hello/             # Example gesture folder

│

├── Model/

│   ├── keras_model.h5     # Trained gesture classification model

│   ├── labels.txt         # Class labels for the model

│

├── dataCollection.py      # Script to collect gesture data

├── test.py                # Script to test and classify gestures

└── README.md              # Project documentation




⚙️ Installation

Clone the repository

git clone https://github.com/<your-username>/Hand-Gesture-Recognition.git
cd Hand-Gesture-Recognition


Install dependencies

pip install opencv-python cvzone numpy tensorflow


Create folder structure

Inside your project directory, create:

Data/
Model/


Add your trained model files inside the Model folder:

keras_model.h5

labels.txt

🎥 Data Collection

Run the following to capture and store hand gesture images:

python dataCollection.py


Press s to save an image of your hand gesture.

Images will be saved inside the folder defined by the folder variable (e.g., Data/hello).

🧠 Testing & Classification

Once your model is trained and placed inside Model/, run:

python test.py


Shows the live camera feed.

Detects hand region, preprocesses it, and classifies the gesture.

Displays the recognized label on the screen.

Press q to quit.

🧑‍💻 Requirements

Python 3.8+

OpenCV

cvzone

numpy

tensorflow / keras

Install all at once:

pip install -r requirements.txt

🧱 How It Works

Uses cvzone.HandTrackingModule to detect hand landmarks.

Crops and resizes the hand image to a fixed 300x300 pixel area.

Feeds the processed image to the trained Keras model.

Displays the predicted gesture label on screen.

🏗️ Future Improvements

Add more gesture classes

Improve accuracy using custom CNN models

Build a Streamlit or Tkinter GUI interface for user interaction

Architecture:


![WhatsApp Image 2025-10-20 at 17 30 23_bda79132](https://github.com/user-attachments/assets/5cec420f-9d8b-4c5d-bdec-a26a2b73cb89)

