import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# ------------------ SETTINGS ------------------
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")  # Make sure these paths are correct
offset = 20
imgSize = 300
labels = ["A", "B", "C", "hello"]  # Match your model classes

# ------------------ MAIN LOOP ------------------
while True:
    success, img = cap.read()
    if not success:
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # --- SAFE CROP ---
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue  # Skip if crop is invalid

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        # --- RESIZE AND CENTER ---
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # --- PREDICTION ---
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # --- DISPLAY ---
        cv2.rectangle(imgOutput, (x1, y1 - 60), (x1 + 150, y1), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)

        # Optional: show intermediate images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
