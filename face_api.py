import cv2
import numpy as np
import dlib
import os
from tqdm import tqdm
import pickle
from keras.models import load_model
import math

# For the output text
font = cv2.FONT_HERSHEY_SIMPLEX

# The labels
#AARNE - 0
#JANNE - 1
#JONNE - 2
#MERI  - 3
#VP    - 4

# Model = the daceNet model; image = face to be embedded
def make_embedding(model, image):
    # Standardize the pixels accross image
    pixels = image.astype('float32')
    m = pixels.mean()
    std = pixels.std()
    st_image = (pixels-m)/ std
    # Make one face to one sample
    sample = np.expand_dims(st_image, axis=0)
    # Get the embedding
    temp = model.predict(sample)
    ret = temp[0]
    return ret


# The pre-trained model to detect face
cascPath = "/home/jonne/anaconda3/envs/face_reg/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
faceC = cv2.CascadeClassifier(cascPath)

video_c = cv2.VideoCapture(0)

# Load the id-model
with open('VIDEOS/MODELS/face_model_1.pkl', 'rb') as f:
    model = pickle.load(f)
print("SVC- model loaded...")

# Load the FaceNet- model
FaceNet_model = load_model("facenet_keras.h5")
print("FaceNet- model loaded...")

while True:
    # frame by frame
    ret, frame = video_c.read()
    # linear color transition to grey scale
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create list of the faces recognized
    face_list = faceC.detectMultiScale(
        im,
        scaleFactor=1.2,
        minNeighbors=10,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around the recognized faces and crop own face to a directory
    for (x,y,w,h) in face_list:
        cv2.rectangle(frame, (x-3,y-3), (x+w+3,y+h+3), (0,0,255), 3)

        roi_color = frame[y:y+h, x:x+w]  # roi with normal color
        resized = cv2.resize(roi_color, (160,160), interpolation = cv2.INTER_AREA)

        emp = make_embedding(FaceNet_model, resized)

        sample = np.expand_dims(emp, axis=0)

        # Input the face embeddings to the model
        clas = model.predict(sample)
        prob = model.predict_proba(sample)
        # prob includes all probabilities accross the labels
        prob = math.floor(prob[0, clas[0]] * 100)

        if clas == 0:
            clas = "AARNE"
        elif clas == 1:
            clas = "JANNE"
        elif clas == 2:
            clas = "JONNE"
        elif clas == 3:
            clas = "MERI"
        elif clas == 4:
            clas = "VP"

        # Output the label and probabilities predicted by the model
        cv2.putText(frame, clas,(x,y-15),font,1,(255,0,0),2)
        cv2.putText(frame, str(prob),(x+w-10,y-15),font,1,(255,0,0),2)

    # Show the video frame
    cv2.imshow('Live', frame)

    # Stop the camera when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kill windows and release captured video
video_c.release()
cv2.destroyAllWindows()
