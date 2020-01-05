import numpy as np
import os
import cv2
from tqdm import tqdm


# To load the Hiroki Tanai's pretrained FaceNet- model
from keras.models import load_model

def get_faces(dir):

    faces = list()
    for file in os.listdir(dir):
        path = dir + file
        face = cv2.imread(path,1)
        face_array = np.asarray(face)
        faces.append(face_array)
    return faces

def load_data(dir):

    x, y = list(), list()
    for subdir in os.listdir(dir):
        path = dir + '/' + subdir + '/'
        faces = get_faces(path)
        labels = [subdir] * len(faces)
        print(">> Loaded "+str(len(faces))+" faces with label "+subdir+" from "+dir)
        x.extend(faces)
        y.extend(labels)
    return np.asarray(x),np.asarray(y)

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


# Get the training and testing data
trainX, trainY = load_data("/home/jonne/back_up/PROJEKTIT/FACE_R/VIDEOS/cropped_video/train")
testX, testY = load_data("/home/jonne/back_up/PROJEKTIT/FACE_R/VIDEOS/cropped_video/test")

# Create the face embeddings with FaceNet
FaceNet_model = load_model("facenet_keras.h5")
print("Model loaded...")

e_trainX = list()
for i in tqdm(trainX):
    embedding = make_embedding(FaceNet_model, i)
    e_trainX.append(embedding)
e_trainX = np.asarray(e_trainX)

e_testX = list()
for j in tqdm(testX):
    embedding = make_embedding(FaceNet_model, j)
    e_testX.append(embedding)
e_testX = np.asarray(e_testX)

np.savez_compressed('face_embeddings_1.npz', e_trainX, trainY, e_testX, testY)

print(e_trainX.shape)
print(e_testX.shape)
print(testY)
