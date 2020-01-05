import numpy as np
import os
import cv2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle

# Get the compiled and embedded data
data = np.load('face_embeddings_1.npz')
trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# Normilize the vectors (good practise)
enc = Normalizer(norm='l2')
trainX = enc.transform(trainX)
testX = enc.transform(testX)

# Change the labels to integers using scikit-learn
enc_out = LabelEncoder()
enc_out.fit(trainY)
print(enc_out.classes_)
trainY = enc_out.transform(trainY)
testY = enc_out.transform(testY)

# Create the linear SVM model using SVC class in scikit-learn; kernel=linear, probability=True
# this type of model is usually used when dealing with normalized and embedded faces
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)

# The evaluatoion of the model
predictions = model.predict(testX)
score = accuracy_score(testY, predictions)

print("The accuracy on test set was ["+str(score)+"]")

# Save the model to path
path = "VIDEOS/MODELS/face_model_1.pkl"
with open(path, 'wb') as f:
    pickle.dump(model, f)
