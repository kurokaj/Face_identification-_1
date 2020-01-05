# Face_identification-_1
First time developing face identification model using opencv, keras and sklearn. The project is compilation of different tips from online and personal skills developed during previous projects and school work. 

# Reguirements:
  - Python 3.7+
    - OpenCV2
    - Numpy
    - Pickle
    - Keras
    - tqdm
    - sklearn
    - Jupyter notebook

# Object
The aim of this project was to get the first touch to face recognition and identification. The project consist of three parts: 
  1) Generate data
  2) Adjust the collected data and create model
  3) Test the model using the the laptop's camera
 
1) I received 30second videos from 5 friends where they turn their head and make different faces to create more versatile data. These videos were then run through a pre-trained HAAR-cascade classifier to be able to extract the face images. Approximately 500 images were generated per face/label. The data generation was made in jupyter notebook file as I initially started testing the HAAR- classifier in that environment. 
2) In the compile_data.py script the data is divided and manipulated simpler for the future classifier to  use. The face images were turned into unique face embeddings using the pre-trained FaceNet model downloaded online. These embeddings are used to train and test SVM-model with five labels. With the validation set the classifier was able to reach over 95% accuracy.
3) Final testing was performed using the laptops camera. The trained model was able to identify rigth person with really high accuracy. When person got further away from the camera (the face became more pixelated) the accuracy dropped as well. As the model only has five labels it of course misslabeled people outside the database. To this project I could not determine a way to label "outsiders" as unknown. For the future project I'll try to use certain facial points and landmarks to identify the right person. Using some sort of distances between the person to be identified and database could be the answer to be able to identify people as unknown. 




  
    
