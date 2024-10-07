import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
model=load_model('BrainTumor10EpochsCategorical.h5')
image=cv2.imread(r'C:\Users\salim\OneDrive\Bureau\deep learning project 2\datasets\no\no2.jpg')
img =Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)
input_img=np.expand_dims(img,axis=0)
predictions = model.predict(input_img)
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
if predicted_classes == 0:
    message="No Brain Tumor Has Been Detected"
elif predicted_classes == 1:
    message="Brain Tumor Has Been Detected"
print("\033[1;31m{}\033[m".format(message))
