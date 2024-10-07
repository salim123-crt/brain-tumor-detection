# import cv2
# import os
# import tensorflow as tf
# from tensorflow import keras
# from PIL import Image
# import numpy as np
# from sklearn.model_selection import train_test_split
# from keras.utils import normalize
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation , Dropout, Flatten, Dense
# from keras.optimizers import Adam
# from keras.utils import to_categorical
# import matplotlib.pyplot as plt
# image_directory='datasets/'
# no_tumour_images =os.listdir(image_directory+ 'no/')
# yes_tumour_images =os.listdir(image_directory+ 'yes/')
# dataset=[]
# label=[]
# INPUT_SIZE=64
# for i , image_name in enumerate (no_tumour_images):
#     if (image_name.split('.')[1]=='jpg'):
#         image=cv2.imread(image_directory +'no/'+image_name)
#         image=Image.fromarray(image,'RGB')
#         image=image.resize((INPUT_SIZE,INPUT_SIZE))
#         dataset.append(np.array(image))
#         label.append(0)
        
# for i , image_name in enumerate (yes_tumour_images):
#     if (image_name.split('.')[1]=='jpg'):
#         image=cv2.imread(image_directory +'yes/'+image_name)
#         image=Image.fromarray(image,'RGB')
#         image=image.resize((INPUT_SIZE,INPUT_SIZE))
#         dataset.append(np.array(image))
#         label.append(1)       

# dataset=np.array(dataset)
# label=np.array(label)
# x_train, x_test, y_train , y_test=train_test_split(dataset, label , test_size=0.2)
# plt.imshow(x_train[3])

# x_train=normalize(x_train, axis=1)
# x_test=normalize(x_test, axis=1)
# y_train=to_categorical(y_train , num_classes=2)
# y_test=to_categorical(y_test , num_classes=2)

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:4]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
# model.add(Activation('relu')) 
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2))
# model.add(Activation('softmax'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=16, 
# verbose=1, epochs=10, 
# validation_data=(x_test, y_test),
# shuffle=False)
# model.save('brainTumor10EpochsCategorical.h5')



import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

# Définition du répertoire contenant les images
image_directory = 'datasets/'

# Chargement des noms des fichiers d'images
no_tumour_images = os.listdir(image_directory + 'no/')
yes_tumour_images = os.listdir(image_directory + 'yes/')

# Initialisation des listes pour stocker les données et les labels
dataset = []
label = []
INPUT_SIZE = 64

# Chargement des images sans tumeur
for i, image_name in enumerate(no_tumour_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

# Chargement des images avec tumeur
for i, image_name in enumerate(yes_tumour_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

# Conversion en numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# Séparation des données d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2)

# Normalisation des données d'image
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Conversion des labels en catégories
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Définition du modèle CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# Compilation du modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test))

# Sauvegarde du modèle
model.save('brainTumor10EpochsCategorical.h5')

# Affichage des graphiques d'accuracy et de loss
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

