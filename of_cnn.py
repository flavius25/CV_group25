import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from keras.applications.efficientnet import EfficientNetB0
from keras.models import Model, Sequential
from keras.layers import TimeDistributed, Input, MaxPooling3D, LSTM, Dense, Conv3D, Conv2D, BatchNormalization, Flatten, Dropout, Convolution2D, Activation, MaxPooling2D, GlobalAveragePooling2D
from keras import losses
from utils import *

""" Global variables"""
NUM_CLASSES = 4
EPOCHS = 15
INSTACK = 16
IMG_SIZE = (224, 224)

""" Load train generators """

train_gen, val_gen, test_gen = loadOFdatagens()

# """ Optical flow model """

# def build_model(inshape=(16, 224, 224, 3), NUM_CLASSES=4):
#   model = Sequential()
#   model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation='relu', padding='same'), input_shape=(16, 224, 224, 3)))
#   model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer="he_normal", activation='relu')))
#   model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
  
#   model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
#   model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
#   model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
  
#   model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
#   model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
#   model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
  
#   model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
#   model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
#   model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

#   model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
#   model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
#   model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
  
#   model.add(TimeDistributed(Flatten()))
  
#   model.add(Dropout(0.5))
#   model.add(LSTM(256, return_sequences=False, dropout=0.5))
#   model.add(Dense(256, activation='relu'))
#   model.add(Dropout(0.5))
#   model.add(Dense(NUM_CLASSES, activation='softmax', name='pred'))

#   model.summary()
#   return model

def build_model(input_shape= (INSTACK,IMG_SIZE[0],IMG_SIZE[1],3),num_classes=NUM_CLASSES):
  
    model = Sequential()

    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=input_shape, padding='same', activation='relu'))
    model.add(Activation('relu'))

    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))

    model.add(Dropout(0.2))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))

    model.add(Dropout(0.2))    
    model.add(Flatten()) #Flatten layer
    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name='pred')) #Output layer


    model.compile(loss='categorical crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()

    return model
    
model = build_model()

""" Instantiating and compiling model """
of_model = build_model((INSTACK, IMG_SIZE[0], IMG_SIZE[1], 3), NUM_CLASSES)
of_model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=["accuracy"])
of_model.fit(train_gen,EPOCHS,validation_data=val_gen)