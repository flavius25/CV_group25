import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import layers
from keras.callbacks import TensorBoard, LearningRateScheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import efficientnet.keras as efn 
from keras.models import Sequential
from keras import layers
from utils import *

""" Global variables """
IMG_SIZE = (224,224)
EPOCHS = 50
BATCH_SIZE = 64

needDataDirs = True # set to False if images already sorted

utils.createDataDirectories(needDataDirs)

""" Load dataset """
#Load the dataset which has already been preprocessed
SF_training_set,  SF_test_set = utils.loadSF40(img_size =  IMG_SIZE)

#Split the trainingset to obtain 10% stratified validation set
train_images, validation_images, train_labels, validation_labels = train_test_split(SF_training_set, train_labels, test_size=0.1, random_state=0, stratify=train_labels)

#no of train and validation images
NO_TRAIN_IMGS = len(train_labels)
NO_VAL_IMGS = len(validation_labels)

#Get iterator for data augmentation and 
train_iterator = getIterator(train_images, train_labels)
validation_iterator =  getIterator(validation_images, validation_labels)
test_iterator = getIterator(SF_test_set, test_labels)

""" This is how we will fit it with the model """
# fit model with generator
# model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=5)
# # evaluate model
# _, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)

# https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/ 

""" One-hot encoding """

def onehot_encoding(image, label):
    label = tf.one_hot(label, class_names)
    return image, label


train_images = train_images.map(
    onehot_encoding, num_parallel_calls=tf.data.AUTOTUNE)

train_images = train_images.batch(batch_size=BATCH_SIZE, drop_remainder=True)
train_images = train_images.prefetch(tf.data.AUTOTUNE)

SF_test_set = SF_test_set.map(onehot_encoding)
SF_test_set = SF_test_set.batch(batch_size=BATCH_SIZE, drop_remainder=True)

""" Build & Train Stanford40 Model EfficientNet"""

# Define the input and output layers of the model 
inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

outputs = efn.EfficientNetB0(include_top=True, weights=None, classes=class_names) #Include top set to true as we are not using pre-trained weights but training from scratch

# Initialise the model. Compule and show summary. 
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

#Fit the model with the data augmentation and normalisation iterators
history = model.fit_generator(train_iterator, epochs = EPOCHS, steps_per_epoch = NO_TRAIN_IMGS//BATCH_SIZE, 
                              validation_data = validation_iterator, validation_steps=NO_VAL_IMGS//BATCH_SIZE, 
                              verbose=1, use_multiprocessing=True, workers=4)

#Save weights as we need these for transfer learning 
model.save_weights("model_weights.h5")


""" Building transfer learning model """

TL_model = 4


TL_model.load_weights("model_weights.h5")





