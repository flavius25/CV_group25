import tensorflow as tf
from keras import layers
from keras.callbacks import TensorBoard, LearningRateScheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import efficientnet.keras as efn 
from utils import *
from keras.models import Sequential

""" Global variables """
IMG_SIZE = (224,224)
EPOCHS = 50
BATCH_SIZE = 64
NUM_CLASSES = 40 # 40 classes in the Stanford40 dataset


""" Load dataset """
#Load the dataset which has already been preprocessed, set needDirectories to False if SF_train, SF_test, and SF_validation dirs already exist
train_ds, test_ds, val_ds, train_labels, test_labels, validation_labels, class_names = loadSF40(img_size =  IMG_SIZE, needDirectories=False)


#no of train and validation images
NO_TRAIN_IMGS = len(train_labels)
NO_VAL_IMGS = len(validation_labels)


#https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
#http://man.hubwiz.com/docset/TensorFlow_2.docset/Contents/Resources/Documents/tf/data/Dataset.html 


""" One-hot encoding """

def onehot_encoding(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

#Apply one hot encoding to the different datasets
train_ds = train_ds.map(onehot_encoding, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(onehot_encoding, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(onehot_encoding)
test_ds = test_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)

print("Done one-hot encoding!")

""" Build & Train Stanford40 Model EfficientNet"""

# Define the input and output layers of the model 
inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

#Fetch image augmentation layers and apply to inputs
img_augmentation = dataAugmentation()
x = img_augmentation(inputs)

outputs = efn.EfficientNetB0(include_top=True, weights=None, classes=class_names)(x) #Include top set to true as we are not using pre-trained weights but training from scratch

# Initialise the model. Compule and show summary. 
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

#Fit the model with the data augmentation and normalisation iterators
history = model.fit(train_ds, epochs = EPOCHS, steps_per_epoch = NO_TRAIN_IMGS//BATCH_SIZE, 
                              validation_data = val_ds, validation_steps=NO_VAL_IMGS//BATCH_SIZE, 
                              verbose=1, use_multiprocessing=True, workers=4)

#Save weights as we need these for transfer learning 
model.save_weights("model_weights.h5")



#plotAccuracy("title", train_acc, val_acc)                     # made these plotting functions in utils for hassle-free plotting
#plotLoss("title", train_loss, val_loss)

""" Building transfer learning model """

TL_model = 4


TL_model.load_weights("model_weights.h5")





