import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, LearningRateScheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#from keras.applications import EfficientNetB0
import efficientnet.keras as efn 
from keras.models import Sequential
from utils import * 


#Global variables
IMG_SIZE = (224,224)
EPOCHS = 50
BATCH_SIZE = 64

#Load the dataset which has already been preprocessed
SF_training_set, train_labels, SF_test_set, test_labels, class_names = loadSF40(img_size =  IMG_SIZE)
print("Here!", train_labels[0])


#Split the trainingset to obtain 10% stratified validation set
train_images, validation_images, train_labels, validation_labels = train_test_split(SF_training_set, train_labels, test_size=0.1, random_state=0, stratify=train_labels)


"""   Data augmentation and Preprocessing """
# data augmentation generator defining the augmentations and data-preprocessing to be made
data_generator = ImageDataGenerator(
        rescale=1.0/255.0, #normalising pixel values to range 0-1
        rotation_range=20, # rotation
        width_shift_range=0.2, # horizontal shift
        height_shift_range=0.2, # vertical shift
        zoom_range=0.2, # zoom
        horizontal_flip=True, # horizontal flip
        brightness_range=[0.5,1.2]  # brightness
        )

#Fit training_set to generator to calculate statistics 
data_generator.fit(SF_training_set)

#Create iterators to pass to the model during training
train_iterator = data_generator.flow(train_images, train_labels, batch_size=64)
validation_iterator =  data_generator.flow(validation_images, validation_labels, batch_size=64)
test_iterator = data_generator.flow(SF_test_set, test_labels, batch_size=64)


""" This is how we will fit it with the model """
# fit model with generator
# model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=5)
# # evaluate model
# _, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)




# inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# #x = img_augmentation(inputs)
# outputs = efn.EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)#(x)

# model = tf.keras.Model(inputs, outputs)
# model.compile(
#     optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
# )

# model.summary()

# epochs = 40  # @param {type: "slider", min:10, max:100}
# hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)









