import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, LearningRateScheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#from keras.applications import EfficientNetB0
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


def input_preprocessing():
    somehthing = 4




#Plotting a subset of data with labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()







