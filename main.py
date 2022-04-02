import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, LearningRateScheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from matplotlib.pyplot import figure
from keras.applications import EfficientNetB0
from utils import * 


#Global variables
img_size = (224,224)
epochs = 50
batch_size = 64

#Load the dataset which has already been preprocessed
SF_training_set, train_labels, SF_test_set, test_labels = loadSF40()

#Split the trainingset to obtain 10% stratified validation set
train_images, validation_images, train_labels, validation_labels = train_test_split(SF_training_set, train_labels, test_size=0.1, random_state=0, stratify=train_labels)