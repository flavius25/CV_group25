import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
import matplotlib.pyplot as plt

""" Global variables """
IMG_SIZE = (224,224)
EPOCHS = 15
BATCH_SIZE = 32
NUM_CLASSES = 40 # 40 classes in the Stanford40 dataset


""" Load dataset """
#Load the dataset which has already been preprocessed, set needDirectories to False if SF_train, SF_test, and SF_validation dirs already exist
train_ds, test_ds, val_ds, train_labels, test_labels, validation_labels, class_names = loadSF40(img_size =  IMG_SIZE, needDirectories=False)


#no of train and validation images
NO_TRAIN_IMGS = len(train_labels)
NO_VAL_IMGS = len(validation_labels)

print("train: ",len(train_ds))
print("test: ",len(test_ds))
print("val: ",len(val_ds))
print(train_ds)
#https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
#http://man.hubwiz.com/docset/TensorFlow_2.docset/Contents/Resources/Documents/tf/data/Dataset.html 



# result = train_ds.map(lambda image, label: image)
# for image in result.take(2):
#     print(image.shape)

# def to_numpy(ds):
#   return list(ds.as_numpy_iterator())
# result = np.asarray(to_numpy(train_ds), dtype=object)

# result = train_ds.unbatch()
# #result = list(result.as_numpy_iterator())
# #print("Result: ", result.shape)
# print(result)
#
#
# images = np.asarray(list(result.map(lambda x, y: x)))
# labels = np.asarray(list(result.map(lambda x, y: y)))
#
# print(images.shape)
# print(labels.shape)

#we can do it as in line 57  and 58 +54+55 and line 48 to unbatch (IF still needed)
#more on that here: https://stackoverflow.com/questions/70535683/extract-data-from-tensorflow-dataset-e-g-to-numpy

# """ One-hot encoding """
#
# def onehot_encoding(image, label):
#     label = tf.one_hot(label, NUM_CLASSES)
#     return image, label
#
# #Apply one hot encoding to the different datasets
# train_ds = train_ds.map(onehot_encoding, num_parallel_calls=tf.data.AUTOTUNE)
# train_ds = train_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
# train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
#
# val_ds = val_ds.map(onehot_encoding, num_parallel_calls=tf.data.AUTOTUNE)
# val_ds = val_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
# val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
#
# test_ds = test_ds.map(onehot_encoding)
# test_ds = test_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
#
# print("Done one-hot encoding!")
#
""" Build & Train Stanford40 Model EfficientNet"""

# # Define the input and output layers of the model
# inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
#
img_augmentation = Sequential(
    [
        #layers.Rescaling(scale=1./255),   if we do normalization here , we cannot see the plot below
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomContrast(factor=0.1),
        layers.RandomZoom(0.1)
    ],
    name="img_augmentation",
    )
#Fetch image augmentation layers and apply to inputs
#img_augmentation = dataAugmentation(images,labels)
#print("Image aug: ", img_augmentation.shape)
#x = img_augmentation(inputs)


for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    print("IMage of 0: ",image[0].shape)
    plt.imshow(image[0].numpy().astype("uint8"))
    #plt.title("{}".format(format_label(label)))
    plt.axis("off")
plt.show() #needed in pycharm


for image, label in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = img_augmentation(tf.expand_dims(image[0], axis=0))
        print("IMage taken: ", aug_img[0])
        plt.imshow(aug_img[0].numpy().astype("uint8"))  #we get dark images here if we do normalizatoin.
        #plt.title("{}".format(labels(label)))
        plt.axis("off")
plt.show() #needed in pycharm

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = img_augmentation(inputs)
outputs = efn.EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(x) #Include top set to true as we are not using pre-trained weights but training from scratch

# Initialise the model. Compute and show summary.
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





