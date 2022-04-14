import cv2
import numpy as np
import os
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers

""" Function for sorting the Stanford40 data in a way that can be accessed by Keras data loading function """

def dataExtractionSF(needDirectories):

    with open('Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
    
    
    with open('Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

        
    action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))  

    #Split training data here 
    train_files, validation_files, train_labels, validation_labels = train_test_split(train_files, train_labels, test_size=0.1, random_state=0, stratify=train_labels)

    if needDirectories:

        print("Beginning sorting images...")
            # Specify names of directories for train and test data
        dirs_needed = ["SF_train", "SF_test", "SF_validation"]
        files_n_labels = [[train_files, train_labels], [test_files, test_labels],[validation_files, validation_labels]]

        for s in range(len(dirs_needed)):

            os.mkdir(dirs_needed[s]) # make directory each for training and test set

            for label in action_categories:
                os.mkdir(f"{dirs_needed[s]}/{label}") # in each directory make directories for all categories

            counter = 0
            #Loop through all images and place them in the correct folder
            for file in range(len(files_n_labels[s][0])):
                label = files_n_labels[s][1][file]
                image = cv2.imread(f"Stanford40/JPEGImages/{files_n_labels[s][0][file]}")
                image_name = f"{files_n_labels[s][1][file]}_{counter}.jpg"
                print(image_name, label)
                path = f'./{dirs_needed[s]}/{label}'
                counter += 1
                cv2.imwrite(os.path.join(path,image_name), image) #Write image to directory 

        print("Done sorting images!")

    return train_labels, test_labels, validation_labels, action_categories


""" Load the Standford40 dataset """

def loadSF40(img_size=(224,224), needDirectories=False):

    train_labels, test_labels, validation_labels, class_names = dataExtractionSF(needDirectories)
 
    train_ds = keras.utils.image_dataset_from_directory(
    directory='SF_train/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=img_size,
    shuffle=True
    )

    val_ds = keras.utils.image_dataset_from_directory(
    directory='SF_validation/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=img_size,
    shuffle=True
    )

    test_ds = keras.utils.image_dataset_from_directory(
    directory='SF_test/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=img_size
    )
    
    return train_ds, test_ds, val_ds, train_labels, test_labels, validation_labels, class_names

""" Do dataExtraction on TVHI dataset, get the middle frame and sort into directories for easier data loading """

def dataExtractionTVHI(needDirectories):
    
    set_1_indices = [[2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
                    [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
                    [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
                    [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42]]
    set_2_indices = [[1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
                    [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
                    [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
                    [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50]]
    classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

    # test set
    test_files = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
    test_labels = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
   
    # training set
    train_files = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
    train_labels = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
     
    #Split training data here 
    train_files, validation_files, train_labels, validation_labels = train_test_split(train_files, train_labels, test_size=0.15, random_state=0, stratify=train_labels)


    if needDirectories:
        
        print("Beginning sorting images...")
            # Specify names of directories for train, validation and test data
        dirs_needed = ["TVHI_train", "TVHI_test", "TVHI_validation"]
        files_n_labels = [[train_files, train_labels], [test_files, test_labels],[validation_files, validation_labels]]

        for s in range(len(dirs_needed)):

            os.mkdir(dirs_needed[s]) # make directory each for training, validation and test sets

            for label in classes:
                os.mkdir(f"{dirs_needed[s]}/{label}") # in each directory make directories for all categories

            counter = 0
            #Loop through all videos, take middle frame and place them in the correct folder
            for video in range(len(files_n_labels[s][0])):
                label = files_n_labels[s][1][video]
                vidcap = cv2.VideoCapture(f'TVHI_data/tv_human_interactions_videos/{files_n_labels[s][0][video]}')
                middle_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/2)
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame) #Get the middle frame of the video
                success, frame = vidcap.read()
                image_name = f"{files_n_labels[s][1][video]}_{counter}.jpg"
                print(image_name, label)
                path = f'./{dirs_needed[s]}/{label}'
                counter += 1
                cv2.imwrite(os.path.join(path,image_name), frame) #Write image to directory 

        print("Done sorting images!")
    
    
    return train_labels, test_labels, validation_labels, classes

def loadTVHI(img_size=(224,224), needDirectories=False):

    train_labels, test_labels, validation_labels, class_names = dataExtractionTVHI(needDirectories)
 
    train_ds = keras.utils.image_dataset_from_directory(
    directory='TVHI_train/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=img_size,
    shuffle=True
    )

    val_ds = keras.utils.image_dataset_from_directory(
    directory='TVHI_validation/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=img_size,
    shuffle=True
    )

    test_ds = keras.utils.image_dataset_from_directory(
    directory='TVHI_test/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=img_size
    )
    
    return train_ds, test_ds, val_ds, train_labels, test_labels, validation_labels, class_names

""" the Optical Flow input to the CNN """

def opticalFlowDataExtraction(IMG_SIZE=(224,224)):
    
    #Take relevant data and create test and training set (set 1 = test set,set 2 = training set)
    set_1_indices = [[2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
                 [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
                 [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
                 [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42]]
    set_2_indices = [[1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
                    [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
                    [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
                    [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50]]
    classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class
 
    # test set
    test_files = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
    test_labels = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
   
    # training set
    train_files = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
    train_labels = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
     
    #Split training data here 
    train_files, validation_files, train_labels, validation_labels = train_test_split(train_files, train_labels, test_size=0.15, random_state=0, stratify=train_labels)

    
    print("Beginning sorting images...")
        # Specify names of directories for train, validation and test data
    dirs_needed = ["OF_train", "OF_test", "OF_validation"]
    files_n_labels = [[train_files, train_labels], [test_files, test_labels],[validation_files, validation_labels]]

    for s in range(len(dirs_needed)):

        os.mkdir(dirs_needed[s]) # make directory each for training, validation and test sets

        for label in classes:
            os.mkdir(f"{dirs_needed[s]}/{label}") # in each directory make directories for all categories

        counter = 0
        #Loop through all videos, take middle frame and place them in the correct folder
        for video in range(len(files_n_labels[s][0])):
            video_name = f"{files_n_labels[s][0][video]}_{counter}"
            os.mkdir(f"{dirs_needed[s]}/{label}/{video_name}")
            label = files_n_labels[s][1][video]
            vidcap = cv2.VideoCapture(f'TVHI_data/tv_human_interactions_videos/{files_n_labels[s][0][video]}')
            starting_frame = int((vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/2)-8)
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame) #Get the middle frame of the video
            success, old_frame = vidcap.read()
            
            #preprocess image
            hsv = np.zeros_like(old_frame) 
            hsv[...,1] = 255                                                # Set HSV's Value-channel to constant
            old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)         # Convert to grayscale to fit algorithm (Farneback)
          

            counter2 = 0
            for i in range(16): #Loop over 16 frames, middle frame will be middle of stack

                success, new_frame = vidcap.read()
                if not success:
                    break
                
                #Do preprocessing of new frame 
                new_frame  = cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)
                new_frame  = cv2.resize(new_frame, IMG_SIZE)

                flow = cv2.calcOpticalFlowFarneback(old_frame,new_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)   # calculate the optical flow for each pixel in the frame with Farneback
                
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])       # find magnitude and direction and encode it in an image
                hsv[..., 0] = ang*180/np.pi/2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                of_frame  = cv2.resize(bgr, IMG_SIZE)                   # Resize image to fit the other data
                
                frame_name = f"{video_name}_{counter2}.jpg"
                path = f'./{dirs_needed[s]}/{label}/{video_name}'
                cv2.imwrite(os.path.join(path,frame_name), of_frame) #Write image to directory
                counter2 += 1 
            
            print(f"{video_name} , {label}")
            counter += 1 

    print("Done sorting images!")

"""   Data augmentation and Normalisation """
def dataAugmentation():

    img_augmentation = Sequential(
    [
        #layers.Rescaling(scale=1./255),
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomContrast(factor=0.1),
        layers.RandomZoom(0.1)
    ],
    name="img_augmentation",
    )

    return img_augmentation


""" Function for plotting accuracy"""

def plotAccuracy(title, train_acc, val_acc):
    plt.title(title)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.ylim([0, 1])
    plt.show()

""" Function for plotting loss"""

def plotLoss(title, train_loss, val_loss):
    plt.title(title)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(['train', 'val'], loc = 'lower left')
    plt.ylim([0, 5])
    plt.show()


opticalFlowDataExtraction()   

# https://medium.com/swlh/building-a-custom-keras-data-generator-to-generate-a-sequence-of-videoframes-for-temporal-analysis-e364e9b70eb 