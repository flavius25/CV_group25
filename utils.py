import cv2
import numpy as np
import os
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
import pandas as pd

from dataStackGenerator import dataStackGenerator

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

            for c_lab in action_categories:
                os.mkdir(f"{dirs_needed[s]}/{c_lab}") # in each directory make directories for all categories

            counter = 0
            #Loop through all images and place them in the correct folder
            for file in range(len(files_n_labels[s][0])):
                c_lab = files_n_labels[s][1][file]
                image = cv2.imread(f"Stanford40/JPEGImages/{files_n_labels[s][0][file]}")
                image_name = f"{files_n_labels[s][1][file]}_{counter}.jpg"
                print(image_name, c_lab)
                path = f'./{dirs_needed[s]}/{c_lab}'
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

            for c_lab in classes:
                os.mkdir(f"{dirs_needed[s]}/{c_lab}") # in each directory make directories for all categories

            counter = 0
            #Loop through all videos, take middle frame and place them in the correct folder
            for video in range(len(files_n_labels[s][0])):
                c_lab = files_n_labels[s][1][video]
                vidcap = cv2.VideoCapture(f'TVHI_data/tv_human_interactions_videos/{files_n_labels[s][0][video]}')
                middle_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/2)
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame) #Get the middle frame of the video
                success, frame = vidcap.read()
                image_name = f"{files_n_labels[s][1][video]}_{counter}.jpg"
                print(image_name, c_lab)
                path = f'./{dirs_needed[s]}/{c_lab}'
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

        for c_lab in classes:
            os.mkdir(f"{dirs_needed[s]}/{c_lab}") # in each directory make directories for all categories

        counter1 = 0
        #Loop through all videos, take middle frame and place them in the correct folder
        for video in range(len(files_n_labels[s][0])):
            video_name = f"{files_n_labels[s][0][video][:-4]}"
            label = files_n_labels[s][1][video]
            print("video_name:  ", video_name)
            os.mkdir(f"{dirs_needed[s]}/{label}/{video_name}")
            vidcap = cv2.VideoCapture(f'TVHI_data/tv_human_interactions_videos/{files_n_labels[s][0][video]}')
            starting_frame = int((vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/2)-8)
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame) #Get the middle frame of the video
            success, old_frame = vidcap.read()
            #preprocess image
            hsv = np.zeros_like(old_frame) 
            hsv[...,1] = 255                                                # Set HSV's Value-channel to constant
            old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)         # Convert to grayscale to fit algorithm (Farneback)

            counter = 0
            for i in range(16): #Loop over 16 frames, middle frame will be middle of stack

                success, new_frame = vidcap.read()
                if not success:
                    break
                
                #Do preprocessing of new frame 
                new_frame  = cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(old_frame,new_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)   # calculate the optical flow for each pixel in the frame with Farneback
                
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])       # find magnitude and direction and encode it in an image
                hsv[..., 0] = ang*180/np.pi/2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                of_frame  = cv2.resize(bgr, IMG_SIZE)                   # Resize image to fit the other data
                frame_name = f"{video_name}_{counter}.jpg"
                path = f'./{dirs_needed[s]}/{label}/{video_name}'
                #print("path: ", path)
                cv2.imwrite(os.path.join(path,frame_name), of_frame) #Write image to directory
                counter += 1 
            
            print(f"{video_name} , {label}")
            counter1 += 1 

    print("Done sorting images!")

def toCSVconverter():
    num_classes = 4
    labels_name = {'handShake' : 0, 'highFive': 1, 'hug' : 2, 'kiss' : 3}

    dirs = ["OF_train", "OF_test", "OF_validation"]

    os.mkdir("data_files")
    for i in dirs:
        os.mkdir(f"data_files/{i}")
        train_data_path = f"{i}"
    
        data_dir_list = os.listdir(train_data_path)
        for data_dir in data_dir_list:
            label = labels_name[str(data_dir)]
            video_list = os.listdir(os.path.join(train_data_path, data_dir))
            for vid in video_list: # Loop over each video
                train_df = pd.DataFrame(columns=["FileName", "Label", "ClassName"]) #Create dataframe for each video
                img_list = os.listdir(os.path.join(train_data_path, data_dir, vid))
                for img in img_list:
                    img_path = os.path.join(train_data_path, data_dir, vid, img) #get the image path for each frame and append it to the created dataframe
                    train_df = train_df.append({"FileName": img_path, "Label" : label, "ClassName" : data_dir}, ignore_index=True)
                file_name = f"{data_dir}_{vid}.csv"
                train_df.to_csv(f"data_files/train/{file_name}")

    #https://medium.com/@anuj_shah/creating-custom-data-generator-for-training-deep-learning-models-part-3-c239297cd5d6

toCSVconverter()

""" Load Optical Flow data as data generator which inputs 16 images as 1 input """

def loadOFdatagens():
    params = {
    'batch_size':64,
    'dim':(48,48),
    'n_classes':2,
    'is_autoencoder':True,
    'shuffle':True }

    train_gen = dataStackGenerator(path_to_traindata,**params)
    val_gen = dataStackGenerator(path_to_validationdata,**params)
    test_gen = dataStackGenerator(path_to_testdata, **params)

    return train_gen, val_gen, test_gen

def filegenerator(CSV_folder,temporal_length,temporal_stride):
    ## Creates a python generator that 'yields' a sequence of frames every time based on the temporal lengtha and stride.
    for file in CSV_folder:
        data = pd.read_csv('path to each .csv file)
        labels = list(data.Label)
        img_list = list(data.FileName)
        samples = deque()
        sample_count = 0

    for img in img_list:
        samples.append(img)
        if len(samples)== temporal_length: 
        samples_c = copy.deepcopy(samples)
        samp_count += 1
        for i in range(temporal_stride):
            samples.popleft() 
        yield samples_c,labels[0]
        samples.popleft()#Eliminates the frame at the left most end to                  #######################accomodate the next frame in the sequence to #######################previous frame.


##Function to create the files structured based on the temporal requirements.:
def seq_of_frames(folder,d_type,length,stride):

    for csv_file in os.listdir(folder+'/'+d_type) 
    file_gen = filegenerator(csv_file,temporal_length,temporal_stride)
    iterator = True
    data_list = []
    while iterator:
        try:
        X,y = next(file_gen)
        X = list(X) 
        data_list.append([X,y])
        except Exception as e:
            print("An exception has occured:",e)
            iterator = False 
    
    return data_list



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

   

# https://medium.com/swlh/building-a-custom-keras-data-generator-to-generate-a-sequence-of-videoframes-for-temporal-analysis-e364e9b70eb 

