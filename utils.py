import cv2
import numpy as np
import os
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


""" Function for sorting the Stanford40 data in a way that can be accessed by Keras data loading function """

def createDataDirectories(boolean):

    if boolean == True:

        with open('Stanford40/ImageSplits/train.txt', 'r') as f:
            train_files = list(map(str.strip, f.readlines()))
            train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
            print(train_files[0:5])
            #print(f'Train files ({len(train_files)}):\n\t{train_files}')
            #print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')
        
        with open('Stanford40/ImageSplits/test.txt', 'r') as f:
            test_files = list(map(str.strip, f.readlines()))
            test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
            #print(f'Test files ({len(test_files)}):\n\t{test_files}')
            #print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')
            print("Lengt of train_files: ", len(train_files))
            
        action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))
        print(f'Action categories ({len(action_categories)}):\n{action_categories}')   

        # Specify names of directories for train and test data
        dirs_needed = ["SF_train", "SF_test"]
        files_n_labels = [[train_files, train_labels], [test_files, train_labels]]

        for set in len(dirs_needed):

            os.mkdir(dirs_needed[set]) # make directory each for training and test set

            for label in action_categories:
                os.makedir(f"{dirs_needed[set]}/{label}") # in each directory make directories for all categories

            #Loop through all images and place them in the correct folder
            for file in len(files_n_labels[set][0]):
                label = files_n_labels[set][1][file]
                image = cv2.imread(f"Stanford40/JPEGImages/{files_n_labels[set][0][file]}")
                image_name = files_n_labels[set][1][file]
                path = f"/{dirs_needed[set]}/{label}"
                (cv2.imwrite(os.path.join(path,image_name), image)) #Write image to directory 

        print("Done sorting images")


""" Load the Standford40 dataset, perform data preprocessing """

def loadSF40(img_size=(224,224)):
 
    train_ds = keras.utils.image_dataset_from_directory(
    directory='Stanford40/SF_train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=img_size,
    shuffle=True
    )

    test_ds = keras.utils.image_dataset_from_directory(
    directory='Stanford40/SF_test/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=img_size
    )
    
    return train_ds, test_ds


""" Load the TVHI dataset, do data preprocessing """

def loadTVHIData(img_size=(224,224)):
    
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
    set_1 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
    set_1_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
    print(f'Set 1 to be used for test ({len(set_1)}):\n\t{set_1}')
    print(f'Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n')

    # training set
    set_2 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
    set_2_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
    print(f'Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}')
    print(f'Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}')
    
    
    # Take middle frame from each video in TVHI dataset
    TVHI_training_set = []
    for video in set_2:
        vidcap = cv2.VideoCapture(f'../TVHI_data/tV_human_interactions_videos/{video}')
        middle_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/2)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame) #Get the middle frame of the video
        success, frame = vidcap.read()
        if success:
            frame = cv2.resize(frame, img_size)
            TVHI_training_set.append(frame)
            
    TVHI_test_set = []
    for video in set_1:
        vidcap = cv2.VideoCapture(f'../TVHI_data/tV_human_interactions_videos/{video}')
        middle_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/2)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        success, frame = vidcap.read()
        if success:
            TVHI_test_set.append(frame)
    
    train_labels = set_2_label
    test_labels = set_1_label
    
    return (TVHI_training_set, train_labels, TVHI_test_set, test_labels, classes) 


""" Function for calculating the optical flow with Farnebäck algorithm """

def opticalFlowCalculator(video_path, img_size=(224,224)):
    optical_flow_data = []
    
    for video in video_path:

        vidcap = cv2.VideoCapture(f'../TVHI_data/tV_human_interactions_videos/{video}') # get video
        middle_frame = int((vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/2)-8)      # get index of middle frame, set to -8 frames back so that when we take stack of frames, the middle one will be in the middle of the stack
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)               # set the video to the middle frame    
        success, old_frame = vidcap.read()                              # read image

        hsv = np.zeros_like(old_frame) 
        hsv[...,1] = 255                                                # Set HSV's Value-channel to constant

        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)         # Convert to grayscale to fit algorithm (Farneback)
        old_frame  = cv2.resize(old_frame, img_size)                   # Resize image to fit the other data

        stackOFframes = []
        
        OF_params = [0.5, 3, 15, 3, 5, 1.2, 0] #default Farnebacks parameters
        
        for i in range(16): #Loop over 16 frames, middle frame will be middle of stack
            success, new_frame = vidcap.read()
            if not success:
                break
            
            #Do preprocessing of new frame 
            new_frame  = cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)
            new_frame  = cv2.resize(new_frame, img_size)

            flow = cv2.calcOpticalFlowFarneback(old_frame,new_frame, None, OF_params)   # calculate the optical flow for each pixel in the frame with Farneback

            stackOFframes.append(flow)           # add the stack of 16 frames to list

            old_frame = new_frame               # update the previous frame to current frame
                
        optical_flow_data.append(np.asarray(stackOFframes))     #make the stack of frames into an np array and store in general optical flow data list
    
    return optical_flow_data


""" the Optical Flow input to the CNN """

def opticalFlowInput():
    
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
    set_1 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
    set_1_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
    print(f'Set 1 to be used for test ({len(set_1)}):\n\t{set_1}')
    print(f'Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n')

    # training set
    set_2 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
    set_2_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
    print(f'Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}')
    print(f'Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}')
    
    
    training_data = opticalFlowCalculator(set_2)
    testing_data = opticalFlowCalculator(set_1)
    train_labels = set_2_label
    test_labels = set_1_label
    
    return (training_data, train_labels, testing_data, test_labels)

"""   Data augmentation and Normalisation """
def getIterator(img_set, img_labels):

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

    #Create iterators to pass to the model during training
    return data_generator.flow(img_set, img_labels, batch_size=64)

