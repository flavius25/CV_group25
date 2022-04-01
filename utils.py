import cv2
import numpy as np



""" Load the Standford40 dataset, perform data preprocessing """

def loadSF40(img_size=(224,224)):
 
    with open('../Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
        print(f'Train files ({len(train_files)}):\n\t{train_files}')
        print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')
    
    with open('../Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
        print(f'Test files ({len(test_files)}):\n\t{test_files}')
        print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')
        
    action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))
    print(f'Action categories ({len(action_categories)}):\n{action_categories}')
    
    #Read images and put into list
    SF_training_set = []
    for i in range(len(train_files)):
        img = cv2.imread(f'../Stanford40/JPEGImages/{train_files[i]}')
        SF_training_set.append(img)
    
    SF_test_set = []
    for im in range(len(test_files)):
        img = cv2.imread(f'../Stanford40/JPEGImages/{test_files[i]}')
        SF_test_set.append(im)
    
    #Preprocess data with dataPreprocessing function
    SF_training_set, SF_test_set = dataPreprocessing(SF_training_set, SF_test_set, img_size=img_size)
    
    return (SF_training_set, train_labels, SF_test_set, test_labels)


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
        success, image = vidcap.read()
        if success:
            frame = cv2.resize(frame, img_size)
            TVHI_training_set.append(frame)
            
    TVHI_test_set = []
    for video in set_1:
        vidcap = cv2.VideoCapture(f'../TVHI_data/tV_human_interactions_videos/{video}')
        middle_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/2)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        success, image = vidcap.read()
        if success:
            TVHI_test_set.append(image)
    
    TVHI_training_set, TVHI_test_set = dataPreprocessing(TVHI_training_set, TVHI_test_set)
    train_labels = set_2_label
    test_labels = set_1_label
    
    return (TVHI_training_set, train_labels, TVHI_test_set, test_labels) 


""" Data Preprocessing Function """

def dataPreprocessing(training_set:list, test_set:list, img_size=(224,224), padding = False):
    
    #Preparing the images so that they fit  in the CNN
    training_set = np.asarray(training_set)
    test_set = np.asarray(test_set)
    
    #Use function reshape to reshape the images to 224,224 
    training_set = training_set.reshape(training_set.shape[0], img_size[0], img_size[1], 3)
    test_set = test_set.reshape(test_set.shape[0],img_size[0], img_size[1], 3)

    # normalise -scale to range 0-1   
    training_set = training_set / 255.0
    test_set = test_set / 255.0

    if padding:
        print(f"Image shape before: {train_images[0].shape}")

        # Pad images with 0s since we want information in the edges
        padding_size = ((0,0),(2,2),(2,2))
        train_images      = np.pad(train_images, padding_size, 'constant')
        validation_images = np.pad(validation_images, padding_size, 'constant')
        test_images       = np.pad(test_images, padding_size, 'constant')

        print(f"Updated Image Shape: {train_images[0].shape}.")

    return training_set, test_set


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

            #Encode the optical flow and convert to bgr image
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            stackOFframes.append(bgr)           # add the stack of 16 frames to list

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