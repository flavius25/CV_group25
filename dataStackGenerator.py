class dataStackGenerator(data_utils.Sequence):
    
    def __init__(self,data,batch_size,dim,n_classes,is_autoencoder,shuffle):
        
        #Initializing the values
        self.dim = dim
        self.data  = data
        self.batch_size = batch_size
        self.list_IDs = np.arange(len(data))
        self.n_classes = n_classes
        self.is_autoencoder = is_autoencoder
        self.shuffle = shuffle 
        
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indexes = self.list_IDs
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __len__(self): 
        return int(np.floor(len(self.data)/self.batch_size))
        
    def __getitem__(self, index): 
        index = self.indexes[index*self.batch_size:(index+1)*self.batch_size] 
        list_IDs_temp = [self.list_IDs[k] for k in index]
        X,y = self.__data_generation(list_IDs_temp)
        return X,y
    
    def __data_generation(self,list_IDs_temp): 
        X_data = []
        y_data = []
        for i,_ in enumerate(list_IDs_temp):
            batch_samples = self.data.iloc[i,0]
            y = self.data.iloc[i,1]
            temp_data_list = []
            for img in batch_samples:
                try:
                    image = cv2.imread(img,0)
                    ext_img = cv2.resize(image,self.dim)
                except Exception as e:
                    print('Value error ',e)  
            
            temp_data_list.append(ext_img) 
        
        X_data.append(temp_data_list)
        y_data.append(y)
    
    X = np.array(X_data)
    y = np.array(y_data)
    if self.is_autoencoder == True:
        return X, X
    else:
        return X, keras.utils.to_categorical(y,num_classes=self.n_classes)