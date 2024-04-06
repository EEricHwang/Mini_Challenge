import numpy as np
import matplotlib.pyplot as plt
import keras
import csv
import cv2
import os
import tensorflow as tf
import re
import natsort
import pandas as pd
import pdb
from PIL import Image
from skimage import io
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

# Step = 1 : Cropped images
# Step = 2 : Generate image values 
# Step = 3 : Generate image labels
# Step = 4 : Cropping Testing Images
# Step = 5 : CNN learning
# Step = 6 : Convert Data for Submission

Image_rows = 200
Image_cols = 200

num_classes = 100

category               =  pd.read_csv('/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/category.csv')
category_tmp           =  category.values
category_label         =  category_tmp[:,0]
category_name          =  category_tmp[:,1]

train_small            =  pd.read_csv('/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/train_small.csv')
train_small_read       =  train_small.values
train_small_read_name  =  train_small_read[:,2]

label = []

step  = 5

count = 0

# Step 1: Cropping images and Labels... 
if step == 1:
    path_dir = "/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v1_small/train_small"
    os.chdir(path_dir)
    
    file_list      = natsort.natsorted(os.listdir(path_dir))
    file_name_list = []

    for i in range(len(file_list)):
        file_name_list.append(file_list[i].replace(".jpg",""))

    for file in file_name_list:  # 3000,6000 is ok... 
        if count == 6800:
            break
        # First, try with cv2.imread
        img  = cv2.imread("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v1_small/train_small/"+file+".jpg")
        name = train_small_read_name[int(file)] 

        if img is not None:
            save         = []
            face_cascade = cv2.CascadeClassifier("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/haarcascade_frontalface_default.xml") 
            gray         = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces        = face_cascade.detectMultiScale(gray,1.3,5)
            if len(faces) != 0:
                # Cropping images with faces...
                for (x,y,w,h) in faces:
                    cropped = img[y:y+h,x:x+w]
                    resize  = cv2.resize(cropped,(Image_rows,Image_cols))
                    # Save Cropped Image 
                    cv2.imwrite(f"/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_small_new/{file}.jpg",resize)
                
                # Generating new labels
                for i in range(len(category_name)):
                    if name == category_name[i]:
                        save.append(category_label[i])    
            else:
                # Cropping images without faces...
                resize  = cv2.resize(img,(Image_rows,Image_cols))
                # Save Cropped Image
                cv2.imwrite(f"/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_small_new/{file}.jpg",resize)
                
                # Generating new labels
                for i in range(len(category_name)):
                    if name == category_name[i]:
                        save.append(category_label[i])
        else:
            # Second, try with io.imread
            img_new = cv2.imread("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v1_small/train_small/"+file+".jpg")
        
            if img_new is not None:
                save         = []
                face_cascade = cv2.CascadeClassifier("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/haarcascade_frontalface_default.xml") 
                gray         = cv2.cvtColor(img_new,cv2.COLOR_BGR2GRAY)
                faces        = face_cascade.detectMultiScale(gray,1.3,5)          
                if len(faces) != 0:
                    # Cropping images
                    for (x,y,w,h) in faces:
                        cropped = img_new[y:y+h,x:x+w]
                        resize  = cv2.resize(cropped,(Image_rows,Image_cols))
                        # Save Cropped Image 
                        cv2.imwrite(f"/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_small_new/{file}.jpg",resize)
                    
                    # Generating new labels
                    for i in range(len(category_name)):
                        if name == category_name[i]:
                            save.append(category_label[i])        
                else:
                    # Cropping images without faces...
                    resize  = cv2.resize(img_new,(Image_rows,Image_cols))
                    # Save Cropped Image
                    cv2.imwrite(f"/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_small_new/{file}.jpg",resize)
                    
                    # Generating new labels
                    for i in range(len(category_name)):
                        if name == category_name[i]:
                            save.append(category_label[i])            
            else:
                 print(file," is not readable. \n")
        
        if len(save) != 0:
            label.append(save)

        save = []
        count = count + 1
        print(file," is saving. \n")

    # Save Labels for Cropped Image 
    np.save("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_small_labels",label)


# Step 2: Generate Image Values and Image Labels
elif step == 2: 
    # Generate Image Values
    path_dir = "/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_small_new"
    os.chdir(path_dir)

    file_list         = natsort.natsorted(os.listdir(path_dir))
    image_data_save   = np.empty((1,Image_rows,Image_cols),int)
    labels_tmp        = np.load("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_small_labels.npy")
    image_label_save  = []
    count             = 0

    for file in file_list:

        if file == '.DS_Store':
            break
        else:
            value            = cv2.imread(file,cv2.IMREAD_COLOR)           # Read cropped images
            value_gray       = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)     # Convert to gray scales
            image_data_save  = np.append(image_data_save,value_gray.reshape(1,Image_rows,Image_cols),axis=0)
            
            # 1 step shifting.... 

            tmp_value = labels_tmp[count][0]
            image_label_save.append(tmp_value)  
            Image_Label = np.array(image_label_save,dtype='i')
            print(file)

        count = count + 1    

    data_save    = image_data_save[1:len(file_list)+1]
    np.save("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_images_values",data_save)     # Save image values
    np.save("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_images_labels",Image_Label)   # Save labels

    #Image_Value  = pd.DataFrame(data_save)
    #Image_Value.to_csv('cropped_image_values.csv',index=False)             # Save image values

# =============================================================================================

# Step 3: Generate Image Labels
elif step == 3: 

    # Generate Labels
    image_label_save  = []
    labels_tmp        = np.load("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_small_labels.npy")

    for i in range(len(labels_tmp)):
        tmp_value = labels_tmp[i][0]
        image_label_save.append(tmp_value)

    Image_Label = np.array(image_label_save,dtype='i')
    np.save("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_images_labels",Image_Label)   # Save labels


# Step 4: Read Data and CNN Learning
elif step == 4:

    Iterations = 50
    save       = []

    cropped_images = np.load("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_images_values.npy")
    cropped_labels = np.load("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/cropped_images_labels.npy")

    # cropped_images_values.npy contains the image values for train_small
    # cropped_images_labels.npy contains the labels for train_small

    x_train     = cropped_images
    y_train     = cropped_labels

    input_shape = (Image_rows, Image_cols, 1)

    x_train = x_train.astype('float32') / 255.

    x_train = np.expand_dims(x_train,-1)
    y_train = keras.utils.to_categorical(y_train,num_classes)
    
    ## ... 
    x_test  = x_train 
    y_test  = y_train

    # ======================================== CNN Design ========================================
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())
    # ============================================================================================

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist  = model.fit(x_train, y_train, batch_size=128, epochs=Iterations, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Load testing images... 
    path_dir_testing = "/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/cropped_small_testing_new"

    file_list        = natsort.natsorted(os.listdir(path_dir_testing))
    file_name_list   = []

    for i in range(len(file_list)):
        file_name_list.append(file_list[i].replace(".jpg",""))

    for file in file_name_list:
        img_test       =   cv2.imread("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/cropped_small_testing_new/"+file+".jpg")
        img_test_gray  =   cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)     
        x_test         =   img_test_gray
        x_test         =   x_test.astype('float32') / 255.
        x_test         =   np.expand_dims(x_test,-1)
        x_test         =   np.expand_dims(x_test,0)    
        yhat           =   model.predict(x_test)
        T              =   yhat[0]
        T_save         =   np.argmax(T)
        save.append(T_save)
        print(file)

    np.save("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/submission_tmp",save)

elif step == 5:

    path_dir = "/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/test"
    os.chdir(path_dir)
    file_list  = natsort.natsorted(os.listdir(path_dir))

    index_filter  = []
    ID_test       = []    
    ID_cropped    = []
    ID            = []
    Category      = []

    Labels = np.load("/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new/submission_tmp.npy")
    
    # [1] Filter IDs from test folder
    for file in file_list:
        numbers  = re.sub(r'[^0-9]','',str(file))
        if file == '.DS_Store':
            break
        else:
            ID_test.append(str(file))
            ID.append(numbers)

    df = pd.DataFrame(ID, columns = ['ID'])

    path_dir = "/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/cropped_small_testing_new"
    os.chdir(path_dir)
    file_list  = natsort.natsorted(os.listdir(path_dir))

    # [2] Index filtering from cropped_small_testing folder
    for file in file_list:
        if file == '.DS_Store':
            break
        else:
            index_filter.append(int(re.sub(r'[^0-9]','',str(file))))
            ID_cropped.append(str(file))

    count = 1

    # [3] Filtering... 
    for file in ID_test:
        numbers  = int(re.sub(r'[^0-9]','',str(file)))

        if file in ID_cropped:
            Category.append(category_name[Labels[index_filter.index(numbers)]])
        else:
            Category.append('Angelina Jolie')
            count = count + 1

    df['Category'] = Category

    path_dir    = "/Users/eric_hwang/Desktop/Python/STAT_59800/Mini_Challenge/v3_small_new"
    os.chdir(path_dir)

    df.to_csv("submission_SH.csv", index = False)
