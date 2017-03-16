# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:39:30 2017

@author: vjethlya
"""

#Import all the required libraries 
import csv
import cv2
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers import Lambda
import sklearn
from sklearn.utils import shuffle
import random


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags 
# 4 flags define : No. of Epochs  , batch size, which model to choose and Visualize the data

flags.DEFINE_integer('epochs', 2, "The number of epochs.")
flags.DEFINE_integer('batch_size', 32, "The batch size.")
flags.DEFINE_string('model','nvidia', "Name of model(lenet/nvidia) to use.")
flags.DEFINE_integer('visualize_data', 0 , "Visualize Data and save graph" )

csv_file = 'data/driving_log.csv'
correction = 0.25
center_images = []
left_images = []
right_images = []
steering_angle = []
throttles =[]
brakes = []
speeds = []

def read_csv(csv_file):
    
    '''
    Description : 
        This function read the csv files and store csv information
        in list
    
    Input parameter :
        csv file location 
    
    Return : 
        List of lines which contain information about car driving data i.e.
        center/left/right car driving image location, steering angle, throttle,
        break and speed. 
    '''
    lines = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in islice(reader, 1, None):
         #for line in reader:
            lines.append(line)
        return lines

def create_data_list(lines):
    
    '''
    Description : 
        This function read data from list lines and create individual list of 
        car data parameter
    Input parameter :
       List of lines consist of car driving parameters
    
    Return : 
        None
        Fill the global list variable of center, left, right images And other 
        car parameters.
    '''
    for line in lines:
        centre_img = 'data/IMG/'+line[0].split('/')[-1]
        left_img = 'data/IMG/'+line[1].split('/')[-1]
        right_img = 'data/IMG/'+line[2].split('/')[-1]
        angle = float(line[3])
        throttle = float(line[4])
        brake = float(line[5])
        speed = float(line[6])
        img = cv2.imread(centre_img)
        center_images.append(img)
        img = cv2.imread(left_img)
        left_images.append(img)
        img = cv2.imread(right_img)
        right_images.append(img)
        steering_angle.append(angle)
        throttles.append(throttle)
        brakes.append(brake)
        speeds.append(speed)
    print('done')

def plot_histogram(steering_angle, throttles, brakes, speeds):
    
    '''
    Description : 
        This function plot and save histogram of steering_angle, throttles,
        brakes and speed.
        
    Input parameter :
       List of steering_angle, throttles, brakes, speeds 
    
    Return : 
        None
    '''
    print('Plotting histogram')
    fig, axs = plt.subplots(1,4, figsize=(20, 5))
    axs[0].hist(steering_angle, bins=40)
    axs[0].set_title('Steering Angle')
    axs[1].hist(throttles, bins=40)
    axs[1].set_title('Throttles')
    axs[2].hist(brakes, bins=40)
    axs[2].set_title('Break')
    axs[3].hist(speeds, bins=40)
    axs[3].set_title('Speed')
    plt.savefig('Data_histogram.png')
    print('Done')

def visualize_agument_data(center_images, left_images, right_images, steering_angle):
    
    '''
    Description : 
        This function visualize and save randomly choose ceter, left, right
        image as 
        a) Original image 
        b) Brightness change image 
        c) Flip the image 

    Input parameter :
       List of center_images, left_images, right_images, steering_angle
    
    Return : 
        None
    '''    
    print('Plotting Agument data')
    data = ['Center', 'Left' , 'Right']
    index = random.randint(0, len(center_images))
    center_img = center_images[index]
    left_img = left_images[index]
    right_img = right_images[index]
    
    center_angle = steering_angle[index]
    left_angle = center_angle + correction
    right_angle = center_angle - correction
    
    orig_img = [center_img, left_img, right_img]
    orig_angle = [center_angle , left_angle , right_angle]
    
    bright_center_img = image_brightness(center_img)
    flip_center_img , flip_center_angle = flip_image(center_img, center_angle)
    
    bright_left_img = image_brightness(left_img)
    flip_left_img , flip_left_angle = flip_image(left_img, left_angle)
    
    bright_right_img = image_brightness(right_img)
    flip_right_img , flip_right_angle = flip_image(right_img, right_angle)
    
    bright_img =[bright_center_img , bright_left_img, bright_right_img]
    
    flip_img =[flip_center_img, flip_left_img , flip_right_img ]
    flip_angle = [flip_center_angle, flip_left_angle, flip_right_angle]
    
    fig, axs = plt.subplots(1,3, figsize=(10, 5))
    for i in range(3):
        axs[i].imshow(orig_img[i])
        axs[i].set_title('Original ' + data[i] + ' Image' +'\n Streeting Angle: ' + str(orig_angle[i]), fontsize=10)
    plt.savefig('Orig_Image.png')
        
    fig, axs = plt.subplots(1,3, figsize=(10, 5))
    for i in range(3):
        axs[i].imshow(bright_img[i])
        axs[i].set_title('Brightness change ' + data[i] + ' Image' +'\n Streeting Angle: ' + str(orig_angle[i]), fontsize=10)
    plt.savefig('Brightness_Image.png')
    
    fig, axs = plt.subplots(1,3, figsize=(10, 5))
    for i in range(3):
        axs[i].imshow(flip_img[i])
        axs[i].set_title('Flip ' + data[i] + ' Image' +'\n Streeting Angle: ' + str(flip_angle[i]), fontsize=10)
    plt.savefig('Flip_Image.png')
    print('Done')

def split_data(lines):
    
    '''
     Description : 
        This function split the data into training and validation data set.
        Training data : 80%
        validation data : 20%
    Input parameter :
        List of lines consist of car driving parameters.
    Return : 
        List of training data set and validation set after splitting.
        
    '''
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    print('Training Dataset size :', len(train_samples))
    print('Validation Dataset Size :',len(validation_samples))
    return train_samples, validation_samples

def image_brightness(image):
    
    '''
    Description : 
        This function does random brightness change on input image
    
    Input parameter :
        Image
    
    Return : 
        Brightness change image
        
    '''
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.4 + np.random.uniform()
    img[:,:,2] = img[:,:,2]*random_bright
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

def flip_image(image, angle):
    
    '''
    Description :
        This function flip the image (horizontal mirror) and multiply steering
        angle by -1.0
    Input Parameter :
        Imange and steering angle
    Return :
        Fliped image and opposite steering angle

    '''
    flip_img = cv2.flip(image,1)
    angle = angle * -1.0
    return flip_img, angle


def augument_data(image, steering_angle):
    
    '''
    Description : 
        This function randomly augment input data as below:
        a) Change the brightness of image and no change in steering angle.
        b) Flip the input image and steering angle (multiply by -1.0).
        c) No change in image and steering angle.

    Input parameter :
        image : Image of car driving.
        steering angle : Steering angle of car
    Return : 
        Modified image and steering angle.
        
    '''
    index = random.randint(0, 2)

    if(index == 0 ):
        bright_image = image_brightness(image)
        steering_angle = steering_angle
        return bright_image, steering_angle
    elif(index == 1):
        flip_img, steering_angle = flip_image(image, steering_angle)
        return flip_img, steering_angle
    else:
        return image, steering_angle

def generator(samples, batch_size):
    
    '''
    Description : 
        This function process the data set. This is a generator function which
        provide a results to its caller without destroying local variables.
        Generator will resume the execution where it was left off. This way it 
        will process all the dataset images. Infinite loop [while 1] ensures 
        all input dataset is processed.
        It assumes the driving Images location at 
        current directory + /data/IMG.
        It reads center, left and right images and apply the augmentation.
        If input count of data is X times, then the return count the 3X times.
        For example : batch size is 32, then it will return 32*3 = 96 no. of 
        processed data set.
        
    Input parameter :
        Samples : Input dataset.
        batch size : The no. of samples to process in each batch. By default 
        value is 32. 
    
    Return : 
        X_train : Processed data set
        y_train : Label of data set. In this case, it is steering angle.
        
    '''
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                centre_img = 'data/IMG/'+batch_sample[0].split('/')[-1]
                left_img = 'data/IMG/'+batch_sample[1].split('/')[-1]
                right_img = 'data/IMG/'+batch_sample[2].split('/')[-1]
                center_angle = float(batch_sample[3])
                
                # Add Centre images
                centre_image, center_angle = augument_data(cv2.imread(centre_img), center_angle)
                images.append(centre_image)
                angles.append(center_angle)

                # Add Left images
                left_angle = center_angle + correction
                left_image, left_angle = augument_data(cv2.imread(left_img), left_angle)
                images.append(left_image)
                angles.append(left_angle)

                # Add Right images
                right_angle = center_angle - correction
                right_image, right_angle = augument_data(cv2.imread(right_img), right_angle)
                images.append(right_image)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
				
def lenet_model():
    
    '''
    Description : 
        This function creates the lenet model. First it does the normalization 
        on input images set and later crop the image to remove unwanted scene.
        We used kereas library to create lenet model. letnet model consists of
        2 convolution layer and 3 fully connected layer.
        
    Input parameter :
        None 
    
    Return : 
        Lenet model function . 
    '''
    
    print('Inside Lenet model')

    model = Sequential()

    # Set up lamda layer
    model.add(Lambda(lambda x: x /255 - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))

    # cropping the image
    model.add(Cropping2D(cropping=((65,25), (0,0))))

    # Conv Layer 1
    model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape=(3,160,320)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Layer 2
    model.add(Convolution2D(16, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flatten the output for FC						
    model.add(Flatten())

    # FC Layer 1
    model.add(Dense(120))

    # FC Layer 2
    model.add(Dense(84))
    # Output Layer
    model.add(Dense(1))
    print('Exiting Lenet model')
    return model

def nvidia_model():
    
    '''
    Description : 
        This function creates Nvidia E2E driving AI model. First it does the
        normalization on input images set and later crop the image to remove 
        unwanted scene. We used kereas library to create Nvidia model. Nvidia 
        model consists of five convolutional layers with three fully connected
        layers and ouput layer. Used dropout function in FC layer for 
        preventing overfitting.
        
    Input parameter :
        None 
    
    Return : 
        Nvidia model.
    '''
    print('Inside nvidia model')
    model = Sequential()

    # Cropping the image
    model.add(Cropping2D(cropping=((60,20),(0,0)), input_shape=(160,320,3))) # crop image to only see section with road

    #Normalize the image
    #model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    # Conv Layer 1 with kernel size 5x5 and stride = 2x2
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

    # Conv Layer 2 with kernel size 5x5 and stride = 2x2
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))

    # Conv Layer 3 with kernel size 5x5 and stride = 2x2
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

    # Conv Layer 4 with kernel size 3x3
    model.add(Convolution2D(64,3,3,activation="relu"))

     # Conv Layer 5 with kernel size 3x3
    model.add(Convolution2D(64,3,3,activation="relu"))

    # Flatten the output for FC
    model.add(Flatten())

    # FC Layer 1
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
    # FC Layer 2
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # FC Layer 3
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # Output layer
    model.add(Dense(1))
    model.add(Activation('linear'))
    print('Exiting nvidia model')
    return model

def plot_loss_graph(history_object):
    
    '''
    Description : 
        This function plot the Training and validation loss graph for no. of
        epochs and save as 'Loss_graph.png' file.
        
    Input parameter :
        Dictionary variable containing training and validation loss values.
    
    Return : 
        None
        
    '''
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('Loss_graph.png')

def train_model(train_generator, validation_generator, train_samples, validation_samples):

    '''
    Description : 
        This function first choose the model based on user input (by default
        it takes nvidia model) . Compile and Train the model for no. of epochs
        (by default 2 epochs). Once the training and validation is done, it 
        saves the model , provide summary and plot the graph.
        
    Input parameter : 
        train_generator , validation_generator, train_samples, 
        validation_samples
      
    Return : 
        None
        
    '''
    
    print('Training the model')
    
    model_name = FLAGS.model
    if(model_name == 'lenet'):
        print('Lenet Model is used')
        model = lenet_model()
    elif(model_name == 'nvidia'):
        print('Nvidia model is used')
        model = nvidia_model()
    else:
        print('This is not correct model. Using "Nvidia" as default model')
        model = nvidia_model()

    # Configure its learning process with .compile().

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples*3), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples*3), nb_epoch=FLAGS.epochs)
    model.save('model.h5')
    model.summary()
    print('Plot the Loss histogram')
    plot_loss_graph(history_object)

def main(_):
    
    '''
    Description : 
        This is main function. It calls all require functions in flow to 
        train and save the model. Flow is below:
        Read the CSV File --> Split the data into training and validation -->
        process training and validation data using generator function for all 
        data set values --> Train the model and save model file.
    Input parameter :
        None
    Return : 
        None
    '''
    # Read the csv file
    lines = read_csv(csv_file)
        
    if(FLAGS.visualize_data):
        # create data list of center,left,right, steering angle, throttle, brake and speed.
        create_data_list(lines)
        # Plot the histogram of steering angle, thottle, brake and speed
        plot_histogram(steering_angle, throttles, brakes, speeds)
        # Visulize agument data 
        visualize_agument_data(center_images, left_images, right_images, steering_angle)
    
    #Split the training and validation set
    train_samples, validation_samples = split_data(lines)
    
    #create train and validation generator 
    train_generator = generator(train_samples, FLAGS.batch_size)
    validation_generator = generator(validation_samples, FLAGS.batch_size)
    
    # Train the model
    train_model(train_generator, validation_generator, train_samples, validation_samples )

if __name__ == '__main__':
    tf.app.run()
