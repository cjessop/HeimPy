# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module
#from configuration.Config import config
from sklearn.model_selection import train_test_split

#Import all necessary libraries for CNN training
try:
    import tensorflow as tf
    from keras import layers
    from keras.models import Sequential
    from keras.layers import Input
    from keras.layers import Dense, Dropout, Lambda, AveragePooling2D, Flatten, Rescaling, MaxPool2D, Conv2D
    from keras.layers import Rescaling, RandomContrast, RandomZoom, RandomTranslation, RandomBrightness, RandomRotation
    from keras.layers import RandomFlip, RandomCrop
    from keras.losses import SparseCategoricalCrossentropy, categorical_crossentropy
    from keras.utils import image_dataset_from_directory    
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.models import load_model
    from keras.optimizers import Adam, SGD
    from keras.applications import mobilenet_v2
    from keras.applications import MobileNetV2
    from keras.utils import image_dataset_from_directory
    from keras.utils import img_to_array
    from keras.preprocessing.image import ImageDataGenerator
    from keras import backend as K
    #from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from itertools import product as product
    import torchvision
    from PIL import Image, ImageDraw, ImageFont
    import cv2
except ImportError:
    print("Unable to Import Tensorflow/Keras/PyTorch inside of the CNN script")

from abc import ABC, abstractmethod
import inspect

def lrdecay(epoch):
                lr = 1e-3
                if epoch > 8:
                    lr *= 0.5e-3
                elif epoch > 6:
                    lr *= 1e-3
                elif epoch > 4:
                    lr *= 1e-2
                elif epoch > 2:
                    lr *= 1e-1
                return lr

class VGG16():
    def __init__(self, train_dir, test_dir, architecture, optimizer=SGD(lr=1e-4, momentum=0.9), class_number=None,
                 weight_path=None, model_path=None, checkpoint_path=None, image_test=None,
                 train_labels=None, test_labels=None) -> None:
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.architecture = architecture
        self.optimizer = optimizer
        self.class_number = class_number
        self.weight_path = weight_path
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.image_test = image_test
        self.train_labels = train_labels
        self.test_labels = test_labels

    def setup(self):
        trdata = ImageDataGenerator()
        traindata = trdata.flow_from_directory(directory=self.train_dir, target_size=(224,224))

        tstdata = ImageDataGenerator()
        testdata = tstdata.flow_from_directory(directory=self.test_dir, target_size=(224,224))

        return traindata, testdata


    def create_model(self):
        model = Sequential()

        if self.model == "VGG16":

            model.add(Conv2D(input_shape=(224,224,3), filters=64, kernel_size=(3,3),
                            padding="same", activation="relu"))
            
            model.add(Conv2D(filters=64,kernel_size=(3,3),
                            padding="same", activation="relu"))
            
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=128, kernel_size=(3,3),
                            padding="same", activation="relu"))
            model.add(Conv2D(filters=128, kernel_size=(3,3),
                            padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=256, kernel_size=(3,3),
                            padding="same", activation="relu"))
            model.add(Conv2D(filters=256, kernel_size=(3,3),
                            padding="same", activation="relu"))
            model.add(Conv2D(filters=256, kernel_size=(3,3),
                            padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=512, kernel_size=(3,3),
                            padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3),
                            padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3),
                            padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=512, kernel_size=(3,3),
                            padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3),
                            padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3),
                            padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name='vgg16'))
            model.add(Flatten(name='flatten'))
            model.add(Dense(256, activation="relu", name="fc1"))
            model.add(Dense(128, activation="relu", name="fc2"))
            model.add(Dense(self.class_number, activation="softmax", name="output"))

        elif self.model == "VGG19":
            from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
            model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
            model.add(Convolution2D(64, 3, 3, activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(64, 3, 3, activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
        
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(128, 3, 3, activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(128, 3, 3, activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
        
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(256, 3, 3, activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(256, 3, 3, activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(256, 3, 3, activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(256, 3, 3, activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
        
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, 3, 3, activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, 3, 3, activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, 3, 3, activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, 3, 3, activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
        
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, 3, 3, activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, 3, 3, activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, 3, 3, activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, 3, 3, activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
        
            model.add(Flatten())
            model.add(Dense(4096, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(4096, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1000, activation='softmax'))

        elif self.model == "UNET":
            from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Conv2DTranspose 
            from tensorflow.keras.layers import MaxPooling2D, Concatenate, Input, Cropping2D, Flatten

            #Contraction path
            c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', 
            kernel_initializer='he_normal', padding='same')(s)
            c1 = tf.keras.layers.Dropout(0.1)(c1)
            c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
            kernel_initializer='he_normal', padding='same')(c1)
            p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

            c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
            kernel_initializer='he_normal', padding='same')(p1)
            c2 = tf.keras.layers.Dropout(0.1)(c2)
            c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
            kernel_initializer='he_normal', padding='same')(c2)
            p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
            
            c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', 
            kernel_initializer='he_normal', padding='same')(p2)
            c3 = tf.keras.layers.Dropout(0.2)(c3)
            c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
            kernel_initializer='he_normal', padding='same')(c3)
            p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
            
            c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', 
            kernel_initializer='he_normal', padding='same')(p3)
            c4 = tf.keras.layers.Dropout(0.2)(c4)
            c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', 
            kernel_initializer='he_normal', padding='same')(c4)
            p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
            
            c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', 
            kernel_initializer='he_normal', padding='same')(p4)
            c5 = tf.keras.layers.Dropout(0.3)(c5)
            c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', 
            kernel_initializer='he_normal', padding='same')(c5)

            u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
            u6 = tf.keras.layers.concatenate([u6, c4])
            c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', 
            padding='same')(u6)
            c6 = tf.keras.layers.Dropout(0.2)(c6)
            c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', 
            padding='same')(c6)
            
            u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
            u7 = tf.keras.layers.concatenate([u7, c3])
            c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', 
            padding='same')(u7)
            c7 = tf.keras.layers.Dropout(0.2)(c7)
            c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', 
            padding='same')(c7)
            
            u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
            u8 = tf.keras.layers.concatenate([u8, c2])
            c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', 
            padding='same')(u8)
            c8 = tf.keras.layers.Dropout(0.1)(c8)
            c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', 
            padding='same')(c8)
            
            u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
            u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
            c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', 
            padding='same')(u9)
            c9 = tf.keras.layers.Dropout(0.1)(c9)
            c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', 
            padding='same')(c9)

            outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

            model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        elif self.model == "RESNET":
            global train_im, test_im
            train_im = self.train_dir # Fix this
            test_im = self.test_dir # and fix this
            train_im = train_im[..., np.newaxis] / 255.0
            test_im = test_im[..., np.newaxis] / 255.0

            train_im_resized = tf.image.resize(self.traindata, [32, 32])
            test_im_resized = tf.image.resize(self.testdata, [32, 32])

            train_lab_categorical = tf.keras.utils.to_categorical(self.train_labels, num_classes=10, dtype='uint8')
            test_lab_categorical = tf.keras.utils.to_categorical(self.test_labels, num_classes=10, dtype='uint8')

            batch_size = 64
            train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2, 
                                                                            width_shift_range=0.1, 
                                                                            height_shift_range=0.1, 
                                                                            horizontal_flip=True)

            # No augmentation for validation set
            valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

            train_im_resized_np = train_im_resized.numpy()
            test_im_resized_np = test_im_resized.numpy()

            from sklearn.model_selection import train_test_split
            train_im, valid_im, train_lab, valid_lab = train_test_split(train_im_resized_np, train_lab_categorical, 
                                                            test_size=0.20, stratify=train_lab_categorical, 
                                                            random_state=40, shuffle=True)



            # Prepare the train and validation sets
            global train_set_conv, valid_set_conv
            train_set_conv = train_DataGen.flow(train_im, train_lab, batch_size=batch_size)
            valid_set_conv = valid_datagen.flow(valid_im, valid_lab, batch_size=batch_size)

            # Use pretrained and prepacked ResNet50 model

            model = tf.keras.applications.ResNet50(weights=None, classes=self.class_number, input_shape=(32, 32, 1))
 
            lrdecay_callback = tf.keras.callbacks.LearningRateScheduler(lrdecay)

            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

            return model

        opt = Adam(lr=0.001)
        model.compile(optimizer=opt, loss=categorical_crossentropy,
                            metrics=['accuracy'])

        return model
    
    def model_summary(self):
        if self.model == "VGG16":
            global model_VGG16
            model_VGG16 = self.create_model()

            model_VGG16.summary()
        elif self.model == "VGG19":
            global model_VGG19
            model_VGG19 = self.create_model()

            model_VGG19.summary()

        elif self.model == "UNET":
            global model_UNET
            model_UNET = self.create_model()

            model_UNET.summary()

    def compile(self):
        from keras.callbacks import ModelCheckpoint, EarlyStopping

        checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1,
                               mode='auto')

        return checkpoint, early
    
    def fit(self):
        model = self.create_model()

        checkpoint, early = self.compile()

        traindata, testdata = self.setup()

        if self.model != "RESNET":

            hist = model.fit_generator(steps_per_epoch=100, generator=traindata, validation_data=testdata,
                                    validation_steps=10, epochs=100, callbacks=[checkpoint,early])
            
            plt.plot(hist.history["acc"])
            plt.plot(hist.history['val_acc'])
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.title("model accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
            plt.show()
        
        else:
            try:
                batch_size = 64
                resnet_train = model.fit(
                    train_set_conv, 
                    epochs=10, 
                    steps_per_epoch=train_im.shape[0] // batch_size, 
                    validation_steps=valid_im.shape[0] // batch_size, 
                    validation_data=valid_set_conv, 
                    callbacks=[lrdecay_callback, earlystop_callback])
                
            except Exception as e:
                return ({'error': str(e)})

    def predict(self):
        img = image.load_img(self.image_test, target_size(224, 224))
        img = np.array(img)

        plt.imshow(img)

        from keras.models import load_model
        saved_model = load_model('vgg16_1.h5')