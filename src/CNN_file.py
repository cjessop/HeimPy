# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module
#from configuration.Config import config
from sklearn.model_selection import train_test_split

#Import all necessary libraries for CNN training
# try:
#     import tensorflow as tf
#     from keras import layers
#     from keras.models import Sequential
#     from keras.layers import Input
#     from keras.layers import Dense, Dropout, Lambda, AveragePooling2D, Flatten, Rescaling
#     from keras.layers import Rescaling, RandomContrast, RandomZoom, RandomTranslation, RandomBrightness, RandomRotation
#     from keras.layers import RandomFlip, RandomCrop
#     from keras.losses import SparseCategoricalCrossentropy
#     from keras.utils import image_dataset_from_directory    
#     from keras.callbacks import EarlyStopping
#     from keras.models import load_model
#     from keras.optimizers import Adam
#     from keras.applications import mobilenet_v2
#     from keras.applications import MobileNetV2
#     from keras.utils import image_dataset_from_directory
#     from keras.utils import img_to_array
#     #from tensorflow.keras.preprocessing.image import ImageDataGenerator
# except ImportError:
#     print("Unable to Import Tensorflow/Keras inside of the Base Classes script")
#     exit(0)

from abc import ABC, abstractmethod
import inspect

# Optimizer retains its American spelling because that is the argument name for the methods required - I'm not happy about it either

def prepare_images(max_pixel_val, width, height, data_path, batch_size):
    """
    A function to prepare images for use in a Convolutional Neural Network (CNN).

    This function uses the ImageDataGenerator from keras to 
    """

    trainDs = image_dataset_from_directory(directory=data_path, 
                                            validation_split=0.2,
                                            subset="training",
                                            seed=123,
                                            image_size=(height, width),
                                            batch_size=batch_size)
    valDs = image_dataset_from_directory(directory=data_path, 
                                            validation_split=0.2,
                                            subset="validation",
                                            seed=123,
                                            image_size=(height, width),
                                            batch_size=batch_size)
    #imageGen = ImageDataGenerator(rescale=1/max_pixel_val, validation_split=0.2)
    #trainDatagen = imageGen.flow_from_directory(directory=data_path, target_size=(width,height), class_mode='binary',
    #                                            batch_size=16, subset='training')
    #valDatagen = imageGen.flow_from_directory(directory=data_path, target_size=(width,height), class_mode='binary',
    #                                            batch_size=16, subset='validation')
    
    return trainDs, valDs

class CNN_config():
    """
     A class for configuring and managing Convolutional Neural Networks (CNN).

    This class provides functionality to read CNN configurations from a file,
    create CNN models based on those configurations, compile and train the models,
    and visualise feature maps.

    Attributes:
        path (str): Path to the configuration file.
        optimizer: The optimizer to be used for model compilation.
        loss: The loss function to be used for model compilation.
        metrics: The metrics to be used for model evaluation.
        train_images: Training image data.
        test_images: Test image data.
        train_labels: Training labels.
        test_labels: Test labels.
        config_list (list): List to store configuration parameters.

    Methods:
        read(): Reads and processes the configuration file.
        createCNN(): Creates a CNN model based on the configuration.
        model_summary(): Displays a summary of the created model.
        model_create(): Compiles and trains the CNN model.
        feature_map(model, image): Generates and displays feature maps for a given image.
    """
    def __init__(self, path, optimizer, loss, metrics, train_images=None, test_images=None, train_labels=None, test_labels=None, config_list=[], conv_iter=0, dense_iter=0) -> None:
        """
        Initialises the CCN_config class with the given parameters.

        Args:
            path (str): Path to the configuration file.
            optimizer: The optimizer to be used for model compilation.
            loss: The loss function to be used for model compilation.
            metrics: The metrics to be used for model evaluation.
            train_images: Training image data.
            test_images: Test image data.
            train_labels: Training labels.
            test_labels: Test labels.
            config_list (list, optional): Initial configuration list. Defaults to an empty list.
        """
        self.path = path
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.config_list = config_list
        self.conv_iter = conv_iter
        self.dense_iter = dense_iter

    def read(self):
        """
        Reads the configuration file and processes its contents.

        Returns:
            list: A list of processed configuration parameters.
        """
        config_list = []
        with open(self.path, 'r') as data:
            for line in data:
                config_list.append(line)
        config_list = [(x.replace('\n', '')) for x in config_list]
        config_list = [(x.replace(' ', '')) for x in config_list]

        return config_list
    
    def createCNN(self):
        """
        Creates a CNN model based on the configuration read from the file.

        Returns:
            tensorflow.keras.Model: The created CNN model.
        """
        config_list = self.read()
        model = None
        
        for item in range(len(config_list)):
            if item == 0:
                if "Sequential" in config_list[item]:
                    model = Sequential()
                else:
                    print("Model must start with a sequential layer")
                    return None  # or raise an exception

            else:
                if model is None:
                    print("Model was not initialized properly")
                    return None  # or raise an exception

                if "Rescaling" in config_list[item]:
                    model.add(layers.Rescaling(1./255))

                elif "Conv2D" in config_list[item]:
                    item_splits = config_list[item].split(",")
                    if self.conv_iter < 1:
                        model.add(layers.Conv2D(int(item_splits[1]), (int(item_splits[2]), int(item_splits[3])), 
                                                activation=item_splits[4], input_shape=(int(item_splits[1]), int(item_splits[1]), int(item_splits[2]))))
                        self.conv_iter += 1
                    else:
                        model.add(layers.Conv2D(int(item_splits[1]), (int(item_splits[2]), int(item_splits[3])), 
                                                activation=item_splits[4]))
                    
                elif "MaxPooling2D" in config_list[item]:
                    item_splits = config_list[item].split(",")
                    model.add(layers.MaxPooling2D((int(item_splits[1]), int(item_splits[2]))))

                elif "Flatten" in config_list[item]:
                    model.add(layers.Flatten())

                elif "Dense" in config_list[item]:
                    item_splits = config_list[item].split(",")
                    if self.dense_iter < 0:
                        model.add(layers.Dense(int(item_splits[1]), activation=item_splits[2]))
                        print(item_splits)
                    else:
                        model.add(layers.Dense(int(item_splits[1])))
        
        return model


    def model_summary(self):
        """
        Displays a summary of the created CNN model.

        Returns:
            model summary
        """
        
        model = self.model_create()

        model_summary = model.summary()
        
        return model_summary

    def model_create(self):
        """
        Compiles and trains the CNN model.

        Returns:
            tuple: A tuple containing the compiled model and its training history.
        """
        
        model = self.createCNN()

        model.compile(optimizer=self.optimizer,
                    loss = tf.keras.losses.SparseCategoricalCrossEntropy(from_logits=True),
                    metrics=[self.metrics])

        history = model.fit(self.train_images, self.train_labels, epochs=10,
                            validation_data=(self.test_images, self.test_labels))

        return model, history
    
    def feature_map(self, model, image):
        """
        Generates and displays feature maps for a given image using the trained model.

        Args:
            model: The trained CNN model.
            image (str): Path to the input image file.

        Returns:
            No return
            Plot displayed
        """
        model = self.model_create()

        #img = load_img(image, target_size=(224, 224))   
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        #img = preprocess_input(img) # Need to use a different function for this

        feature_maps = model.predict(img)

        square = 8
        ix = 1

        for _ in range(square):
            for _ in range(square):
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')

                ix += 1
        plt.show()

    def show_performance(self, history, metric, metric_label):
        """
        Generates plots to compare the performance of the model on both the training and test sets.

        Args:
            history: The trained model 
            metric: A measure of the performance of the model (string)

        Returns:
            No return
            Plot displayed
        """
        if (isinstance(metric, str)):
            train_performance = history.history[metric]
            valid_performance = history.history['val_' + metric]
            intersection_index = np.argwhere(np.isclose(train_performance, valid_performance, atol=1e-2)).flatten()[0]
            intersecion_val = train_performance[intersection_index]
        else:
            print("metric must be of string type")
            exit(0)


        plt.plot(train_performance, label=metric_label)
        plt.plot(valid_performance, label='val_'+metric)

        plt.axvline(x=intersection_index, color='r', linestyle='--', label='Intersecion Index')
        plt.annotate(f'Optimal Value: {intersecion_val}:.4f', xy=(intersection_index, intersecion_val),
                     xycoords='data', fontsize=12, color='g')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_label)
        plt.legend(loc='lower right')

    def cm_plot(self, model):
        """
        A method to plot the confusion matrix of an input trained CNN model.

        Args:
            model: The trained CNN model.

        Returns:
            No return.
            Plots confusion matrix to display
        """
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        except ImportError("Error importing skearn methods"):
            exit(0)

        test_pred = model.predict(self.test_images)
        test_pred_labels = np.argmax(test_pred, axis=1)
        test_truth_labls = np.argmax(self.test_labels, axis=1)

        cm = confusion_matrix(test_truth_labls, test_pred_labels)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        cm_disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation='horizontal')
        plt.show()

class Simple_CNN():
    """
    A simple Convolutional Neural Network (CNN) for binary image classification using TensorFlow.

    Args:
        TRAIN_PATH (str): Path to the training dataset.
        VAL_PATH (str): Path to the validation dataset.
        HEIGHT (int, optional): Height of the input images. Defaults to 224.
        WIDTH (int, optional): Width of the input images. Defaults to 224.
        EPOCHS (int, optional): Number of epochs for training. Defaults to 20.
        BATCH_SIZE (int, optional): Size of the training batches. Defaults to 32.
        LEARNING_RATE (float, optional): Learning rate for the optimiser. Defaults to 1e-4.
        RGB (int, optional): Number of colour channels in the images. Defaults to 3.

    Attributes:
        HEIGHT (int): Height of the input images.
        WIDTH (int): Width of the input images.
        EPOCHS (int): Number of epochs for training.
        BATCH_SIZE (int): Size of the training batches.
        LEARNING_RATE (float): Learning rate for the optimiser.
        TRAIN_PATH (str): Path to the training dataset.
        VAL_PATH (str): Path to the validation dataset.
        RGB (int): Number of colour channels in the images.
    """
    def __init__(self, TRAIN_PATH, VAL_PATH, HEIGHT=224, WIDTH=224, EPOCHS=20, BATCH_SIZE=32, LEARNING_RATE=1e-4, RGB=3) -> None:
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.TRAIN_PATH = TRAIN_PATH
        self.VAL_PATH = VAL_PATH
        self.RGB = RGB


    def preprocess(self):
        """
        Preprocesses the training and validation datasets and sets up the base model for transfer learning.

        Applies data augmentation to the training dataset and prepares it for input into the model.
        Also sets up the MobileNetV2 model as the base model.

        Returns:
            tuple: A tuple containing the preprocessed training dataset, validation dataset, and the base model.
        """
        train_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, rotation_image=20, zoom_range=0.15,
                                                                     width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                                                     horizontal_flip=True, fill_mode='nearest')
        train = train_data.flow_from_directory(directory=self.TRAIN_PATH, shuffle=True, target_size=(self.HEIGHT, self.WIDTH), class_mode='binary', batch_size=self.BATCH_SIZE)

        train_ds = image_dataset_from_directory(directory=self.TRAIN_PATH, labels='inferred', label_mode='int', class_names=None,
                                                color_mode='rgb', batch_size=self.BATCH_SIZE, image_size=(self.WIDTH, self.HEIGHT), shuffle=True,
                                                seed=None, validation_split=None, subset=None, interpolation='bilinear', follow_links=False,
                                                crop_to_aspect_ratio=False, pad_to_aspect_ratio=False, data_format=None, verbose=True)

        data_augmentation = Sequential([
            Rescaling(1./255),
            RandomRotation(0.2),
            RandomZoom(0.15),
            RandomFlip('horizontal'),
            RandomTranslation(height_factor=0.2, width_factor=0.2)
        ])

        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        val_data = tf.keras.preprocessing.image.ImageDataGenerator()
        validation = val_data.flow_from_directory(directory=self.VAL_PATH, shuffle=False, target_size=(self.HEIGHT, self.WIDTH), class_mode='binary', batch_size=self.BATCH_SIZE)

        baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(self.HEIGHT, self.WIDTH, self.RGB)))
        baseModel.trainable = False

        return train_ds, validation, baseModel
    
    def model(self):
        """
        Constructs and compiles the CNN model, and trains it on the preprocessed datasets.

        Configures the model architecture, compiles it with the Adam optimiser, and trains it using the training dataset.

        Returns:
            tf.keras.Model: The trained CNN model.
        """
        train_ds, validation, baseModel = self.preprocess()

        model = Sequential([
            Lambda(mobilenet_v2.preprocess_input, name='preprocessing', input_shape=(self.HEIGHT, self.WIDTH, self.RGB)),
            AveragePooling2D(pool_size=(7, 7)),
            Flatten(name='flatten'),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])

        model.summary()

        optimiser = Adam(learning_rate=self.LEARNING_RATE, decay=self.LEARNING_RATE / self.EPOCHS)

        model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'])

        model.fit(train_ds, steps_per_epoch= train_ds.n // self.BATCH_SIZE, validation_data=validation, validation_steps=validation.n // self.BATCH_SIZE, epochs=self.EPOCHS)

        if self.save == True:
            model.save("SimpleCNNModel.model")
            
            return model
        else:
            
            return model

class CNN():
    """
    A Convolutional Neural Network (CNN) model for predicting image data with multiple output branches.

    Args:
        X_test (numpy.ndarray): Test input data.
        y_test (numpy.ndarray): Test output data.

    Attributes:
        X_test (numpy.ndarray): Test input data.
        y_test (numpy.ndarray): Test output data.
        config (object): Configuration object containing model parameters.
        N_VARS (int): Number of output variables.
    """
    def __init__(self, X_test, y_test):
        #self.input_data=input_data
        self.X_test = X_test
        self.y_test = y_test
        self.config = config()
        self.N_VARS = self.config.N_VARS

    def get_model(self):
        """
        Constructs and compiles a CNN model with multiple branches based on the number of output variables.

        Configures the CNN architecture with multiple convolutional layers and output branches. 
        Compiles the model with specified loss functions and returns it.

        Returns:
            tuple: A tuple containing the compiled CNN model and an empty list (placeholder for additional outputs).
        """
        model_config = {
                        'input_shape': (32, 32, 3), 
                        'padding': 'same',
                        'pad_out': 2,
                        'N_VARS': 3,
                        'FLUCTUATIONS_PREDICT': True
                        }
        input_shape = model_config['input_shape']
        padding = model_config['padding']
        pad_out = model_config['pad_out']
        N_VARS = model_config['N_VARS']

        input_shape = layers.Input(shape = input_shape, name = 'input_data')
        conv_1 = layers.Conv2D(64, (5, 5), padding=padding, data_format='channels_first')(self.X_test)
        batch_1 = layers.BatchNormalization(axis=1)(conv_1)
        activation_1 = layers.Activation('relu')(batch_1)
        conv_2 = layers.Conv2D(128, (3,3), padding=padding, data_format='channels_first')(activation_1)
        batch_2 = layers.BatchNormalization(axis=1)(conv_2)
        activation_2 = layers.Activation('relu')(batch_2)
        conv_3 = layers.Conv2D(256, (3,3), padding=padding, data_format='channels_first')(activation_2)
        batch_3 = layers.BatchNormalization(axis=1)(conv_3)
        activation_3 = layers.Activation('relu')(batch_3)
        conv_4 = layers.Conv2D(256, (3,3), padding=padding, data_format='channels_first')(activation_3)
        batch_4 = layers.BatchNormalization(axis=1)(conv_4)
        activation_4 = layers.Activation('relu')(batch_4)
        conv_5 = layers.Conv2D(128, (3,3), padding=padding, data_format='channels_first')(activation_4)
        batch_5 = layers.BatchNormalization(axis=1)(conv_5)
        activation_5 = layers.Activation('relu')(batch_5)

        conv_branch1 = layers.Conv2D(1, (3,3), padding=padding, data_format = 'channels_first')(activation_5)
        if (self.config.FLUCTUATIONS_PREDICT == True):
            activation_branch1 = layers.Activation('thres_relu')(conv_branch1)
            output_branch1 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                           (int(pad_out/2), int(pad_out/2))),
                                                           data_format='channels_first', name='output_branch1')(activation_branch1)
        else:
            activation_branch1 = layers.Activation('relu')(conv_branch1)
            output_branch1 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                           (int(pad_out/2), int(pad_out/2))),
                                                           data_format='channels_first', name='output_branch1')(activation_branch1)
            
        losses = {'output_branch1': 'mse'}

        if (N_VARS == 2):
            conv_branch2 = layers.conv2D(1, (3,3), padding=padding, data_format = 'channels_first')(activation_5)
            if (self.FLUCTUATIONS_PREDICT == True):
                activation_branch2 = layers.Activation('thres_relu')(conv_branch2)
                output_branch2 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch2')(activation_branch2)
            else:
                activation_branch2 = layers.Activation('relu')(conv_branch2)
                output_branch2 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch2')(activation_branch2)
                
            losses['output_branch2'] = 'mse'

        elif (N_VARS == 3):
            conv_branch2 = layers.Conv2D(1, (3,3), padding=padding, data_format = 'channels_first')(activation_5)
            if (self.FLUCTUATIONS_PREDICT == True):
                activation_branch2 = layers.Activation('thres_relu')(conv_branch2)
                output_branch2 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch2')(activation_branch2)
            else:
                activation_branch2 = layers.Activation('relu')(conv_branch2)
                output_branch2 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch2')(activation_branch2)
                
            losses['output_branch2'] = 'mse'

            conv_branch3 = layers.Conv2D(1, (3,3), padding=padding, data_format = 'channels_first')(activation_5)
            if (self.FLUCTUATIONS_PREDICT == True):
                activation_branch3 = layers.Activation('thres_relu')(conv_branch3)
                output_branch3 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch3')(activation_branch3)
            else:
                activation_branch3 = layers.Activation('relu')(conv_branch3)
                output_branch3 = layers.Cropping2D(cropping = ((int(pad_out/2), int(pad_out/2)),
                                                            (int(pad_out/2), int(pad_out/2))),
                                                            data_format='channels_first', name='output_branch3')(activation_branch3)

            outputs_model = [output_branch1, output_branch2, output_branch3]

            losses['output_branch3'] = 'mse'

        else:
            outputs_model = output_branch1

        CNN_model = tf.keras.models.Model(inputs=self.config.inputs, threshold=self.config.RELU_THRESHOLD, outputs=outputs_model)
        CNN_model.compile(optimizer='adam', loss=losses)
        return CNN_model, []
    
    def get_dataset(self):
        """
        Loads and preprocesses the dataset for training and validation.

        Splits the dataset into training and validation sets.

        Returns:
            tuple: A tuple containing the training dataset and validation dataset.
        """

        # Split the dataset into training and validation sets
        dataset_train, dataset_val = train_test_split(self.input_data, test_size=0.2, random_state=42)

        return dataset_train, dataset_val