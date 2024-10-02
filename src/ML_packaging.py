# Author: Christopher Jessop, DSTL

# HEIMDALL v0.5

from audioop import cross
from .BaseMLClasses import ML
from .BaseMLClasses import ffnn
from .CNN_file import CNN

try:
    import tensorflow as tf
    import keras
except ImportError:
    print("Unable to import Tensorflow/Keras inside of the ML packaging script")
    

#from configuration.Config import config
import pickle
import sys
import os
import numpy as np
import pandas as pd
#import plotly.express as px
from .yolo_class import Utils, YOLO_main, Object_tracker, YOLO_detector
import matplotlib.pyplot as plt
import warnings
import glob
import seaborn as sns

warnings.filterwarnings("ignore")

# Create meta class to apply all machine learning algorithms
# The ML_meta class acts as a coordinator, interacting with other classes for specific model implementations
class ML_meta:
    """
    A meta class that handles the application of all ML models. 
    The current classical models are:
    - Support Vector Machine
    - Naive Bayes
    - Decision Tree
    - Logistic Regression
    - Multi-Layered Perceptron
    - Random Forest
    - k-Nearest-Neighbour
    - Ensemble Classifier (all models combined)
    - Gradient Boosted Classifier
    - Ada Boosted Classifier

    The current Deep Learning methods are:
    - CNN
    - YOLOv8

    Includes the call to instantiate the ML class and apply test-train split
    """
    def __init__(self, data, ffnn=False, all=True, model=None, model_dict={
                                        "SupportVector": "SVM",
                                        "KNearestNeighbour": "kNN",
                                        "LinearRegression": "LinReg",
                                        "NaiveBayes": "NB",
                                        "MultiLayerPerceptron": "MLP",
                                        "DecisionTree": "DT",
                                        "RandomForest": "RF",
                                        "NeuralNetwork": "NN",
                                        "EnsembleClassifier": "EC"
                                    }, target='target', help=False, clean=False, 
                                    search=None, cross_val=False, CNN=None,
                                    on_GPU=False, YOLO=False, data_path=None,
                                    image_path=None, image_number=None, YOLO_model=None,
                                    video_path=None, video_capture=False, YOLO_train=False,
                                    YOLO_save=False, test=False):
        """
        Class initialisation - Initialise the ML_Meta class 


        Args: 
            data: input dataset in disordered format - Column labelled dataset
            ffnn: whether usage of the feed-forward neural network to make a prediction - True or False
            all: whether or not to apply all classifier models to the singe dataset - True or False
            model: the name of the model to be applied - String: default None
            model_dict: dictionary of all models and their corresponding names
            target: the name of the target feature from input dataset - String
            help: whether or not to print the help message - True or False
            clean: whether or not to delete all saved models - True or False
            search: perform grid search either randomly or evenly spaced on a grid - String of 'random' or 'grid'
            cross_val: perform k-fold cross validation - True or False
            CNN: Apply a convolutional Neural Network - True or False
            on_GPU: Run the CNN on a GPU - True or False
            YOLO: Instantiate an instance of the YOLO class for training or predicition - True or False
            data_path: The path to the dataset - default None
            image_path: The path to the image library - default None
            YOLO_model: The path or name of the trained YOLO algoritm - default None
            video_path: The path to the video on which you would like to predict on - default None
            video_capture: Use a connected image or camera sensor for live input to predict on - True or False
            YOLO_train: Flag to train a new YOLO model on, requires data_path and image_path to be not None - True or False
            test: Flag to check if the models are being used in the test script or not
    
        Returns:
            None
        """
        self.data          = data
        self.ffnn          = ffnn
        self.all           = all
        self.model         = model
        self.model_dict    = model_dict
        self.target        = target
        self.help          = help
        self.clean         = clean
        self.search        = search
        self.cross_val     = cross_val
        self.CNN           = CNN
        self.on_GPU        = on_GPU
        self.YOLO          = YOLO
        self.data_path     = data_path
        self.image_path    = image_path
        self.image_number  = image_number
        self.YOLO_model    = YOLO_model
        self.video_path    = video_path
        self.video_capture = video_capture
        self.YOLO_train    = YOLO_train
        self.YOLO_save     = YOLO_save
        self.test          = test

    def misc(self):
        """
        Handles miscellaneous operations such as displaying help information and deleting saved models.

        Args:
            None

        Returns:
            None
        """
        if self.help is True:
            print("This is a meta class that handles the application of all ML models. The current models are: Support Vector Machine, \
                  Naive Bayes, Decision Tree, Logistic Regression, Multi-Layered Perceptron, Random Forest, k-Nearest-Neighbour, Ensemble Classifier (all models combined). \
                  Includes the call to instantiate the ML class and apply test-train split \n")
            print(ML_meta.__doc__)

        if self.clean is True:
            delete_var = input("Are you sure you want to delete all saved models? (y/n)")
            if delete_var == "y" or delete_var == "Y":
                print("Deleting saved models")
                # Delete any saved models inclduing all files that end in .pkl
                for filename in os.listdir():
                    if filename.endswith(".pkl"):
                        os.remove(filename)
                    else:
                        continue
            else:
                print("Not deleting saved models")
                pass

    # Call the ML class to apply all machine learning algorithms
    def call_ML(self):
        """
        Instantiates the ML class to apply machine learning algorithms to the dataset.

        Args:
            None

        Returns:
            ML: An instance of the ML class
        """
        ml = ML(self.data) # Creates an instance of the ML class
        return ml

    #  Splits data into features (X) and target (y), with optional encoding of categorical features
    def split_data(self, encode_categorical=True, y='target', bool_missing_data=False):
        """
        Splits the dataset into features (X) and target (y), with optional encoding of categorical features.

        Args:
            encode_categorical (bool, optional): If True, encodes categorical features. Defaults to True.
            y (str, optional): Name of the target variable. Defaults to 'target'

        Returns:
            tuple: X and y data after splitting (and encoding if applicable)
        """
        ml = self.call_ML()
        X, y = ml.split_X_y(self.target)
        if encode_categorical is True:
            X, y = ml.encode_categorical(X, y)
        #if bool_missing_data:
        X, y = ml.missing_data(X, y)

        return X, y

    # Applies multiple ML models and compares their scores
    def apply_all_models(self, flag=False, data=None, target=None):
        """
        Applies multiple machine learning models to the dataset and compares their scores.

        Args:
            flag (bool, optional): If True, applies the models. Defaults to False.
            data: Input data (X_train)
            target: Label (y_train)
        
        Returns:
            None
        """

        ml = self.call_ML()

        if (self.test == False):
            X, y = self.split_data(encode_categorical=True)
        else:
            X, y = data, target

        X_train, X_test, y_train, y_test = self.call_ML().split_data(X, y)
        if flag == False:
            pass
        else:
            rf = ml.rf(X_train, X_test, y_train, y_test)
            svm = ml.svm(X_train, X_test, y_train, y_test)
            knn = ml.knn(X_train, X_test, y_train, y_test)
            lr = ml.lr(X_train, X_test, y_train, y_test)
            nb = ml.nb(X_train, X_test, y_train, y_test)
            dt = ml.dt(X_train, X_test, y_train, y_test)
            ec = ml.ec(X_train, X_test, y_train, y_test, voting='hard')
            gbc = ml.gbc(X_train, X_test, y_train, y_test)
            abc = ml.abc(X_train, X_test, y_train, y_test)
            
            models = [rf, svm, knn, lr, nb, dt, ec, gbc, abc]

            # Evaluate the performance of each model
            scores = []
            for model in models:
                score = ml.model_score(model, X_test, y_test, cross_flag=False)
                # if ("Voting" in str(model)):
                #     score_df["model VotingClassifier"] = score
                # else:
                #     score_df["model " + str(model)] = score
                scores.append(score)
                #scores.append(str(model))
            #column = [f"{item}" for item in models]
            #column = [f"Model {item}" for item in models]
            #score_df['Model'] = column

            for title in models:
                column_names = [f"Model {title[0]} accuracy: " for title in models]
                score_df = pd.DataFrame(columns=column_names)

            iterindex = 0
            for i, name in score_df.items():
                #print(iterindex)
                for label, content in score_df.items():
                    print(iterindex)
                    print(f'label: {label}')
                    print(f'content: {content}', sep='')
                #print(i, name)
                #print(score_df.loc[f'Model {title[0]} accuracy: '])
                score_df.loc[f'Model {title[0]} accuracy: '] = scores[iterindex]
                iterindex += 1 
            #score_df[[f"Model {item} accuracy" for item in models]]
            # for i, row in score_df.iterrows():
            #     print(row)
            #     score_df.loc[i, 'Accuracy'] = scores[i]
            #print(scores)
            #score_df.to_csv("test_csv.csv", index=None)

        return scores, rf, svm, knn, lr, nb, dt, ec, gbc, abc
    
    # Applies a specified single model
    def apply_single_model(self, cm=False, save_model='No', save_model_name='', data=None, target=None):
        """
        Applies a single machine learning model to the dataset.

        Args:
            cm (bool, optional): If True, plots a confusion matrix. Defaults to False
            save_model (str, optional): pickle/onnx, saves the trained model in either pickle or onnx format. Defaults to False
            save_model_name (str, optional): Name to use for the saved model file. Defaults to False

        Returns:
            Model object
            Accuracy parameter
        """
        if (self.test == False):
            # Split data into features (X) and target (y) without categorical encoding
            X, y = self.split_data(encode_categorical=True)
            X_train, X_test, y_train, y_test = self.call_ML().split_data(X, y)
        else:
            X, y = data, target
            X_train, X_test, y_train, y_test = self.call_ML().split_data(X, y)
        #  Define a dictionary mapping full model names to their abbreviated names
        self.model_dict = {
                                        "SupportVector": "SVM",
                                        "KNearestNeighbour": "kNN",
                                        "LinearRegression": "LinReg",
                                        "NaiveBayes": "NB",
                                        "MultiLayerPerceptron": "MLP",
                                        "DecisionTree": "DT",
                                        "RandomForest": "RF",
                                        "NeuralNetwork": "NN",
                                        "EnsembleClassifier": "EC",
                                        "GradientBoosted" : "GBC",
                                        "AdaBooster": "ABC"
                                    }

        model_list = []
        model_list.append(self.model)
        if self.model is not False:
            ml_single_model = ML(self.data)
            self.model_dict = {
                                        "SVM": ml_single_model.svm,
                                        "KNN": ml_single_model.knn,
                                        "LR": ml_single_model.lr,
                                        "NB": ml_single_model.nb,
                                        "MLP": ml_single_model.mlp,
                                        "DT": ml_single_model.dt,
                                        "RF": ml_single_model.rf,
                                        #"NN": ml_single_model.nn,
                                        "EC": ml_single_model.ec,
                                        "GBC": ml_single_model.gbc,
                                        "ABC": ml_single_model.abc
                                    }
            model = None
            accuracy = None

            if self.model in self.model_dict.keys():
                #print("Selected single model is " + str(self.model_dict[self.model]))
                model, accuracy = self.model_dict[self.model](X_train, X_test, y_train, y_test)
                # Perform hyperparameter tuning if requested
                if self.search is not None:
                    if self.model == "SVM":
                        param_grid = {  
                                        'C': [0.1, 1, 10, 100, 1000], 
                                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                                        'kernel': ['rbf']
                                        }
                        
                    elif self.model == "KNN":
                        param_grid = { 
                                        'n_neighbors' : [5, 7, 9, 11, 13, 15],
                                        'weights' : ['uniform', 'distance'],
                                        'metric' : ['minkowski', 'euclidean', 'manhattan']
                                        }

                    elif self.model == "NB":
                        param_grid = { 'var_smoothin' : np.logspace(0, 9, num=100)}

                    elif self.model == "RF":
                        param_grid = { 
                                        'n_estimators': [25, 50, 100, 150, 200],
                                        'max_features': ['auto', 'sqrt', 'log2', None],
                                        'max_depth': [3, 5, 7, 9, 11] 
                                        }

                    elif self.model == "DT":
                        param_grid = { 
                                        'max_features': ['auto', 'sqrt'],
                                        'max_depth': 8 
                                        }

                    elif self.model == "LR":
                        param_grid = { 'solver' : ['lbfgs', 'sag', 'saga', 'newton-cg'] }

                    elif self.model == "GBC":
                        param_grid = { 
                                        'n_estimators': [25, 50, 100, 150, 200],
                                        'max_features': ['auto', 'sqrt', 'log2', None],
                                        'max_depth': [3, 5, 7, 9, 11] 
                                        }

                    elif self.model == "ABC":
                        param_grid = { 
                                        'n_estimators': [25, 50, 100, 150, 200, 500],
                                        'algorithm': ['SAMME', 'SAMME.R', None],
                                        'learning_rate': [3, 5, 7, 9, 11], 
                                        }
                                        #'max_depth': [1, 3, 5, 7, 9, 11] }

                    else:
                        print(f"Model '{self.model}' not found in model dictionary. Available models: {list(self.model_dict.keys())}")
                        pass

                if self.search == "random":
                    random_ = ml_single_model.randomised_search(model, X_train, y_train, param_grid=param_grid)
                elif self.search == "grid":
                    grid = ml_single_model.grid_search(model, param_grid, X_train, X_test, y_train, y_test, cv=10)
                elif self.cross_val is not False:
                    ml_single_model.cross_validation(model, X_train, y_train)  
                # else:
                #     model = self.model_dict[self.model](X_train, X_test, y_train, y_test)
                 # Save the trained model if requested
                if save_model.lower() == 'pickle':
                    pickle.dump(model, open(save_model_name, 'wb'))
                elif save_model.lower() == 'onnx':
                    try: 
                        from skl2onnx import to_onnx

                        onx = to_onnx(model, X[:1])
                        with open("rf_iris.onnx", "wb") as f:
                            f.write(onx.SerializeToString())
                    except Exception as e:
                        print(f"Import Error: {e}")
                    pass
                # Plot the confusion matrix if requested
                if cm is True:
                    ML.plot_confusion_matrix(self, model, X_test, y_test)

                
                        
        self.misc()
        if self.search == "random":
            return model, accuracy, random_
        elif self.search == "grid":
            model, accuracy, grid
        else:
            return model, accuracy

    # Applies a feedforward neural network model (FFNN)
    def apply_neural_net(self):
        """
        Applies a feedforward neural network (FFNN) to the dataset for predictions.

        Args:
            None

        Returns:
            None
        """
        if self.ffnn:
            ffnn_predictor = ffnn(3, activation='sigmoid', batch_size=5)
            ml_obj = ML(self.data)
            x, Y = ml_obj.split_X_y(X='target', y='target')
            X_train, X_test, y_train, y_test = ml_obj.split_data(x, Y)

            ffnn_predictor.fit(X_train, y_train)
            ffnn_predictor.predict(X_test)

    def apply_CNN(self):
        """
        Applies the Convolutional Neural Network (CNN) architecture defined in BaseMLClasses.

        Args:
            None

        Returns:
            None
        """
        if self.CNN:
            X, y = self.split_data(encode_categorical=True)
            X_train, X_test, y_train, y_test = self.call_ML().split_data(X, y)
            _CNN = CNN(X_test, y_test)
            config_ = config()
            os.environ["CUDA_VISIBLE_DEVICES"]=str(config_.WHICH_GPU_TRAIN)

            config_.DATA_AUG = False
            config_.TRANSFER_LEARNING = False

            physical_GPUs = tf.config.list_physical_devices('GPU')
            avail_GPUs = len(physical_GPUs)

            print("TensorFlow ", tf.__version__, " GPU: ", avail_GPUs)
            print("Keras: ", keras.__version__)

            if avail_GPUs:
                try:
                    for gpu in physical_GPUs:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    print(RuntimeError)

            on_GPU = config_.ON_GPU
            num_GPU = config_.N_GPU

            if (on_GPU and num_GPU >= 1):
                distributed_training = self.on_GPU
            else:
                print("Need at least one GPU")
                exit(0)


            #Actually define the function get_dataset, or just use the regular keras model = tf.keras.Model() method
            #dataset_train, dataset_test, n_train_sample, n_validation_sample, model_config = _CNN.get_dataset(config_, train=True, distributed_training=distributed_training)
            #dataset_train, dataset_val = _CNN.get_dataset()
            
            #Again, actially define the get_model function, or just call the method and load the function in
            CNN_model, callbacks = _CNN.get_model()

            train_history = CNN_model.fit(X_train,
                                       epochs=config_.N_EPOCH,
                                       steps_per_epoch=int(np.ceil(config_.N_SAMPLES_TRAIN/config_.BATCH_SIZE)),
                                       validation_data=X_test,
                                       validation_steps=int(np.ceil(config_.N_SAMPLES_TRAIN/config_.BATCH_SIZE)),
                                       verbose=2,
                                       callbacks=callbacks)
            
            save_path='./saved_CNN_models'
            if os.path.exists(save_path) == False:
                os.mkdir(save_path)

            keras.models.save_model(CNN_model, save_path+config_.NAME,
                                overwrite=True, include_optimizer=True,
                                save_format='h5')
            
            train_loss = train_history.history['loss']
            val_loss = train_history.history['val_loss']
            tTrain = callbacks[-1].times

            np.savez(save_path+config_.NAME+'_log', tLoss=train_loss, vLoss=val_loss, tTrain=tTrain)
        else:
            print("No available GPUs for training. Please check your configuration")

    def apply_YOLO(self):
        """
        Applies or trains a YOLO model on an input video or live capture.

        Args:
            None

        Returns:
            None by default
            If video_capture is True, a window showing the current capture of the connected camera system is displayed with bounding boxes
        """
        if self.YOLO:
            detector = YOLO_main(self.data_path, self.image_path, self.image_number, self.YOLO_model, self.video_path, self.video_capture)

            if self.YOLO_train:
                try:
                    detector.train()
                except ValueError:
                    print("Invalid arguments to train method")
            
            elif(self.YOLO_train is False and self.YOLO_save is False):
                if self.video_path is not None:
                    detector.video_stream()

                else:
                    detector.cam_capture()
            
            else:
                detector.main()

        else:
            pass





class ML_post_process(ML_meta):
    """
    A class that handles the post-processing functionality of any saved ML models.

 
    """
    def __init__(self, data, saved_model=None, predict=False, target=None, con_cols=None, feature=None):
        """
        Args: 
            model: Input model saved as .pkl - Binary Machine Learning Model string name
            data: Input dataframe in the same format as the data used to test to train the model, i.e. the same labelled columns
            predict: Whether or not to predict on input data - Boolean True or False
            target: The name of the target feature - String
            con_cols: The continuous column names - String or list of strings

        univariate analysis - method that takes a string to perform exploratory data analysis on an input data set. string inputs include:
            - 'output' plots the target variable output as a bar graph
            - 'corr' plots the correlation matrices between features
            - 'pair' plots the pairwise relationships in the input dataset
            - 'kde' kernel density estimate plot of a feature against the target - input string is the name of the feature
    
    
        Returns:
            None
        """
        self.saved_model = saved_model
        #self.X_test = X_test
        self.predict = predict
        self.data= data
        self.target = target
        self.con_cols = con_cols
        self.feature = feature

    def split_data(self, encode_categorical=True, y='target'):
        """
        Function to call the split data method from BaseMLClasses.py 

        Args:
            encode_categorical - method to encode categorical data - Boolean True or False
            target -  string of the name of the target variable in the dataset - String

        Returns:
            X and y data
        """

        ml = self.call_ML()
        X, y = ml.split_X_y(self.target)
        if encode_categorical is True:
            X, y = ml.encode_categorical(X, y)

        return X, y

    def get_X_test(self):
        """
        Function to get the X_test portion of the dataset from the split. Calls to the split
        data method

        Args:
            None

        Returns:
            pandas.DataFrame: The X_test data
        """

        X, y = self.split_data()
        #X, y = self.split_data(encode_categorical=False)
        _, X_test, _, _ = self.call_ML().split_data(X, y)

        return X_test

    def load_and_predict(self): 
        """
        Function to load a saved serialised trained ML/AI model and if required make a prediction on 
        a set of input variables

        (Currently only pickled .pkl formatted networks are permitted)

        Args:
            None

        Returns:
            numpy.ndarray: The prediction results
        """
        if self.saved_model is not None:
            saved_predictions= []
            cwd = os.getcwd()
            path = str(cwd)
            pickled_model = pickle.load(open(self.model, 'rb'))

        for filename in os.listdir():
                try:
                    if filename.endswith(".pkl"):
                        file = str(glob.glob('*.pkl')[0])
                        pickled_model = pickle.load(open(file, 'rb'))
                    else:
                        continue

                except:
                    print("Error loading " + str(self.model) + " Machine Learning model")

        if self.predict == True:
            X_test = self.get_X_test()
            print(X_test)
            print(pickled_model.predict(X_test))
            result = pickled_model.predict(X_test)
            saved_predictions.append(result)

        return result

    def data_info(self):
        """
        A simple method to output various information on the dataset, such as the shape, values, and unique counts

        Args: 
            None

        Returns:
            None
        """
        print("The shape of the dataset is " + str(self.data.shape))
        print(self.data.head())
        dict = {}
        for i in list(self.data.columns):
            dict[i] = self.data[i].value_counts().shape[0]

        print(pd.DataFrame(dict, index=['Unique count']).transpose())
        print(self.data.describe().transpose())

    def target_plot(self):
        """
        Plots the target variable distribution.

        Args:
            None

        Returns:
            None
        """
        fig = plt.figure(figsize=(18,7))
        gs =fig.add_gridspec(1,2)
        gs.update(wspace=0.3, hspace=0.3)
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1])

        background_color = "#ffe6f3"
        color_palette = ["#800000","#8000ff","#6aac90","#da8829"]
        fig.patch.set_facecolor(background_color) 
        ax0.set_facecolor(background_color) 
        ax1.set_facecolor(background_color) 

        # Title of the plot
        ax0.text(0.5,0.5,"Target Count\n",
                horizontalalignment = 'center',
                verticalalignment = 'center',
                fontsize = 20,
                fontweight='bold',
                fontfamily='serif',
                color='#000000')

        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.tick_params(left=False, bottom=False)

        # Target Count
        ax1.text(0.35,177,"Output",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
        ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
        sns.countplot(ax=ax1, data = self.data, x = self.target, palette=["#8000ff","#da8829"])
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        #ax1.set_xticklabels([" "])

        ax0.spines["top"].set_visible(False)
        ax0.spines["left"].set_visible(False)
        ax0.spines["bottom"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        plt.show()

    def corr_plot(self):
        """
        Plots the correlation matrix between parameters of the input dataset.

        Args:
            None

        Returns:
            None
        """
        df_corr = self.data[self.con_cols].corr().transpose()
        df_corr
        fig = plt.figure(figsize=(10,10))
        gs = fig.add_gridspec(1,1)
        gs.update(wspace=0.3, hspace=0.15)
        ax0 = fig.add_subplot(gs[0,0])

        color_palette = ["#5833ff","#da8829"]
        mask = np.triu(np.ones_like(df_corr))
        ax0.text(1.5,-0.1,"Correlation Matrix",fontsize=22, fontweight='bold', fontfamily='serif', color="#000000")
        df_corr = df_corr[self.con_cols].corr().transpose()
        sns.heatmap(df_corr, mask=mask, fmt=".1f", annot=True, cmap='YlGnBu')

        plt.show()

        fig = plt.figure(figsize=(12,12))
        corr_mat = self.data.corr().stack().reset_index(name="correlation")
        g = sns.relplot(
            data=corr_mat,
            x="level_0", y="level_1", hue="correlation", size="correlation",
            palette="YlGnBu", hue_norm=(-1, 1), edgecolor=".7",
            height=10, sizes=(50, 250), size_norm=(-.2, .8),
        )
        g.set(xlabel="features on X", ylabel="featurs on Y", aspect="equal")
        g.fig.suptitle('Scatterplot heatmap',fontsize=22, fontweight='bold', fontfamily='serif', color="#000000")
        g.despine(left=True, bottom=True)
        g.ax.margins(.02)
        for label in g.ax.get_xticklabels():
            label.set_rotation(90)
        for artist in g.legend.legendHandles:
            artist.set_edgecolor(".7")
        plt.show()

    # def corr_plot2(self):
    #     px.imshow(self.data.corr())

    def linearality(self):
        """
        (Unused) Method intended to plot the linearity of features in the dataset.

        Args:
            None

        Returns:
            None
        """
        plt.figure(figsize=(18,18))
        for i, col in enumerate(self.data.columns, 1):
            plt.subplot(4, 3, i)
            sns.histplot(self.data[col], kde=True)
            plt.tight_layout()
            plt.plot()
        plt.show()


    def pairplot(self):
        """
        Plots the pairwise relationships between the parameters within the dataset.

        Args:
            None

        Returns:
            None
        """
        sns.pairplot(self.data, hue=self.target, palette=["#8000ff","#da8829"])
        plt.show()
        sns.pairplot(self.data, hue=self.target, kind='kde')
        plt.show()

    def kde_plot(self):
        """
        Plots the Kernel Density Estimate (KDE) to visualise the distribution of observations within the dataset.

        Args:
            None

        Returns:
            None
        """
        fig = plt.figure(figsize=(18,18))
        gs = fig.add_gridspec(1,2)
        gs.update(wspace=0.5, hspace=0.5)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1])
        bg = "#ffe6e6"
        ax0.set_facecolor(bg) 
        ax1.set_facecolor(bg) 

        fig.patch.set_facecolor(bg)
        #sns.kdeplot(ax=ax0, data=self.data, x=self.feature, hue=self.target, zorder=0, dashes=(1,5))
        ax0.text(0.5, 0.5, "Distribution of " + str(self.feature) + " to\n " + str(self.target) + "\n",
            horizontalalignment = 'center',
            verticalalignment = 'center',
            fontsize = 18,
            fontweight='bold',
            fontfamily='serif',
            color='#000000')

        ax1.text(1, 0.25, "feature",
            horizontalalignment = 'center',
            verticalalignment = 'center',
            fontsize = 14
            )
        ax0.spines["bottom"].set_visible(False)
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.tick_params(left=False, bottom=False)

        ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
        sns.kdeplot(ax=ax1, data=self.data, x=self.feature, hue=self.target, alpha=0.7, linewidth=1, fill=True, palette=["#8000ff","#da8829"])
        ax1.set_xlabel("")
        ax1.set_ylabel("")

        for i in ["top","left","right"]:
            ax0.spines[i].set_visible(False)
            ax1.spines[i].set_visible(False)
        #sns.kdeplot(data=self.data, x=self.feature, hue=self.target, dashes=(1,5), alpha=0.7, linewidth=0, palette=["#8000ff","#da8829"])
        plt.show()

    def univariate_analysis(self, output_plot=None):
        """
        Performs univariate analysis for the features within the dataset, calling one of the available methods.

        Args:
            output_plot (str, optional): Type of analysis to perform. One of: 'output', 'corr', 'pair', 'kde', or 'linearality'

        Returns:
            None
        """
        try:
            if output_plot == 'output':
                self.target_plot()
            elif output_plot == 'corr':
                self.corr_plot()
            elif output_plot == 'pair':
                self.pairplot()
            elif output_plot == 'kde':
                self.kde_plot()
            elif output_plot == 'linearality':
                self.linearality()
        except ValueError:
            print("Invalid argument given to method, please select one of: 'output', 'corr', 'pair', 'kde', or 'linerality'")

if __name__ == "__main__":
    # Initialise the meta class
    meta_obj = ML_meta(data, all=False, model="MLP")
    meta_obj.apply_single_model()
    
