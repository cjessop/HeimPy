import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import builtins
import yaml
import re
#from configuration.Config import config
from importlib import import_module

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

try:
    import tensorflow as tf
    from keras import layers
    from keras.models import Sequential
    from keras.layers import Input
    from keras.layers import Dense, Dropout, Lambda, AveragePooling2D, Flatten
    from keras.layers import Rescaling, RandomContrast, RandomZoom, RandomTranslation, RandomBrightness, RandomRotation
    from keras.layers import RandomFlip, RandomCrop
    from keras.utils import image_dataset_from_directory    
    from keras.callbacks import EarlyStopping
    from keras.models import load_model
    from keras.optimizers import Adam
    from keras.applications import mobilenet_v2
    from keras.applications import MobileNetV2
except ImportError:
    print("Unable to Import Tensorflow/Keras inside of the Base Classes script")
    exit(0)

from abc import ABC, abstractmethod
import inspect

model_dict = {
    "SupportVector": "SVM",
    "KNearestNeighbour": "kNN",
    "LinearRegression": "LinReg",
    "NaiveBayes": "NB",
    "MultiLayerPerceptron": "MLP",
    "DecisionTree": "DT",
    "RandomForest": "RF",
    "NeuralNetwork": "NN"
}

class BasePredictor(ABC):
    """
    Base class for predictive models.

    Defines the core functionalities expected from predictive models, such as fitting and predicting.
    Also provides utility methods for parameter management and model resetting/loading.

    Attributes:
        _get_param_names (classmethod): Retrieves the names of parameters for the class's constructor.
        get_params (method): Returns a dictionary of the class's hyperparameters.
        reset (method): Creates a new instance of the class with the same parameters as the current instance.
        load_params (method): Loads new parameters into the class instance.

    Abstract Methods:
        fit (abstractmethod): Trains the model on given data.
        predict (abstractmethod): Makes predictions on new data using the trained model.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Trains the model on the given data.

        Args:
            X (pandas.DataFrame): Input features.
            y (pandas.Series): Target variable.

        Raises:
            NotImplementedError: This is an abstract method and should be
                implemented in the derived class.
        """

        raise NotImplementedError("fit method must be implemented in the derived class.")

    @abstractmethod
    def predict(self, X):
        """
        Makes predictions on new data using the trained model.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            numpy.ndarray: Predicted values.

        Raises:
            NotImplementedError: This is an abstract method and should be
                implemented in the derived class.
        """
        
        raise NotImplementedError("predict method must be implemented in the derived class.")

    @classmethod 
    def _get_param_names(cls):
        """
        Retrieves the names of parameters for the class's constructor.

        Returns:
            list: A sorted list of parameter names, excluding 'self'.
        """
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return [] 
        
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError('scikit-learn estimators should always '
                                   'specify their parameters in the signature'
                                   ' of their __init__ (no varargs).')
            
        return sorted([p.name for p in parameters])
    
    def get_params(self, deep=True):
        """
        Returns a dictionary of the class's hyperparameters.

        Args:
            deep (bool, optional): If True, includes parameters for sub-objects that have a get_params() method. Defaults to True.

        Returns:
            dict: A dictionary of parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items) 
            out[key] = value 
             
        return out 

    def reset(self):
        """
        Resets the model to its initial state.

        Returns:
            BasePredictor: A new instance of the class with the same parameters.
        """
        new = self.__class__(**self.get_params())
        return new

    def load_params(self, params=None):
        """
        Loads new parameters into the class instance.

        Args:
            params (dict, optional): Dictionary of new parameters. Defaults to None.

        Returns:
            BasePredictor: Updated instance of the class.
        """
        self = self.__class__(**params)
        print("params loaded")
        return self
    
def import_function(qualified_name):
    """
    Imports a callable object (function, class, etc.) based on its qualified name.

    Args:
        qualified_name (str): The fully qualified name of the callable.

    Returns:
        callable: The imported callable object.
    """
    path = qualified_name.split('.')
    module = builtins

    for i, key in enumerate(path[1:]):
        if not hasattr(module, key):
            module = import_module('.'.join(path[:i+2]))
        else:
            module = getattr(module, key)
        
    function = getattr(module, path[-1])
    
    return function

class PipelineLoader(yaml.SafeLoader):
    """
    A custom YAML loader for pipeline configurations.
    """
    @classmethod
    def load(cls, instream):
        """
        Loads a YAML stream and returns the data.

        Args:
            instream (file-like object): The input stream to read from.

        Returns:
            dict: The parsed YAML data.
        """
        loader = cls(instream)

        try:
            return loader.get_single_data()
        finally:
            loader.dispose()

    def construct_map(self, node, deep=False):
        """
        Constructs a mapping (dictionary) from a YAML node.

        Args:
            node (yaml.Node): The YAML node to construct the mapping from.
            deep (bool, optional): Whether to deeply construct the mapping. Defaults to False.

        Returns:
            dict: The constructed mapping.
        """
        mapping = super().construct_mapping(node, deep)

        for key in mapping:
            if not isinstance(key, str):
                raise ValueError(f'key {key} is not the correct type (string), it is {type(key).__name__} \n'
                                 f'{node.start_mark}')
            
        return mapping
    
    def construct_ref(self, node):
        """
        Constructs a reference from a YAML node.

        Args:
            node (yaml.Node): The YAML node containing the reference.

        Returns:
            Ref: A Ref object constructed from the reference.
        """
        ref = self.construct_scalar(node)
        
        if ref[:1] == "$":
            ref = ref[1:]
        if not ref:
            raise ValueError(f'An empty reference was provided {node.start_mark}')
        
        return Ref(ref)
    
    def construct_call(self, name, node):
        """
        Constructs a callable object from a YAML node.

        Args:
            name (str): The name of the object to call.
            node (yaml.Node): The YAML node containing arguments for the call.

        Returns:
            Call: A Call object representing the function call.
        """
        try: 
            object = import_function(name)
        except (ModuleNotFoundError, AttributeError) as err:
            raise ImportError(f'{err} ') from err
        
        if isinstance(node, yaml.ScalarNode):
            if node.value:
                raise ValueError(f'The ScalarNode {node.value} has to be empty to import the object')
            return object
        
        else:
            if isinstance(node, yaml.SequenceNode):
                args = self.construct_sequence(node)
                kwargs = {}
            elif isinstance(node, yaml.MappingNode):
                args = []
                kwargs = self.construct_map(node, deep=False)

            return Call(object, args, kwargs)
    
    @staticmethod
    def load_pipeline_yaml(filename):
        """
        Loads a pipeline configuration from a YAML file.

        Args:
            filename (str): The path to the YAML file.

        Returns:
            dict: The loaded pipeline configuration.
        """
        with open(filename, 'r') as data:
            return PipelineLoader.load(data) or {}

class ML(BasePredictor):
    """
    Class containing various methods to format data and apply machine learning algorithms.
    """
    def __init__(self, data):
        """
        Initialises the ML class with the given data.

        Args:
            data (pandas.DataFrame): The dataset to work with.
        """
        self.data = data

    def split_X_y(self, y):
        """
        Splits the data into features (X) and target variable (y).

        Args:
            y (str): Name of the target variable column.

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector.
        """
        X = self.data.drop(y, axis=1)
        y = self.data[y]
        return X, y
    
    def encode_categorical(self, X, y):
        """
        Encodes categorical features in the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (pandas.Series): Target variable.

        Returns:
            tuple: (X, y) with encoded categorical features.
        """
        X = pd.get_dummies(X, drop_first=True)
        y = pd.get_dummies(y, drop_first=True)
        return X, y
    
    def missing_data(self, X, y, strategy='mean'):
        """
        Handles missing data in the dataset.

        Args:
            X (pandas.DataFrame): Input features.
            y (pandas.Series): Target variable.
            strategy (str, optional): Imputation strategy for missing values. Defaults to 'mean'.

        Returns:
            tuple: (X, y) with imputed missing values.
        """
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        X = imputer.fit_transform(X)
        y = imputer.fit_transform(y)
        return X, y

    def extract_features(self, X, y, test_size=0.2):
        """
        Extracts features from classification data.

        Args:
            X (pandas.DataFrame): Input features.
            y (pandas.Series): Target variable.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def split_data(self, X, y, test_size=0.2):
        """
        Splits data into training and testing sets.

        Args:
            X (pandas.DataFrame): Input features.
            y (pandas.Series): Target variable.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def scale_data(self, X_train, X_test):
        """
        Scales the data using standardisation.

        Args:
            X_train (pandas.DataFrame): Training features.
            X_test (pandas.DataFrame): Test features.

        Returns:
            tuple: (X_train, X_test) with scaled features.
        """
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test
    
    def prepare_data(self, name, label, columns, end_range, index_column=False):
        """
        Prepares data for modelling.

        Args:
            name (str): Name of the dataset file.
            label (str): Name of the target variable column.
            columns (list): List of columns to drop.
            end_range (int): End range for iterating over the dataset.
            index_column (bool, optional): Whether to use the first column as an index. Defaults to False.
        """
        df1 = pd.read_csv(self.data, index_col=index_column)
        df1['class'] = label
        drop_columns = columns

        for item in drop_columns:
            df1 = df1.drop(item, axis=1)

        Total_df = pd.DataFrame()

        for i in range(0, end_range):
            Total_df = Total_df.append(df1)

    def lr(self, X_train, X_test, y_train, y_test):
        """
        Trains a logistic regression model.

        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target variable.
            X_test  (pandas.DataFrame): Testing features.
            y_test  (pandas.Series): Testing target variable.

        Returns:
            LogisticRegressionModel: Trained logistic regression model.
        """
        logmodel = LogisticRegression()
        logmodel.fit(X_train, y_train)
        predictions = logmodel.predict(X_test)
        #print(classification_report(y_test, predictions))
        return logmodel
    
    def knn(self, X_train, X_test, y_train, y_test, n_neighbors=5):
        """
        Trains a k-nearest neighbours (KNN) model.

        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target variable.
            X_test  (pandas.DataFrame): Testing features.
            y_test  (pandas.Series): Testing target variable.
            n_neighbors (int, optional): Number of neighbours to consider. Defaults to 5.

        Returns:
            KNNModel: Trained KNN model.
        """
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        #print(classification_report(y_test, pred))
        return knn

    def svm(self, X_train, X_test, y_train, y_test, kernel='rbf'):
        """
        Trains a Support Vector Machine (SVM) model.

        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target variable.
            X_test  (pandas.DataFrame): Testing features.
            y_test  (pandas.Series): Testing target variable.
            kernel  (str, optional): Desired SVM kernel. Defaults to 'rbf'.

        Returns:
            svc_model: Trained SVM model.
        """
        svc_model = SVC(kernel=kernel)
        svc_model.fit(X_train, y_train)
        predictions = svc_model.predict(X_test)
        #print(classification_report(y_test, predictions))
        return svc_model
    
    def dt(self, X_train, X_test, y_train, y_test, max_depth=8):
        """
        Trains a decision tree classifier.

        Args:
            X_train (pandas.DataFrame): Training features.
            X_test (pandas.DataFrame): Test features.
            y_train (pandas.Series): Training target variable.
            y_test (pandas.Series): Test target variable.
            max_depth (int, optional): Maximum depth of the tree. Defaults to 8.

        Returns:
            DecisionTreeClassifier: Trained decision tree model.
        """
        dtree = DecisionTreeClassifier(max_depth=max_depth)
        dtree.fit(X_train, y_train)
        predictions = dtree.predict(X_test)
        #print(classification_report(y_test, predictions))
        return dtree
    
    def rf(self, X_train, X_test, y_train, y_test, n_estimators=100, max_depth=8):
        """
        Trains a random forest classifier.

        Args:
            X_train (pandas.DataFrame): Training features.
            X_test (pandas.DataFrame): Test features.
            y_train (pandas.Series): Training target variable.
            y_test (pandas.Series): Test target variable.
            n_estimators (int, optional): Number of trees in the forest. Defaults to 100.
            max_depth (int, optional): Maximum depth of the trees. Defaults to 8.

        Returns:
            RandomForestClassifier: Trained random forest model.
        """
        rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rfc.fit(X_train, y_train)
        predictions = rfc.predict(X_test)
        #print(classification_report(y_test, predictions))
        return rfc
    
    def nb(self, X_train, X_test, y_train, y_test):
        """
        Trains a naive bayes classifier.

        Args:
            X_train (pandas.DataFrame): Training features.
            X_test (pandas.DataFrame): Test features.
            y_train (pandas.Series): Training target variable.
            y_test (pandas.Series): Test target variable.

        Returns:
            MultinomialNB: Trained naive bayes model.
        """
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        predictions = nb.predict(X_test)
        #print(classification_report(y_test, predictions))
        return nb

    def gbc(self, X_train, X_test, y_train, y_test, random_state=1):
        """
        Trains a gradient boosted classifier.

        Args:
            X_train (pandas.DataFrame): Training features.
            X_test (pandas.DataFrame): Test features.
            y_train (pandas.Series): Training target variable.
            y_test (pandas.Series): Test target variable.
            random_state (int, optional): Random state for reproducibility. Defaults to 1.

        Returns:
            GradientBoostingClassifier: Trained gradient boosted classifier model.
        """
        gbc = GradientBoostingClassifier(random_state=random_state)
        gbc.fit(X_train, y_train)
        predictions = gbc.predict(X_test)
        #print(classification_report(y_test, predictions))
        return gbc

    def abc(self, X_train, X_test, y_train, y_test):
        """
        Trains an AdaBoost classifier.

        Args:
            X_train (pandas.DataFrame): Training features.
            X_test (pandas.DataFrame): Test features.
            y_train (pandas.Series): Training target variable.
            y_test (pandas.Series): Test target variable.

        Returns:
            AdaBoostClassifier: Trained AdaBoost classifier model.
        """
        abc = AdaBoostClassifier()
        abc.fit(X_train, y_train)
        predictions = abc.predict(X_test)
        #print(classification_report(y_test, predictions))
        return abc
    
    def ec(self, X_train, X_test, y_train, y_test, voting='hard', random_state=1):
        """
        Trains an ensemble classifier using voting.

        Args:
            X_train (pandas.DataFrame): Training features.
            X_test (pandas.DataFrame): Test features.
            y_train (pandas.Series): Training target variable.
            y_test (pandas.Series): Test target variable.
            voting: Voting method for final classification (hard or soft voting).

        Returns:
            VotingClassifier: Trained ensemble classifier model.
        """
        clf1 = LogisticRegression(random_state=random_state)
        clf2 = RandomForestClassifier(random_state=random_state)
        clf3 = GaussianNB()
        clf4 = SVC(random_state=random_state)
        clf5 = DecisionTreeClassifier(random_state=random_state)
        clf6 = KNeighborsClassifier()

        estimators = [('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svc', clf4), ('dt', clf5), ('knn', clf6)]

        eclf = VotingClassifier(estimators=estimators, voting=voting)
        eclf.fit(X_train, y_train)
        predictions = eclf.predict(X_test)
        #print(classification_report(y_test, predictions))
        return eclf
    
    def cross_validation(self, model, X, y, cv=5):
        """
        Applies K-fold cross-validation to a selected model.

        Args:
            model: The machine learning model to apply K-fold CV to.
            X: The feature data.
            y: The target variable.
            cv: Number of folds in a stratified KFold.

        Returns:
            array: Scores from K-fold cross-validation for the selected model.
        """
        scores = cross_val_score(model, X, y, cv=cv)
        #print(scores)
        #print(scores.mean())
        return scores

    def grid_search(self, model, param_grid, X_train, X_test, y_train, y_test, cv=10):
        """
        Performs grid search to find the best hyperparameters.

        Args:
            model: The machine learning model to tune.
            param_grid (dict): The parameter grid to search over.
            X_train (pandas.DataFrame): Training features.
            X_test (pandas.DataFrame): Test features.
            y_train (pandas.Series): Training target variable.
            y_test (pandas.Series): Test target variable.
            cv (int, optional): Number of cross-validation folds. Defaults to 10.

        Returns:
            GridSearchCV: The fitted grid search object.
        """
        grid = GridSearchCV(model, param_grid, cv=cv)
        grid.fit(X_train, y_train)
        grid_predict = grid.predict(X_test)
        #print(classification_report(grid_predict, y_test))
        print(grid.best_params_)
        print(grid.best_estimator_)
        return grid
    
    def randomised_search(self, model, X, y, cv=5, n_iter=100, param_grid=None):
        """
        Performs randomised search to find the best hyperparameters.

        Args:
            model: The machine learning model to tune.
            X (pandas.DataFrame): Feature data.
            y (pandas.Series): Target variable.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
            n_iter (int, optional): Number of parameter settings sampled. Defaults to 100.
            param_grid (dict, optional): The parameter grid to search over. Defaults to None.

        Returns:
            RandomizedSearchCV: The fitted randomised search object.
        """
        random = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=n_iter)
        random.fit(X, y)
        print(random.best_params_)
        print(random.best_estimator_)
        return random
    
    def mlp(self, X_train, X_test, y_train, y_test, hidden_layers=1, neurons=8, activation='relu', optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], epochs=25, batch_size=32, validation_split=0.2, verbose=1):
        """
        Trains a Multi-Layer Perceptron (MLP) model.

        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target variable.
            X_test (pandas.DataFrame): Testing features.
            y_test (pandas.Series): Testing target variable.
            hidden_layers (int, optional): Number of hidden layers. Defaults to 1.
            neurons (int, optional): Number of neurons per hidden layer. Defaults to 8.
            activation (str, optional): Activation function for hidden layers. Defaults to 'relu'.
            optimizer (str, optional): Optimiser to use. Defaults to 'adam'.
            loss (str, optional): Loss function. Defaults to 'binary_crossentropy'.
            metrics (list, optional): List of metrics to evaluate. Defaults to ['accuracy'].
            epochs (int, optional): Number of training epochs. Defaults to 25.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            validation_split (float, optional): Proportion of training data to use as validation set. Defaults to 0.2.
            verbose (int, optional): Verbosity mode. Defaults to 1.

        Returns:
            Sequential: Trained MLP model.
        """
        model = Sequential()
        model.add(Dense(neurons, activation=activation))
        for i in range(hidden_layers):
            model.add(Dense(neurons, activation=activation))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
        model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose, callbacks=[early_stop])
        loss_df = pd.DataFrame(model.history.history)
        loss_df.plot()
        predictions = model.predict(X_test)
        print(predictions)
        return model
    
    def plot_confusion_matrix(self, model, X_test, y_test):
        """
        Plots a confusion matrix for the given model's predictions.

        Args:
            model: The trained machine learning model.
            X_test (pandas.DataFrame): Test features.
            y_test (pandas.Series): Test target variable.

        Returns:
            ConfusionMatrix: A seaborn heatmap of the confusion matrix.
        """
        predictions = model.predict(X_test)
        predictions = np.round(predictions)
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True)
        plt.show()
        return cm
    
    def compare_classifier_reports(self, models, X_test, y_test):
        """
        Compares classification reports for multiple models.

        Args:
            models (list): List of trained models.
            X_test (pandas.DataFrame): Test features.
            y_test (pandas.Series): Test target variable.
        """
        for model in models:
            predictions = model.predict(X_test)
            print(classification_report(y_test, predictions))

    def find_best_model(self, models, X_test, y_test):
        """
        Identifies the best model based on accuracy.

        Args:
            models (list): List of trained models.
            X_test (pandas.DataFrame): Test features.
            y_test (pandas.Series): Test target variable.

        Returns:
            tuple: The best model and its accuracy.
        """
        best_model = None
        best_accuracy = 0
        for model in models:
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        return best_model, best_accuracy

    def model_score(self, model, X_test, y_test, cross_flag=False):
        """
        Evaluates the model using repeated stratified K-fold cross-validation.

        Args:
            model: The trained machine learning model.
            X_test (pandas.DataFrame): Test features.
            y_test (pandas.Series): Test target variable.
            cross_flag: Flag to check if the model score is determined via cross validation

        Returns:
            array: Cross-validation scores.
        """

        if (cross_flag == True):
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)

            return scores
        
        else:
            predictions = model.predict(X_test)
            scores = accuracy_score(y_test, predictions)

            return scores
        
        

    def fit(self, X, y):
        """
        Placeholder for model training method.
        """
        pass

    def predict(self, model, X):
        """
        Makes predictions using the given model.

        Args:
            model: The trained machine learning model.
            X (pandas.DataFrame): Input features.

        Returns:
            numpy.ndarray: Predicted values.
        """
        prediction = model.predict(X)
        return prediction

class svm(ML):
    """
    A class for running Support Vector Machine (SVM) models.
    """
    def run(self, X_train, X_test, y_train, y_test, kernel='rbf'):
        """
        Trains and runs an SVM model.

        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target variable.
            X_test (pandas.DataFrame): Test features.
            y_test (pandas.Series): Test target variable.
            kernel (str, optional): Kernel type for SVM. Defaults to 'rbf'.

        Returns:
            SVC: Trained SVM model.
        """
        svc_model = SVC(kernel=kernel)
        svc_model.fit(X_train, y_train)
        predictions = svc_model.predict(X_test)
        print(classification_report(y_test, predictions))
        return svc_model

class ffnn(BasePredictor):
    """
    Class for creating and running a Feedforward Neural Network (FFNN).
    """
    def __init__(self, hidden_layers=[], dropout=0, epochs=5, activation=[], batch_size=None):
        """
        Initialises the ffnn class for creating a feedforward neural-network.

        Args:
            hidden_layers (list, optional): List of hidden layer sizes. Defaults to [].
            dropout (float, optional): Dropout rate for regularisation. Defaults to 0.
            epochs (int, optional): Number of training epochs. Defaults to 5.
            activation (list, optional): List of activation functions for hidden layers. Defaults to [].
            batch_size (int, optional): Batch size for training. Defaults to None.
        """
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.epochs = epochs
        self.activation = activation
        self.batch_size = batch_size

        self.model = Sequential()
        for i in range(len(self.hidden_layers)):
            if i == 0:
                self.model.add(Dense(self.hidden_layers[i], activation=self.activation[i], input_dim=self.hidden_layers[i]))
            else:
                self.model.add(Dense(self.hidden_layers[i], activation=self.activation[i]))
            if self.dropout > 0:
                self.model.add(Dropout(self.dropout))

        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, X, y):
        """
        Trains the feedforward neural network.

        Args:
            X (pandas.DataFrame): Input features.
            y (pandas.Series): Target variable.
        """
        try: 
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
        except ValueError as e:
            print(f'Error in fitting the model: {e}')
            pass

    def predict(self, X):
        """
        Makes predictions using the trained feedforward neural network.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            numpy.ndarray: Predicted values.
        """
        try:
            return self.model.predict(X)
        except ValueError as e:
            print(f'Error in predicting the model: {e}')
            pass
# Generate some data to test the class using numpy and pandas
data = np.random.randint(0, 100, (1000, 50))
data = pd.DataFrame(data)
data['target'] = np.random.randint(0, 2, 1000)
