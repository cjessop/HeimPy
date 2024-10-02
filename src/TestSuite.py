"""
Course of testing for the Heimdall code toolkit.

The structure of the file is as follows:

- The basic function definitions for the test script are defined

- The main test function that takes in the command line arguments

- Within the main test are the ``all`` test, which performs a 
  simple comparison between all sklearn toy datasets and outputs
  their accuracies

- The ``comprehensive`` test, which allows the user to select
  the machine learning model they want to test against the set
  of large datasets

- The ``search`` test allows the user to test both the randomised
  and grid search methods fro sklearn on the ``heart.csv`` dataset.
  The test outputs information comparing a hyperparamter unoptimised
  model against that of a model that has undergone hyperparameter 
  search

- The ``save`` test allows the user to save their choice of model
  from skelearn in either pickle or ONNX format (assuming that
  the sklearn2onnx package is installed on the user system).
  
"""

import yaml
import pandas as pd
import numpy as np
import sys
import os
import time
import logging
from scipy.io.arff import loadarff
from sklearn import datasets
from ML_packaging import ML, ML_post_process, ML_meta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def percent(i):
    """
    Convert a fraction to a percentage.
    
    Args:
        i (float): The fraction to convert
    
    Returns:
        float: The fraction multiplied by 100 to get the percentage
    """
    return i * 100

def load_config(config_file):
    """
    Load configuration settings from a YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file
    
    Returns:
        dict: Configuration settings as a dictionary
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def evaluate_model(ML_instance, data, target_col):
    """
    Evaluate a machine learning model using various metrics.
    
    Args:
        ML_instance (ML_meta): An instance of the ML_meta class used for model evaluation
        data (pd.DataFrame): The dataset to evaluate
        target_col (str): The name of the target column in the dataset
        
    Returns:
        tuple: A tuple containing accuracy, precision, recall, and F1 score of the model
    """
    try:
        model, predictions = ML_instance.apply_single_model(cm=False, save_model=False, save_model_name=False)
        accuracy = accuracy_score(data[target_col], predictions)
        precision = precision_score(data[target_col], predictions, average='weighted')
        recall = recall_score(data[target_col], predictions, average='weighted')
        f1 = f1_score(data[target_col], predictions, average='weighted')
        return accuracy, precision, recall, f1
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return None
    
def evaluate_model_accuracy(ML_instance, data, target_col):
    """
    Evaluate a machine learning model using just accuracy.
    
    Args:
        ML_instance (ML_meta): An instance of the ML_meta class used for model evaluation
        data (pd.DataFrame): The dataset to evaluate
        target_col (str): The name of the target column in the dataset
    
    Returns:
        tuple: A tuple containing the model and its accuracy
    """
    try:
        model, accuracy = ML_instance.apply_single_model(cm=False, save_model=False, save_model_name=False)
        #accuracy_list.append(accuracy)
        #model_list.append(model)
        return model, accuracy
    except Exception as e:
        logging.error(f"Error evaluating model accuracy: {e}")

def clean_model_name(model_str):
    # Extract just the class name from the model string
    return model_str.split('(')[0]

def report(result, n_top=3):
    """
    Utility function to report the best scoore from a randomised or ordered grid search.

    Args:
        result: The resultant model from the grid search
    
    Returns:
        None
    """
    for i in range(1, n_top+1):
        candidates = np.flatnonzero(result["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    result["mean_test_score"][candidate],
                    result["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(result["params"][candidate]))
            print("")

def main(args):
    """
    Main function to handle command-line arguments and execute the model evaluation process.
    
    Args:
        (argparse.Namespace): Command-line arguments parsed by argparse

    Returns:
        None
    """

    data_df = pd.DataFrame()
    config = load_config('config.yaml')
    logging.info("Reading config.yaml")
    datasets_dir = config['datasets_dir']
    save_dir = config['save_dir']
    
    if (args.mode == 'comprehensive'):
        model_in = args.model.upper()
        column = [f"{model_in} {item}" for item in os.listdir(datasets_dir)]
        data_df['Model & Dataset'] = column
        accuracy_dict = {}

    accuracy_scores = pd.DataFrame()
    accuracy_list, model_list = [], []
    model_list = ['RF', 'SVM', 'KNN', 'LR', 'NB', 'DT', 'VC', 'GBC', 'ABC']

    if args.mode.lower() == "all":
        iris = datasets.load_iris()
        digits = datasets.load_digits()
        wine = datasets.load_wine()
        bc = datasets.load_breast_cancer()

        dataset_list = [iris, digits, wine, bc]
        datasets_basic = ['iris', 'digits', 'wine', 'bc']
        
        data_df = pd.DataFrame({'Model & Dataset': datasets_basic})
        model_list = ['RF', 'SVM', 'KNN', 'LR', 'NB', 'DT', 'VC', 'GBC', 'ABC']
        for model in model_list:
            data_df[model] = 0.0

        logging.info(f"Initial Score DataFrame: \n {data_df.head()} \n")
        logging.info("All sklearn model tests on basic datasets")
        
        ML_instance = ML_meta(data_df, all=True, test=True)

        for idx, dataset in enumerate(dataset_list):
            dataset_name = datasets_basic[idx]
            logging.info(f"Processing dataset: {dataset_name}")
            
            try:
                scores, *_ = ML_instance.apply_all_models(True, data=dataset.data, target=dataset.target)
                logging.info(f"Scores for {dataset_name}: {scores}")
                
                if len(scores) != len(model_list):
                    logging.warning(f"Mismatch in number of scores ({len(scores)}) and models ({len(model_list)}) for {dataset_name}")
                    continue
                
                for model_name, score in zip(model_list, scores):
                    data_df.loc[data_df['Model & Dataset'] == dataset_name, model_name] = f"{score*100:.2f}%"
                    logging.info(f"Assigned {score*100:.2f}% to {dataset_name} for {model_name}")
            
            except Exception as e:
                logging.error(f"Error processing {dataset_name}: {str(e)}")
            
            print(f"After processing {dataset_name}:")
            print(data_df)
            print("/" * 100)

        print("Final DataFrame:")
        print(data_df)

        data_df.to_csv("sklearn_all_methods.csv", index=None)

    elif args.mode.lower() == "comprehensive":
        logging.info("Comprehensive sklearn model test on datasets")

        for filename in os.listdir(datasets_dir):
            file_path = os.path.join(datasets_dir, filename)
            if filename.endswith(".arff"):
                try:
                    raw_data = loadarff(file_path)
                    df_data = pd.DataFrame(raw_data[0])
                    df_data = df_data.apply(pd.to_numeric, errors='ignore')
                    target_col = 'Class/ASD'
                except Exception as e:
                    logging.error(f"Error loading ARFF file: {e}")
                    continue

            elif filename.endswith(".csv"):
                try:
                    df_data = pd.read_csv(file_path)
                    if "titanic" in filename:
                        target_col = 'Survived'
                    elif "heart" in filename:
                        target_col = 'output'
                    elif "cancer" in filename:
                        target_col = 'Class'
                    else:
                        continue
                except Exception as e:
                    logging.error(f"Error loading CSV file: {e}")
                    continue

            else:
                logging.warning(f"Unsupported file type: {filename}")
                continue

            ML_instance = ML_meta(df_data, all=False, model=model_in, CNN=False, target=target_col, test=False)
            model, accuracy = evaluate_model_accuracy(ML_instance, df_data, target_col=target_col)

            if accuracy is not None:
                accuracy_list.append(np.round(percent(accuracy), decimals=1))
                model_list.append(model_in)
                print(accuracy_list)
                print(model_list)

        data_df['Accuracy (%)'] = accuracy_list
        if os.path.exists('model_evaluation_results.csv'):
            logging.warning("Model evaluation results csv already exists, overwrite? [y/n]")
            response = input().strip()
            if response.upper() == 'y' or response.lower() == 'y':
                os.remove('model_evaluation_results.csv')
                data_df.to_csv('model_evaluation_results.csv', index=False)
                logging.info(f"Results saved to model_evaluation_results.csv \n")
                logging.info(f"Results DataFrame: \n {data_df} \n")
            else:
                sys.exit(0)

    elif args.mode == 'search':
        model_in = args.model.upper()
        for filename in os.listdir(datasets_dir):
            if filename == 'heart.csv':
                file_in = datasets_dir + filename
        logging.info(f"Grid search test on {file_in} using a {model_in} model")
        df_data = pd.read_csv(file_in)
        target_col = 'output'

        ML_instance_base = ML_meta(df_data, all=False, model=model_in, target=target_col, test=False, cross_val=False)
        input_int = int(input("Would you like to perform a randomised [1] or grid [2] search? "))
        if input_int == 1:
            search = 'random'
            ML_instance_search = ML_meta(df_data, all=False, model=model_in, target=target_col, test=False, cross_val=False, search=search)
        elif input_int == 2:
            search = 'grid'
            ML_instance_search = ML_meta(df_data, all=False, model=model_in, target=target_col, test=False, cross_val=False, search=search)

        _, accuracy_base =  ML_instance_base.apply_single_model(cm=False, save_model_name='')
        start = time.time()
        _, accuracy_search, search_ = ML_instance_search.apply_single_model(cm=False, save_model_name='')
        print(
        f"{search}SearchCV" + " took %.2f seconds for %d candidate parameter settings."
        % (time.time() - start, len(search_.cv_results_["params"])))

        logging.info(f"The base model accuracy is {accuracy_base*100:.2f}%, the hyperparamter optimised models accuracy is {accuracy_search*100:.2f}% \n")

        report(search_.cv_results_)
        logging.warning("Note: if the accuracy of the base model is higher than that of the parameter optimised model that is expected behaviour \n")

    elif args.mode == 'save':
        start = time.time()
        if not os.path.exists(save_dir):
            os.mkdir('../test_saves')
        model_in = args.model.upper()
        for filename in os.listdir(datasets_dir):
            if filename == 'heart.csv':
                file_in = datasets_dir + filename
        logging.info(f"Save model test with a {model_in} model")
        df_data = pd.read_csv(file_in)
        target_col = 'output'

        ML_instance_save = ML_meta(df_data, all=False, model=model_in, target=target_col, test=False, cross_val=False)
        input_int = int(input("Would you like to save in a pickled [1] or ONNX [2] format? "))
        if input_int == 1:
            _, accuracy_save =  ML_instance_save.apply_single_model(cm=False, save_model='pickle', save_model_name= (str(model_in) + save_dir + 'pickled_test_model.pkl'))
        elif input_int == 2:
            _, accuracy_save =  ML_instance_save.apply_single_model(cm=False, save_model='onnx', save_model_name= (str(model_in) + save_dir + 'onnx_test_model.onnx'))

        print(
        f"Training and saving the model on the heart.csv dataset" + " took %.2f seconds"
        % (time.time() - start))

    elif args.mode == 'postprocess':
        start = time.time()
        plot_list = ['output', 'corr', 'pair', 'kde']

        model_in = args.model.upper()
        for filename in os.listdir(datasets_dir):
            if filename == 'heart.csv':
                file_in = datasets_dir + filename
        logging.info(f"Post-processing and EDA on {file_in} using a {model_in} model")
        df_data = pd.read_csv(file_in)
        target_col = 'output'

        ML_post_ = ML_post_process(df_data, saved_model=None, predict=False, target=target_col, con_cols=config['cat_cols'], feature='caa')
        ML_post_.data_info()
        for plot_type in plot_list:
            try:
                ML_post_.univariate_analysis(plot_type)
            except Exception as e:
                logging.warning(f"Unable to post-process model: {e}")
        print(
        f"Post processing time" + " took %.2f seconds"
        % (time.time() - start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate machine learning models on datasets.")
    parser.add_argument('mode', choices=['all', 'comprehensive', 'search', 'save', 'postprocess'], help="Mode of operation")
    parser.add_argument('model', nargs='?', help="Model to evaluate (required for comprehensive and save mode)")
    args = parser.parse_args()

    if (args.mode == 'comprehensive' and args.model is None) or (args.mode == 'search' and args.model is None):
        parser.error("The 'model' argument is required when mode is 'comprehensive'")

    main(args)