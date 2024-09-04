import yaml
import pandas as pd
import sys
import os
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
    i (float): The fraction to convert.
    
    Returns:
    float: The fraction multiplied by 100 to get the percentage.
    """
    return i * 100

def load_config(config_file):
    """
    Load configuration settings from a YAML file.
    
    Args:
    config_file (str): Path to the YAML configuration file.
    
    Returns:
    dict: Configuration settings as a dictionary.
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def evaluate_model(ML_instance, data, target_col):
    """
    Evaluate a machine learning model using various metrics.
    
    Args:
    ML_instance (ML_meta): An instance of the ML_meta class used for model evaluation.
    data (pd.DataFrame): The dataset to evaluate.
    target_col (str): The name of the target column in the dataset.
    
    Returns:
    tuple: A tuple containing accuracy, precision, recall, and F1 score of the model.
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
    ML_instance (ML_meta): An instance of the ML_meta class used for model evaluation.
    data (pd.DataFrame): The dataset to evaluate.
    target_col (str): The name of the target column in the dataset.
    
    Returns:
    tuple: A tuple containing the model and its accuracy.
    """
    try:
        model, accuracy = ML_instance.apply_single_model(cm=False, save_model=False, save_model_name=False)
        #accuracy_list.append(accuracy)
        #model_list.append(model)
        return model, accuracy
    except Exception as e:
        logging.error(f"Error evaluating model accuracy: {e}")
    
def main(args):
    """
    Main function to handle command-line arguments and execute the model evaluation process.
    
    Args:
    args (argparse.Namespace): Command-line arguments parsed by argparse.
    """
    config = load_config('config.yaml')
    datasets_dir = config['datasets_dir']
    model_in = args.model.upper()

    # Initialise the ML package
    data_df = pd.DataFrame()
    accuracy_scores = pd.DataFrame()
    accuracy_list, model_list = [], []

    column = [f"{model_in} {item}" for item in os.listdir(datasets_dir)]
    data_df['Model & Dataset'] = column
    accuracy_dict = {}

    if args.mode.lower() == "all":
        ML_instance = ML_meta(data_df, all=False, model=model_in, CNN=False, target='class', test=True)
        score_df, *_ = ML_instance.apply_all_models(True, data=datasets.load_iris().data, target=datasets.load_iris().target)
        logging.info(f"Score DataFrame: {score_df.head()}")

    elif args.mode.lower() == "comprehensive":
        logging.info("Comprehensive sklearn model test on datasets")
        logging.info("Please type the name of the classifier you want to test: \n"
                     "SVM\nKNN\nLR\nRF\nDT\nNB\nGBC\nABC\nEC\nOr Q to exit")
        input_model = input().strip()
        if input_model.upper() == 'Q':
            sys.exit(0)

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
            #accuracy, precision, recall, f1 = evaluate_model(ML_instance, df_data, target_col)
            model, accuracy = evaluate_model_accuracy(ML_instance, df_data, target_col=target_col)

            if accuracy is not None:
                accuracy_list.append(percent(accuracy))
                model_list.append(model_in)
                print(accuracy_list)
                print(model_list)

        print(data_df)
        data_df['Accuracy (%)'] = accuracy_list
        data_df.to_csv('model_evaluation_results.csv', index=False)
        logging.info(f"Results saved to model_evaluation_results.csv \n")
        logging.info(f"Results DataFrame: {data_df} \n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate machine learning models on datasets.")
    parser.add_argument('mode', choices=['all', 'comprehensive'], help="Mode of operation")
    parser.add_argument('model', help="Model to evaluate")
    args = parser.parse_args()
    
    main(args)
