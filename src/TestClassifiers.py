import yaml
import pandas as pd
import sys
import os
from scipy.io.arff import loadarff

from ML_packaging import ML, ML_post_process
from ML_packaging import ML_meta


from sklearn import datasets

def percent(i):
    return i * 10

model_in = sys.argv[2].upper()
datasets_dir = "../datasets/"

iris = datasets.load_iris()
digits = datasets.load_digits()
diabetes = datasets.load_diabetes()
linnerud = datasets.load_linnerud()
wine = datasets.load_wine()
bc = datasets.load_breast_cancer()

data_df = pd.DataFrame()
accuracy_scores = pd.DataFrame()
accuracy_list, model_list = [], []

column = []
for item in os.listdir(datasets_dir):
    column.append(str(model_in) + " " + item)

data_df['Model & Dataset'] = column
accuracy_dict = {}

if (sys.argv[1].lower() == "all"):

    ML = ML_meta(data_df, all=False, model=model_in, CNN=False, target='class', test=True)
    #ML.apply_single_model(cm=False, save_model=False, save_model_name=False, data=iris.data, target=iris.target)
    score_df, _, _, _, _, _, _, _, _, _ = ML.apply_all_models(True, data=iris.data, target=iris.target)

    print(score_df.head())

elif (sys.argv[1].lower() == "comprehensive"):
    print("-" * 100)
    print("Comprehensive sklearn model test on the datasets provided in the dataset directory \n")
    print("Please type the name of the classifier you want to test: \n")
    print("SVM")
    print("KNN")
    print("LR")
    print("RF")
    print("DT")
    print("NB")
    print("GBC")
    print("ABC")
    print("EC\n")
    print("Or Q to exit")
    input_model = str(input())
    if input_model.upper() == 'Q':
        exit(0)

    for filename in os.listdir(datasets_dir):
        if filename.endswith(".arff"):
            raw_data = loadarff(datasets_dir + filename)
            df_data = pd.DataFrame(raw_data[0])
            for col in df_data:
                if pd.api.types.is_object_dtype(df_data[col]):
                    try:
                        df_data[col] = df_data[col].astype(int)
                    except ValueError:
                        pass
            
            ML = ML_meta(df_data, all=False, model=model_in, CNN=False, target='Class/ASD', test=False)
            model, accuracy = ML.apply_single_model(cm=False, save_model=False, save_model_name=False)  
            #print("The accuracy on the " + filename + " dataset is: " + str(accuracy))
            accuracy_list.append(accuracy)
            model_list.append(model)

        elif filename.endswith(".csv"):
            df_data = pd.read_csv(datasets_dir + filename, index_col=None)
            if "titanic" in filename:
                ML = ML_meta(df_data, all=False, model=model_in, CNN=False, target='Survived', test=False)
                model, accuracy = ML.apply_single_model(cm=False, save_model=False, save_model_name=False)  
                #print("The accuracy on the " + filename + " dataset is: " + str(accuracy))
                accuracy_list.append(accuracy)
                model_list.append(model)

            elif "heart" in filename:
                ML = ML_meta(df_data, all=False, model=model_in, CNN=False, target='output', test=False)
                model, accuracy = ML.apply_single_model(cm=False, save_model=False, save_model_name=False)  
                #print("The accuracy on the " + filename + " dataset is: " + str(accuracy))
                accuracy_list.append(accuracy)
                model_list.append(model)

            elif "cancer" in filename:
                ML = ML_meta(df_data, all=False, model=model_in, CNN=False, target='Class', test=False)
                model, accuracy = ML.apply_single_model(cm=False, save_model=False, save_model_name=False)  
                #print("The accuracy on the " + filename + " dataset is: " + str(accuracy))
                accuracy_list.append(accuracy)
                model_list.append(model)
                

    accuracy_list = list(map(percent, accuracy_list))          
    data_df['Accuracy (%)'] = accuracy_list
    print(data_df)
    print("-" * 100)