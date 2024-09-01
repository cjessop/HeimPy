import yaml
import pandas as pd

from ML_packaging import ML, ML_post_process
from ML_packaging import ML_meta

from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()
diabetes = datasets.load_diabetes()
linnerud = datasets.load_linnerud()
wine = datasets.load_wine()
bc = datasets.load_breast_cancer()

data_df = pd.DataFrame()

ML = ML_meta(data_df, all=False, model='SVM', CNN=False, target='class', test=True)
#ML.apply_single_model(cm=False, save_model=False, save_model_name=False, data=iris.data, target=iris.target)
score_df, _, _, _, _, _, _, _, _, _ = ML.apply_all_models(True, data=iris.data, target=iris.target)
#print(score_df.head())
