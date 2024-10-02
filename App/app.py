from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from src import ML_packaging, BaseMLClasses
import pickle
import numpy as np
import logging
import joblib

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


# try: 
#     #from src.ML_packaging import ML_meta, ML_post_process, ML
#     from ..src import ML_packaging
# except ImportError as e:
#     print(f"Unable to import {e}")
#     exit(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'saved_models'
app.config['SECRET_KEY'] = 'secret'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({
            'message': f'File {filename} uploaded successfully',
            'filename': filename,
            'view_data_url': url_for('view_data', filename=filename),
            'all_url': url_for('all', filename=filename)
        })

@app.route('/view_data/<filename>')
def view_data(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        df = pd.read_csv(file_path)  # Assuming CSV file, adjust as needed
        info = {
            'filename': filename,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'head': df.head().to_html(classes='table table-striped table-bordered preview-table')
        }
        return render_template('view_data.html', info=info)
    except Exception as e:
        return f"Error: {str(e)}"
    
@app.route('/all/<filename>')
def all(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        df = pd.read_csv(file_path)
        info = {
            'filename': filename,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'head': df.head().to_html(classes='table table-striped table-bordered preview-table')
        }
        return render_template('all.html', info=info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/apply_model', methods=['POST'])
def apply_model():
    filename = request.form['filename']
    model_type = request.form['model_type']
    target_column = request.form['target_column']
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metric = 'Accuracy'
    
    if model_type == 'linear_regression':
        model_in = 'LR'
    elif model_type == 'random_forest':
        model_in = 'RF'
    elif model_type == 'naive_bayes':
        model_in = 'NB'
    elif model_type == 'decision_tree':
        model_in = 'DT'
    elif model_type == 'voting_classifier':
        model_in = 'EC'
    elif model_type == 'ada_boost':
        model_in = 'ABC'
    elif model_type == 'gradient_boost':
        model_in = 'GBC'
    elif model_type == 'support_vector_machine':
        model_in = 'SVM'
    elif model_type == 'k_nearest':
        model_in = 'KNN'

    ML_instance_save = ML_packaging.ML_meta(df, all=False, model=model_in, target=target_column, test=False, cross_val=False)
    model, accuracy = ML_instance_save.apply_single_model(cm=False, save_model='None', save_model_name='False')

    accuracy = float(accuracy)

    #session['current_model'] = model
    global current_model
    current_model = model

    result = {
        'model_type': model_type,
        'target_column': target_column,
        'metric': metric,
        'score': accuracy
    }
    
    return jsonify(result)

@app.route('/save_model', methods=['POST'])
def save_model():
    #if 'current_model' not in session:
    if current_model is None:
        return jsonify({'error': 'No model to save'}), 400
    
    #model = session['current_model']
    model = current_model
    model_type = request.form['model_type']
    target_column = request.form['target_column']
    
    # Create a unique filename
    filename = f"{model_type}_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
    
    # Save the model
    joblib.dump(model, filepath)
    
    return jsonify({'message': f'Model saved successfully as {filename}'})

def run():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    app.secret_key = 'secret'  # needed for session
    app.run(debug=True)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    app.secret_key = 'secret'  # needed for session
    app.run(debug=True)