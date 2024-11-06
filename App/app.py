from flask import Flask, render_template, request, jsonify
from flask import json, redirect, url_for, session
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from src import ML_packaging, BaseMLClasses
import pickle
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
        return model, accuracy
    except Exception as e:
        logging.error(f"Error evaluating model accuracy: {e}")

def json_serialisable(obj):
    if isinstance(obj, (np.int64, np.float64)):
        return int(obj) if isinstance(obj, np.int64) else float(obj)
    elif pd.isna(obj):
        return None
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)
    
    logging.basicConfig(level=logging.DEBUG)

class RobustJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, (np.int64, np.float64)):
                return int(obj) if isinstance(obj, np.int64) else float(obj)
            elif pd.isna(obj):
                return None
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return super(RobustJSONEncoder, self).default(obj)
        except:
            return str(obj)

def robust_json_dumps(obj):
    return json.dumps(obj, cls=RobustJSONEncoder)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'saved_models'
app.config['SECRET_KEY'] = 'secret'
app.config['TRAIN_PATH'] = 'train_path'
app.config['TEST_PATH'] = 'test_path'

@app.template_filter('tojson')
def tojson_filter(s):
    return json.dumps(s)

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
    
    #try:
    df = pd.read_csv(file_path) 
    app.logger.info(f"Successfully read CSV. Shape: {df.shape}")

    plots_data = {
        'histograms': {},
        'boxplots': {},
        'bar_charts': {},
        'correlation': {}
    }

    # Histogram and boxplots for numerical columns
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        data = df[col].dropna().tolist()
        plots_data['histograms'][col] = {
            'x': data,
            'type': 'histogram',
            'name': col
        }
        plots_data['boxplots'][col] = {
            'y': data,
            'type': 'box',
            'name': col
        }

    # Bar charts for categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        value_counts = df[col].value_counts()
        plots_data['bar_charts'][col] = {
            'x': value_counts.index.tolist(),
            'y': value_counts.values.tolist(),
            'type': 'bar',
            'name': col
        }

    # Correlation heatmap
    corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr().fillna(0)

    plots_data['correlation'] = {
        'z': corr_matrix.values.tolist(),
        'x': corr_matrix.columns.tolist(),
        'y': corr_matrix.columns.tolist(),
        'type': 'heatmap'
    }
    if 'correlation' not in plots_data or plots_data['correlation'] is None:
        plots_data['correlation'] = {}

    stats_info = {
        'length': len(df.columns),
        'obs_num': df.shape[0],
        'missing_values': int(df.isnull().sum().sum()),
        'duplicates': int(df.duplicated().sum()),
        
        'average_val': df.mean().to_dict(),
        'mode_val': df.mode().iloc[0].to_dict(),
        'median_val': df.median().to_dict(),
        'std_val': df.std().to_dict(),
        'min_val': df.min().to_dict(),
        'max_val': df.max().to_dict(),
        'q1_val': df.quantile(0.25).to_dict(),
        'q3_val': df.quantile(0.75).to_dict(),
        'skew_val': df.skew().to_dict(),
        'kurt_val': df.kurtosis().to_dict(),
        
        'num_columns': len(df.select_dtypes(include=['int64', 'float64']).columns),
        'cat_columns': len(df.select_dtypes(include=['object']).columns),
        'date_columns': len(df.select_dtypes(include=['datetime64']).columns),
        
        'correlation_matrix': df.corr().round(2).to_html(classes='table table-bordered table-striped'),
        
        'value_counts': {col: df[col].value_counts().head().to_dict() for col in df.select_dtypes(include=['object']).columns}
    }

    info = {
        'filename': filename,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'head': df.head().to_html(classes='table table-striped table-bordered preview-table'),   
    }

    print(plots_data)
    # Convert plots_data to JSON using our robust encoder
    plots_data_json = robust_json_dumps(plots_data)

    # Pass the JSON string directly to the template
    return render_template('view_data.html', info=info, stats_info=stats_info, plots_data=plots_data_json)
    # except Exception as e:
    #     app.logger.error(f"Error in view_data: {str(e)}", exc_info=True)
    #     return f"Error in view_data: {str(e)}", 500

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
    
@app.route('/CNN')
def CNN():
    return render_template('CNN.html')

@app.route('/CNN_build', methods=['POST'])
def build_CNN():
    train_dir = request.form['train_path']
    return jsonify({'message': 'CNN building not implemented yet'})

@app.route('/apply_model', methods=['POST'])
def apply_model():
    global save_state, current_model

    filename = request.form['filename']
    model_type = request.form['model_type']
    target_column = request.form['target_column']
    save_state = request.form['save_state']

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metric = 'Accuracy'
    
    model_mapping = {
        'linear_regression': 'LR',
        'random_forest': 'RF',
        'naive_bayes': 'NB',
        'decision_tree': 'DT',
        'voting_classifier': 'EC',
        'ada_boost': 'ABC',
        'gradient_boost': 'GBC',
        'support_vector_machine': 'SVM',
        'k_nearest': 'KNN'
    }

    model_in = model_mapping.get(model_type, 'LR')  # Default to LR if not found

    ML_instance_save = ML_packaging.ML_meta(df, all=False, model=model_in, target=target_column, test=False, cross_val=False)
    model, accuracy = ML_instance_save.apply_single_model(cm=False, save_model='None', save_model_name='False')

    accuracy = float(accuracy)

    current_model = model

    result = {
        'model_type': model_type,
        'target_column': target_column,
        'metric': metric,
        'score': accuracy
    }
    
    return jsonify(result)

@app.route('/apply_all', methods=['POST'])
def apply_all():
    filename = request.form['filename']
    target_column = request.form['target_column']
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    metric = 'Average Accuracy'
    
    ML_instance = ML_packaging.ML_meta(df, all=True, test=True)
    scores, _, _, _, _, _, _, _, _, _, max_model = ML_instance.apply_all_models(True, data=X, target=y)

    accuracy = np.mean(scores)

    result = {
        'target_column': target_column,
        'metric': metric,
        'score': accuracy,
        'best_model': max_model[0],
        'best_model_accuracy': max_model[1]
    }
    
    return jsonify(result)

@app.route('/save_model', methods=['POST'])
def save_model():
    if current_model is None:
        return jsonify({'error': 'No model to save'}), 400
    
    model = current_model
    model_type = request.form['model_type']
    target_column = request.form['target_column']
    
    filename = f"{model_type}_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    filepath = os.path.join(app.config['MODEL_FOLDER'], filename)

    if save_state == 'pickle':
        pickle.dump(model, open(filepath, 'wb'))
    elif save_state == 'onnx':
        try: 
            from skl2onnx import to_onnx
            onx = to_onnx(model, X[:1])
            with open(filepath, 'wb') as f:
                f.write(onx.serialiseToString())
        except Exception as e:
            return jsonify({'error': f'Error saving as ONNX: {str(e)}'}), 500
    elif save_state == 'joblib':
        joblib.dump(model, filepath)
    else:
        return jsonify({'error': 'Invalid save format'}), 400

    return jsonify({'message': f'Model saved successfully as {filename}'})
    
@app.route('/dataset-guide')
def dataset_guide():
    return render_template('dataset_guide.html')

@app.route('/cnn_guide')
def cnn_guide():
    return render_template('CNN_guide.html')

def run():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    app.secret_key = 'secret'  
    app.run(debug=True)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    app.secret_key = 'secret'  
    app.run(debug=True)