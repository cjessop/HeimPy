import yaml
import pandas as pd

from ML_packaging import ML, ML_post_process
from ML_packaging import ML_meta

def load_config(filename):
    try:
        with open(filename, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            validate_config(config)  # Validate the loaded config
            return config
    except FileNotFoundError:
        print(f"Config file '{filename}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error loading config file '{filename}': {e}")
        return None

def validate_config(config):
    # Check for required keys and their types
    required_keys = ['-data', '-model', '-post_processing', '-architecture']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key '{key}' not found in config.")
    # Add additional validation as needed

# Example usage:
config_filename = "runtime.yml"
config = load_config(config_filename)
if config:
    data_filename = config['-data']['data']
    df = pd.read_csv(data_filename)
    encode = config['-data']['encode']

    model = config['-model']['model']
    all = config['-model']['all']
    search = config['-model']['search']
    save = config['-model']['save']
    save_name = config['-model']['save_model_name']
    cross_val = config['-model']['cross_val']
    cm = config['-model']['confusion']

    saved_model = config['-post_processing']['saved_model']
    predict = config['-post_processing']['predict']
    target = config['-target']['target']
    cat_cols = config['-post_processing']['cat_cols']
    con_cols = config['-post_processing']['con_cols']
    feature = config['-post_processing']['feature']

    print(target)


# if config['-architecture']['architecture'].lower() == 'keras':
#     ML = ML_meta(df, all=all, model=False, CNN=True, on_GPU=True, target=target)
#     ML.apply_CNN()

for item in config['-architectureList']['list']:
    if item == 'keras' and config['-architecture']['architecture'].lower() == 'keras':
        try:
            ML = ML_meta(df, all=all, model=False, CNN=True, on_GPU=True, target=target)
            ML.apply_CNN()
        except RuntimeError:
            raise RuntimeError
        
    elif item == 'sklearn' and config['-architecture']['architecture'].lower() == 'sklearn':
        try: 
            ML = ML_meta(df, all=all, model=model, target=target, search=search, cross_val=cross_val)
            if all == False:
                ML.apply_single_model(save_model=save, save_model_name=save_name, cm=cm)
            elif all == True:
                ML.apply_all_models(True)
        except ValueError:
            raise ValueError