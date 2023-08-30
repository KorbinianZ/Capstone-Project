import json
import numpy as np
import os
import joblib

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data'][0]
        data = np.array([list(data.values())])
        #data = np.array([[2, 4, 9.9, 8.3]])
        result = model.predict(data)

        return result.tolist()
    except Exception as e:
        error = str(e)
        return error