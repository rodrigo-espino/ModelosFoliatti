from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import numpy as np
import joblib
import pandas as pd
app = Flask(__name__)
# CORS(app)

# Load the trained model
filename = 'models/knn_pipeline_foliattiGeneral_v0.pkl'
# with open(filename, 'rb') as file:
#     model = pickle.load(file)
model = joblib.load(filename)
features = ['AVG_BET','INITIAL_AMOUNT','GAMES_PLAYED_TOTAL','GAMES_WON_TOTAL','Rango_Edad_le']

@app.route('/predict', methods=['POST'])
def predict():
    print(type(model))
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data['input']], columns=features)

        print(f"Input Data: {input_data}")
        prediction = model.predict(input_data)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/ok', methods=['GET'])
def return_okey():
    return jsonify({'msg':'Ok'})

if __name__ == '__main__':
    app.run(debug=True)