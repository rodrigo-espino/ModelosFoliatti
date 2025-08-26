from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from func import calcular_edad_y_rango_encoded
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)
# CORS(app)

# Load the trained model
filename_cluster = 'models/knn_pipeline_foliattiGeneral_v0.pkl'
model_cluster = joblib.load(filename_cluster)

filename_time = 'models/time_on_device_pipeline_foliattiGeneral_v1.pkl'
model_time = joblib.load(filename_time)

filename_bet = 'models/bet_total_pipeline_foliattiGeneral_v0.pkl'
model_bet = joblib.load(filename_bet)

@app.route('/predict-cluster', methods=['POST'])
def assign_cluster():
    """
        Asignar cluster con base al comportamiento del usuario
        ---
        consumes:
          - application/json
        parameters:
          - in: body
            name: body
            description: Datos del usuario para predecir cluster
            required: true
            schema:
              type: object
              properties:
                AVG_BET:
                  type: number
                  example: 10.47
                INITIAL_AMOUNT:
                  type: number
                  example: 420
                GAMES_PLAYED_TOTAL:
                  type: integer
                  example: 109
                GAMES_WON_TOTAL:
                  type: integer
                  example: 435
                DOB:
                  type: string
                  format: date
                  example: "2001-08-12"
        responses:
          200:
            schema:
              type: object
              properties:
                prediction:
                  type: int
                  example: 0
    """
    try:
        name_clusters = {
            0: "Jugadores mayores, baja actividad y bajo riesgo",
            1: "Alta actividad, consistencia moderada",
            2: "Jovenes, apuestas mas altas pero poca actividad",
            3: "Jugadores intensivos, alto volumen y constancia"
        }

        features = ['AVG_BET','INITIAL_AMOUNT','GAMES_PLAYED_TOTAL','GAMES_WON_TOTAL','Rango_Edad_le']
        data = request.get_json()
        input_data = pd.DataFrame([data])

        edad, rango, rango_encoded = calcular_edad_y_rango_encoded(data['DOB'])
        input_data['Rango_Edad_le'] = rango_encoded

        prediction = model_cluster.predict(input_data[features])
        name_of_result = name_clusters.get(prediction[0], 4)
        return jsonify({'prediction': prediction.tolist(),
                        'label': name_of_result})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict-time', methods=['POST'])
def predict_time_on_device():
    """
        Predecir el tiempo de estancia en la maquina
        ---
        consumes:
          - application/json
        parameters:
          - in: body
            name: body
            description: Datos del usuario para predecir el tiempo que estara usando la maquina
            required: true
            schema:
              type: object
              properties:
                AVG_BET:
                  type: number
                  example: 10.47
                INITIAL_AMOUNT:
                  type: number
                  example: 420
                GAMES_PLAYED_TOTAL:
                  type: integer
                  example: 109
                GAMES_WON_TOTAL:
                  type: integer
                  example: 435
                DOB:
                  type: string
                  format: date
                  example: "2001-08-12"
        responses:
          200:
            schema:
              type: object
              properties:
                prediction:
                  type: int
                  example: 3
    """
    try:
        name_time = {
            0: "< 10 minutos", 
            1: ">= 10 minutos y < 30 minutos",
            2: ">= 30 minutos y < 60 minutos",
            3: "> 60 minutos"
        }

        features_cluster = ['AVG_BET','INITIAL_AMOUNT','GAMES_PLAYED_TOTAL','GAMES_WON_TOTAL','Rango_Edad_le']
        features = ['INITIAL_AMOUNT', 'Rango_Edad_le','Cluster', 
                    'AVG_BET', 'GAMES_PLAYED_TOTAL']
        
        data = request.get_json()
        input_data = pd.DataFrame([data])

        edad, rango, rango_encoded = calcular_edad_y_rango_encoded(data['DOB'])
        input_data['Rango_Edad_le'] = rango_encoded

        cluster = model_cluster.predict(input_data[features_cluster])
        input_data['Cluster'] = cluster[0]
        
        prediction = model_time.predict(input_data[features])
        name_of_result = name_time.get(prediction[0], 4)
        return jsonify({'prediction': prediction.tolist(),
                        'label': name_of_result})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict-bet', methods=['POST'])
def predict_bet_total():
    """
        Predecir la apuesta total del usuario
        ---
        consumes:
          - application/json
        parameters:
          - in: body
            name: body
            description: Datos del usuario para predecir la apuesta total del usuario
            required: true
            schema:
              type: object
              properties:
                AVG_BET:
                  type: number
                  example: 10.47
                INITIAL_AMOUNT:
                  type: number
                  example: 420
                GAMES_PLAYED_TOTAL:
                  type: integer
                  example: 109
                GAMES_WON_TOTAL:
                  type: integer
                  example: 435
                DOB:
                  type: string
                  format: date
                  example: "2001-08-12"
        responses:
          200:
            schema:
              type: object
              properties:
                prediction:
                  type: int
                  example: 2
    """
    try:
        name_bet = {
            0: "< 500",
            1: ">= 500 y <= 750",
            2: "> 750 y <= 1000",
            3: "> 1000 y < 5000",
            4: "> 5000"
        }
        features_cluster = ['AVG_BET','INITIAL_AMOUNT','GAMES_PLAYED_TOTAL','GAMES_WON_TOTAL','Rango_Edad_le']
        features = ['Cluster', 'INITIAL_AMOUNT', 'AVG_BET']
        
        data = request.get_json()
        input_data = pd.DataFrame([data])

        edad, rango, rango_encoded = calcular_edad_y_rango_encoded(data['DOB'])
        input_data['Rango_Edad_le'] = rango_encoded

        cluster = model_cluster.predict(input_data[features_cluster])
        input_data['Cluster'] = cluster[0]
        print(input_data)
        prediction = model_bet.predict(input_data[features])
        name_of_result = name_bet.get(prediction[0], 4)
        return jsonify({'prediction': prediction.tolist(), 
                        'label': name_of_result})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/ok', methods=['GET'])
def return_okey():
    return jsonify({'msg':'Ok'})

if __name__ == '__main__':
    app.run(debug=True)