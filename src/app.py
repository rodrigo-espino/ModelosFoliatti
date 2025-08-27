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
      Asignacion de cluster
      ---
      tags:
        - Predicciones
      consumes:
        - application/json
      produces:
        - application/json
      parameters:
        - in: body
          name: body
          required: true
          description: Datos del jugador para realizar las predicciones
          schema:
            type: object
            properties:
              Cluster:
                type: number                
                example: 1
              INITIAL_AMOUNT:
                type: number
                example: 500
              AVG_BET:
                type: number
                example: 25.5
              GAMES_PLAYED_TOTAL:
                type: integer
                example: 120
      responses:
        200:
          description: Predicciones exitosas
          schema:
            type: object
            properties:
              cluster:
                type: number
                example: 2
              interpretation:
                type: string
                example: "Jovenes, apuestas mas altas pero poca actividad"
              
        400:
          description: Error en la entrada de datos
        500:
          description: Error interno del servidor
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
        return jsonify({'cluster': prediction.toList(),
                        'interpretation': name_of_result})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict-time-bet', methods=['POST'])
def predict_time_bet():
    """
      Predicción de tiempo en dispositivo y monto de apuesta
      ---
      tags:
        - Predicciones
      consumes:
        - application/json
      produces:
        - application/json
      parameters:
        - in: body
          name: body
          required: true
          description: Datos del jugador para realizar las predicciones
          schema:
            type: object
            properties:
              Cluster:
                type: number                
                example: 1
              INITIAL_AMOUNT:
                type: number
                example: 500
              AVG_BET:
                type: number
                example: 25.5
              GAMES_PLAYED_TOTAL:
                type: integer
                example: 120
      responses:
        200:
          description: Predicciones exitosas
          schema:
            type: object
            properties:
              time:
                type: object
                properties:
                  prediction_time:
                    type: integer
                    example: 2
                  interpretation_time:
                    type: string
                    example: "7 - 16 minutos"
              bet_total:
                type: object
                properties:
                  prediction_betTotal:
                    type: integer
                    example: 1
                  interpretation_betTotal:
                    type: string
                    example: "$117 - $319"
        400:
          description: Error en la entrada de datos
        500:
          description: Error interno del servidor
    """
    try:
        name_time = {
            0: "0 - 2 minutos", 
            1: "3 - 6 minutos",
            2: "7 - 16 miniutos",
            3: "> 16 minutos"
        }
        name_bet = {
            0: "$0 - $116", 
            1: "$117 - $319",
            2: "$320 - $786",
            3: "> $786"
        }
        features_time = ['INITIAL_AMOUNT','Cluster', 
                        'AVG_BET', 'GAMES_PLAYED_TOTAL']
        
        features_bet = ['Cluster', 'INITIAL_AMOUNT', 
                        'AVG_BET', 'time_on_device_label']

        data = request.get_json()
        input_data = pd.DataFrame([data])     

        prediction_time = model_time.predict(input_data[features_time])
        result_time = name_time.get(prediction_time[0], 4)
        input_data['time_on_device_label'] = prediction_time[0]

        prediction_bet = model_bet.predict(input_data[features_bet])
        result_bet = name_bet.get(prediction_bet[0], 4)

        return jsonify({'time': {'prediction_time': prediction_time.toList(),
                        'interpretation_time': result_time},
                'bet_total': {'prediction_betTotal': prediction_bet.toList(),
                             'interpretation_betTotal': result_bet}})
    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/predict-cluster-time-bet', methods=['POST'])
def predict_time_bet_with_cluster():
    """
      Predicción de cluster, tiempo en dispositivo y monto de apuesta
      ---
      tags:
        - Predicciones
      consumes:
        - application/json
      produces:
        - application/json
      parameters:
        - in: body
          name: body
          required: true
          description: Datos del jugador para realizar las predicciones
          schema:
            type: object
            properties:
              DOB:
                type: string
                format: date
                example: "1985-07-12"
              INITIAL_AMOUNT:
                type: number
                example: 500
              AVG_BET:
                type: number
                example: 25.5
              GAMES_PLAYED_TOTAL:
                type: integer
                example: 120
              GAMES_WON_TOTAL:
                type: integer
                example: 40
      responses:
        200:
          description: Predicciones exitosas
          schema:
            type: object
            properties:
              time:
                type: object
                properties:
                  prediction_time:
                    type: integer
                    example: 2
                  interpretation_time:
                    type: string
                    example: "7 - 16 minutos"
              bet_total:
                type: object
                properties:
                  prediction_betTotal:
                    type: integer
                    example: 1
                  interpretation_betTotal:
                    type: string
                    example: "$117 - $319"
              cluster:
                type: object
                properties:
                  prediction_cluster:
                    type: integer
                    example: 0
                  interpretation_cluster:
                    type: string
                    example: "Jugadores mayores, baja actividad y bajo riesgo"
        400:
          description: Error en la entrada de datos
        500:
          description: Error interno del servidor
    """
    try:
        name_clusters = {
            0: "Jugadores mayores, baja actividad y bajo riesgo",
            1: "Alta actividad, consistencia moderada",
            2: "Jovenes, apuestas mas altas pero poca actividad",
            3: "Jugadores intensivos, alto volumen y constancia"
        }
        name_time = {
            0: "0 - 2 minutos", 
            1: "3 - 6 minutos",
            2: "7 - 16 miniutos",
            3: "> 16 minutos"
        }
        name_bet = {
            0: "$0 - $116", 
            1: "$117 - $319",
            2: "$320 - $786",
            3: "> $786"
        }
        features_time = ['INITIAL_AMOUNT','Cluster', 
                        'AVG_BET', 'GAMES_PLAYED_TOTAL']
        
        features_bet = ['Cluster', 'INITIAL_AMOUNT', 
                        'AVG_BET', 'time_on_device_label']

        features_cluster = ['AVG_BET','INITIAL_AMOUNT','GAMES_PLAYED_TOTAL',
                    'GAMES_WON_TOTAL','Rango_Edad_le']
        data = request.get_json()
        input_data = pd.DataFrame([data])     

        edad, rango, rango_encoded = calcular_edad_y_rango_encoded(data['DOB'])
        input_data['Rango_Edad_le'] = rango_encoded

        prediction_cluster = model_cluster.predict(input_data[features_cluster])
        result_cluster = name_clusters.get(prediction_cluster[0], 4)
        input_data['Cluster'] = prediction_cluster[0]

        prediction_time = model_time.predict(input_data[features_time])
        result_time = name_time.get(prediction_time[0], 4)
        input_data['time_on_device_label'] = prediction_time[0]

        prediction_bet = model_bet.predict(input_data[features_bet])
        result_bet = name_bet.get(prediction_bet[0], 4)

        return jsonify({'time': {'prediction_time': prediction_time.tolist(),
                        'interpretation_time': result_time},
                'bet_total': {'prediction_betTotal': prediction_bet.tolist(),
                             'interpretation_betTotal': result_bet},
                'cluster': {'prediction_cluster': prediction_cluster.tolist(),
                           'interpretation_cluster': result_cluster}})
    except Exception as e:
        return jsonify({'error': str(e)})

# Ruta de prueba para verificar que la API está funcionando
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'API is running',
        'message': 'Flask ML API is working correctly',
        'endpoints': [
            '/predict-cluster',
            '/predict-time-bet',
            '/predict-cluster-time-bet'
        ]
    })

@app.route('/ok', methods=['GET'])
def return_okey():
    return jsonify({'msg':'Ok'})

if __name__ == '__main__':
    app.run(debug=True)