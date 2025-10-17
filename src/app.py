from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from flasgger import Swagger
from func import calcular_edad_y_rango_encoded
import os
import tensorflow as tf
from tensorflow import keras
import warnings

app = Flask(__name__)
swagger = Swagger(app)
# CORS(app)

class BusinessModelAPI:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.pca_models = {}
        self.load_models()
    
    def load_models(self, model_dir="models"):
        """Cargar modelos TensorFlow/Keras y scalers entrenados"""
        try:
            # Configurar TensorFlow para evitar warnings
            tf.get_logger().setLevel('ERROR')
            warnings.filterwarnings('ignore', category=UserWarning)
            
            # Cargar modelo TIEMPO
            tiempo_model_path = os.path.join(model_dir, "tiempo_model.h5")
            tiempo_scaler_path = os.path.join(model_dir, "tiempo_scaler.pkl")
            
            if os.path.exists(tiempo_model_path) and os.path.exists(tiempo_scaler_path):
                try:
                    # Cargar con custom_objects para manejar métricas personalizadas
                    self.models['tiempo'] = keras.models.load_model(
                        tiempo_model_path,
                        custom_objects={
                            'mse': tf.keras.metrics.MeanSquaredError(),
                            'mae': tf.keras.metrics.MeanAbsoluteError(),
                            'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                            'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError()
                        },
                        compile=False  # No compilar el modelo al cargar
                    )
                    # Recompilar el modelo con métricas actualizadas
                    self.models['tiempo'].compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=['mae']
                    )
                    self.scalers['tiempo'] = joblib.load(tiempo_scaler_path)
                    print("✅ Modelo TIEMPO cargado")
                except Exception as e:
                    print(f"❌ Error cargando modelo TIEMPO: {e}")
            else:
                print("❌ Archivos del modelo TIEMPO no encontrados")
            
            # Cargar modelo BET
            bet_model_path = os.path.join(model_dir, "bet_model.h5")
            bet_scaler_path = os.path.join(model_dir, "bet_scaler.pkl")
            
            if os.path.exists(bet_model_path) and os.path.exists(bet_scaler_path):
                try:
                    self.models['bet'] = keras.models.load_model(
                        bet_model_path,
                        custom_objects={
                            'mse': tf.keras.metrics.MeanSquaredError(),
                            'mae': tf.keras.metrics.MeanAbsoluteError(),
                            'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                            'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError()
                        },
                        compile=False
                    )
                    self.models['bet'].compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=['mae']
                    )
                    self.scalers['bet'] = joblib.load(bet_scaler_path)
                    print("✅ Modelo BET cargado")
                except Exception as e:
                    print(f"❌ Error cargando modelo BET: {e}")
            else:
                print("❌ Archivos del modelo BET no encontrados")
            
            # Cargar modelo WIN
            win_model_path = os.path.join(model_dir, "win_model.h5")
            win_scaler_path = os.path.join(model_dir, "win_scaler.pkl")
            
            if os.path.exists(win_model_path) and os.path.exists(win_scaler_path):
                try:
                    self.models['win'] = keras.models.load_model(
                        win_model_path,
                        custom_objects={
                            'mse': tf.keras.metrics.MeanSquaredError(),
                            'mae': tf.keras.metrics.MeanAbsoluteError(),
                            'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                            'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError()
                        },
                        compile=False
                    )
                    self.models['win'].compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=['mae']
                    )
                    self.scalers['win'] = joblib.load(win_scaler_path)
                    print("✅ Modelo WIN cargado")
                except Exception as e:
                    print(f"❌ Error cargando modelo WIN: {e}")
            else:
                print("❌ Archivos del modelo WIN no encontrados")

            # Cargar modelo Final Amount
            final_amount_model_path = os.path.join(model_dir, "final_amount_model_rf.pkl")
            final_amount_scaler_path = os.path.join(model_dir, "final_amount_scaler.pkl")

            if os.path.exists(final_amount_model_path) and os.path.exists(final_amount_scaler_path): 
                try:
                    self.models['final_amount'] = joblib.load(final_amount_model_path)

                    self.scalers['final_amount'] = joblib.load(final_amount_scaler_path)
                    print("✅ Modelo Final Amount cargado")
                except Exception as e:
                    print(f"❌ Error cargando modelo Final Amount: {e}")
            else:
                print("❌ Archivos del modelo Final Amount no encontrados")
            
            # Cargar modelo Games Played
            games_played_model_path = os.path.join(model_dir, "games_played_model_mlp.pkl")
            games_played_scaler_path = os.path.join(model_dir, "games_played_scaler.pkl")
            games_played_targetScaler_path = os.path.join(model_dir, "target_games_played_scaler.pkl")

            if os.path.exists(games_played_model_path) and os.path.exists(games_played_scaler_path):
                try:
                    self.models['games_played'] = joblib.load(games_played_model_path)
                    self.scalers['games_played'] = joblib.load(games_played_scaler_path)
                    self.scalers['games_played_target'] = joblib.load(games_played_targetScaler_path)

                    print("✅ Modelo Games Played cargado")
                except Exception as e:
                    print(f"❌ Error cargando modelo Games Played: {e}")
            else:
                print("❌ Archivos del modelo Games Played no encontrados")

            # Cargar modelos PCA (opcionales)
            for model_name in ['tiempo', 'bet', 'win']:
                pca_path = os.path.join(model_dir, f"{model_name}_pca.pkl")
                if os.path.exists(pca_path):
                    try:
                        self.pca_models[model_name] = joblib.load(pca_path)
                        print(f"✅ PCA {model_name.upper()} cargado")
                    except Exception as e:
                        print(f"⚠️ Error cargando PCA {model_name}: {e}")
            
            if len(self.models) > 0:
                print(f"🎉 {len(self.models)} modelos cargados exitosamente!")
                return True
            else:
                print("❌ No se pudieron cargar los modelos")
                return False
            
        except Exception as e:
            print(f"❌ Error general cargando modelos: {e}")
            return False
    
    def create_business_features(self, df, tiempo_pred=None, bet_pred=None, win_pred = None, final_pred= None, games_played_pred= None):
        """Crear features que reflejen la lógica real del negocio de casino"""
        # Features base (deben coincidir con el entrenamiento)
        features = df[['INITIAL_AMOUNT', 'AVG_BET', 'Cluster']].copy()
        
        if tiempo_pred is not None:
            features['tiempo_pred'] = tiempo_pred

        if win_pred is not None:
            features['win_pred'] = win_pred
        
        if final_pred is not None: 
            features['final_pred'] = final_pred    
        if bet_pred is not None:
            # Features de negocio (deben coincidir exactamente con el entrenamiento)
            features['bet_pred'] = bet_pred
            features['total_money_handled'] = bet_pred
            features['house_edge_effect'] = bet_pred * 0.05
            features['money_multiplier'] = bet_pred / (df['INITIAL_AMOUNT'] + 1)
            features['reinvestment_indicator'] = np.where(bet_pred > df['INITIAL_AMOUNT'], 1, 0)
            features['excess_betting'] = np.maximum(0, bet_pred - df['INITIAL_AMOUNT'])
            features['money_at_risk'] = np.minimum(bet_pred, df['INITIAL_AMOUNT'])
            features['cluster_risk_adjusted'] = df['Cluster'] * features['money_multiplier']
            
        return features
    
    def session_features(self, df, tiempo_pred=None, bet_pred=None, win_pred = None, final_pred= None, games_played_pred= None):
        features = df[['INITIAL_AMOUNT', 'AVG_BET', 'Cluster']].copy()

        features = features.rename(columns={'INITIAL_AMOUNT': 'initial_amount', 
                                 'AVG_BET': 'avg_bet',
                                 'Cluster': 'cluster'})
        
        if tiempo_pred is not None:
            features['time_on_device'] = tiempo_pred

        if win_pred is not None:
            features['win_total'] = win_pred
        
        if final_pred is not None: 
            features['final_pred'] = final_pred    
        if bet_pred is not None:
            # Features de negocio (deben coincidir exactamente con el entrenamiento)
            features['bet_total'] = bet_pred
        features = features[['time_on_device', 'bet_total', 'win_total', 
                             'initial_amount', 'cluster', 'avg_bet']]
        return features

    def predict_session(self, initial_amount, avg_bet, cluster, weekday=1, weekend=0, month=1):
        """Hacer predicción completa para una sesión usando los modelos MLP en secuencia"""
        if not self.models:
            return {"error": "Modelos no cargados correctamente"}
        
        # Preparar datos base (debe coincidir con las columnas del entrenamiento)
        base_data = pd.DataFrame({
            'INITIAL_AMOUNT': [initial_amount],
            'AVG_BET': [avg_bet], 
            'Cluster': [cluster],
            'Weekday': [weekday],
            'Weekend': [weekend],
            'Month': [month]
        })
        
        try:
            results = {}
            
            # 1. Predecir TIEMPO (si el modelo está disponible)
            if 'tiempo' in self.models:
                try:
                    X_tiempo = self.create_business_features(base_data)
                    X_tiempo_scaled = self.scalers['tiempo'].transform(X_tiempo)
                    tiempo_pred = float(self.models['tiempo'].predict(X_tiempo_scaled, verbose=0)[0][0])
                    res_tempo = round(max(0, tiempo_pred), 2)  # Asegurar valor positivo
                    mae_tiempo = 8.64
                    min_value = max(0, res_tempo - mae_tiempo, res_tempo)
                    max_value = res_tempo + mae_tiempo
                    results['tiempo_minutos'] = f"{min_value:.2f} - {max_value:.2f} minutos"

                except Exception as e:
                    print(f"Error prediciendo tiempo: {e}")
                    tiempo_pred = 30.0
                    results['tiempo_minutos'] = tiempo_pred
            else:
                tiempo_pred = 30.0  # Valor por defecto si no hay modelo
                results['tiempo_minutos'] = tiempo_pred
            
            # 2. Predecir BET TOTAL (si el modelo está disponible)
            if 'bet' in self.models:
                try:
                    X_bet = self.create_business_features(base_data, tiempo_pred=tiempo_pred)
                    X_bet_scaled = self.scalers['bet'].transform(X_bet)
                    bet_pred = float(self.models['bet'].predict(X_bet_scaled, verbose=0)[0][0])
                    res_bet = round(max(0, bet_pred), 2)  # Asegurar valor positivo
                    
                    mae_bet = 454
                    min_value = max(0, res_bet - mae_bet, res_bet)
                    max_value = res_bet + mae_bet
                    
                    results['bet_total'] = f"${min_value:.2f} - ${max_value:.2f}"
                except Exception as e:
                    print(f"Error prediciendo bet: {e}")
                    bet_pred = initial_amount * 2
                    results['bet_total'] = bet_pred
            else:
                bet_pred = initial_amount * 2  # Valor por defecto si no hay modelo
                results['bet_total'] = bet_pred
            
            # 3. Predecir WIN TOTAL (si el modelo está disponible)
            if 'win' in self.models:
                try:
                    X_win = self.create_business_features(base_data, tiempo_pred=tiempo_pred, bet_pred=bet_pred)
                    X_win_scaled = self.scalers['win'].transform(X_win)
                    win_pred = float(self.models['win'].predict(X_win_scaled, verbose=0)[0][0])
                    res_win = round(max(0, win_pred), 2)  # Asegurar valor positivo
                    mae_win = 511
                    min_value = max(0, res_win - mae_win, res_win)
                    max_value = res_win + mae_win

                    results['win_total'] = f"${min_value:.2f} - ${max_value:.2f}"

                    # Calcular el rango
                    
                except Exception as e:
                    print(f"Error prediciendo win: {e}")
                    win_pred = bet_pred * 0.95
                    results['win_total'] = win_pred
            else:
                win_pred = bet_pred * 0.95  # Valor por defecto si no hay modelo (house edge)
                results['win_total'] = win_pred

            if 'final_amount' in self.models: 
                try:
                    X_final = self.session_features(base_data, tiempo_pred=tiempo_pred, bet_pred=bet_pred, win_pred=win_pred)
                    X_final_scaled = self.scalers['final_amount'].transform(X_final)

                    final_pred = self.models['final_amount'].predict(X_final_scaled)
                    results['ganancia'] = 'Casino' if final_pred else 'Jugador'
                    
                except Exception as e:
                    print(f"Error prediciendo final amount: {e}")
                    final_pred = 'Casino'
                    results['ganancia'] = final_pred
                else:
                    final_pred = 'Casino'
                    results['ganancia'] = final_pred
            
            if 'games_played' in self.models:
                try:
                    X_games_played = self.session_features(base_data, tiempo_pred=tiempo_pred, bet_pred=bet_pred, win_pred=win_pred)
                    X_games_played_scaled = self.scalers['games_played'].transform(X_games_played)

                    games_played_pred = self.models['games_played'].predict(X_games_played_scaled)
                    
                    # CORRECCIÓN: Asegurar que sea 2D para el scaler
                    if games_played_pred.ndim == 1:
                        games_played_pred = games_played_pred.reshape(-1, 1)
                    
                    games_played_pred_scaled = self.scalers['games_played_target'].transform(games_played_pred)
                    
                    # Si necesitas el resultado como escalar
                    games_played_pred_value = games_played_pred_scaled.flatten()[0] if games_played_pred_scaled.size == 1 else games_played_pred_scaled

                    mae_games_played = 102
        
                    # Calcular el rango
                    min_value = max(0, games_played_pred_value - mae_games_played, games_played_pred_value)
                    max_value = games_played_pred_value + mae_games_played
                    print(f"MAE: {mae_games_played}. MIN: {min_value}. MAX: {max_value}")
                    # Formatear el resultado como rango
                    results['games_played_pred'] = f"{min_value:.1f} - {max_value:.1f}"

                except Exception as e:
                    print(f"Error prediciendo games played: {e}")
                    games_played_pred = 0
                    results['games_played_pred'] = games_played_pred
            else: 
                games_played_pred = 0
                results['games_played_pred'] = games_played_pred
            return results
            
        except Exception as e:
            return {"error": f"Error en predicción: {str(e)}"}

# Instanciar el modelo de negocio
business_model = BusinessModelAPI()

# Load additional models (mantener compatibilidad con tu estructura actual)
try:
    filename_cluster = 'models/knn_pipeline_foliattiGeneral_v0.pkl'
    model_cluster = joblib.load(filename_cluster)
    print("✅ Modelo de cluster cargado")
except:
    model_cluster = None
    print("⚠️ Modelo de cluster no encontrado")

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
            GAMES_WON_TOTAL:
              type: number
              example: 140
            DOB:
              type: string
              example: "1990-05-15"
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
    if not model_cluster:
        return jsonify({'error': 'Modelo de cluster no disponible'}), 500
        
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

        # Necesitarás implementar esta función o adaptarla
        edad, rango, rango_encoded = calcular_edad_y_rango_encoded(data['DOB'])
        input_data['Rango_Edad_le'] = rango_encoded

        prediction = model_cluster.predict(input_data[features])
        name_of_result = name_clusters.get(prediction[0], 4)
        return jsonify({
            'cluster': prediction.tolist(),
            'interpretation': name_of_result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-time-bet-win', methods=['POST'])
def predict_time_bet_win():
    """
    Prediccion de Tiempo, Bet Total y Win Total con modelos MLP
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
            INITIAL_AMOUNT:
              type: number
              example: 500
            AVG_BET:
              type: number
              example: 25.5
            Cluster:
              type: integer
              example: 1
            INITIAL_TIME:
              type: string
              example: "2024-01-15 14:30:00"
    responses:
      200:
        description: Predicciones exitosas
        schema:
          type: object
          properties:
            bet_total:
              type: string
              example: "$1250.5 - $1650"
            win_total:
              type: string
              example: "$1100.3 - $1242.3"
            tiempo_minutos:
              type: string
              example: "55.97 - 73.25 minutos"        
      400:
        description: Error en la entrada de datos
      500:
        description: Error interno del servidor
    """
    try:
        data = request.get_json()
        
        # Validar datos requeridos
        required_fields = ['INITIAL_AMOUNT', 'AVG_BET', 'Cluster', 'INITIAL_TIME']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo requerido: {field}'}), 400
        
        # Procesar fecha y tiempo
        input_data = pd.DataFrame([data])
        input_data['INITIAL_TIME'] = pd.to_datetime(input_data['INITIAL_TIME'])
        
        weekday = input_data['INITIAL_TIME'].dt.weekday.iloc[0]  # 0=Lunes, 6=Domingo
        weekend = 1 if weekday >= 5 else 0
        month = input_data['INITIAL_TIME'].dt.month.iloc[0]
        
        # Hacer predicción usando el modelo de negocio MLP
        result = business_model.predict_session(
            initial_amount=float(data['INITIAL_AMOUNT']),
            avg_bet=float(data['AVG_BET']),
            cluster=int(data['Cluster']),
            weekday=weekday,
            weekend=weekend,
            month=month
        )
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-cluster-time-bet-win', methods=['POST'])
def predict_cluster_time_bet_win():
    """
    Prediccion completa: Cluster + Tiempo + Bet Total + Win Total
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
        description: Datos del jugador para realizar todas las predicciones
        schema:
          type: object
          properties:
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
              type: number
              example: 140
            DOB:
              type: string
              example: "1990-05-15"
            INITIAL_TIME:
              type: string
              example: "2024-01-15 14:30:00"
    responses:
      200:
        description: Predicciones completas exitosas
        schema:
          type: object
          properties:
            cluster:
              type: object
              properties:
                cluster_number:
                  type: integer
                  example: 2
                cluster_interpretation:
                  type: string
                  example: "Jovenes, apuestas mas altas pero poca actividad"
            predictions:
              type: object
              properties:
                bet_total:
                  type: string
                  example: "$1250.5 - $1650"
                win_total:
                  type: string
                  example: "$1100.3 - $1242.3"
                tiempo_minutos:
                  type: string
                  example: "55.97 - 73.25 minutos"
                ganancia:
                  type: string
                  example: "Casino"
                games_played_pred:
                  type: string
                  example: "45.2 - 147.2"
      400:
        description: Error en la entrada de datos
      500:
        description: Error interno del servidor
    """
    if not model_cluster:
        return jsonify({'error': 'Modelo de cluster no disponible'}), 500
        
    try:
        data = request.get_json()
        
        # Validar datos requeridos para cluster
        required_cluster_fields = ['INITIAL_AMOUNT', 'AVG_BET', 'GAMES_PLAYED_TOTAL', 'GAMES_WON_TOTAL', 'DOB']
        for field in required_cluster_fields:
            if field not in data:
                return jsonify({'error': f'Campo requerido para cluster: {field}'}), 400
        
        # Validar datos requeridos para predicciones MLP
        if 'INITIAL_TIME' not in data:
            return jsonify({'error': 'Campo requerido para predicciones MLP: INITIAL_TIME'}), 400
        
        # PASO 1: Predecir cluster
        name_clusters = {
            0: "Jugadores mayores, baja actividad y bajo riesgo",
            1: "Alta actividad, consistencia moderada",
            2: "Jovenes, apuestas mas altas pero poca actividad",
            3: "Jugadores intensivos, alto volumen y constancia"
        }
        
        features = ['AVG_BET','INITIAL_AMOUNT','GAMES_PLAYED_TOTAL','GAMES_WON_TOTAL','Rango_Edad_le']
        input_data_cluster = pd.DataFrame([data])
        
        # Calcular edad y rango encoded
        edad, rango, rango_encoded = calcular_edad_y_rango_encoded(data['DOB'])
        input_data_cluster['Rango_Edad_le'] = rango_encoded
        
        # Predecir cluster
        cluster_prediction = model_cluster.predict(input_data_cluster[features])
        cluster_number = int(cluster_prediction[0])  # Convertir a int nativo de Python
        cluster_interpretation = name_clusters.get(cluster_number, "Cluster desconocido")
        
        # PASO 2: Usar el cluster predicho para las predicciones MLP
        # Procesar fecha y tiempo
        input_data_mlp = pd.DataFrame([data])
        input_data_mlp['INITIAL_TIME'] = pd.to_datetime(input_data_mlp['INITIAL_TIME'])
        
        weekday = input_data_mlp['INITIAL_TIME'].dt.weekday.iloc[0]  # 0=Lunes, 6=Domingo
        weekend = 1 if weekday >= 5 else 0
        month = input_data_mlp['INITIAL_TIME'].dt.month.iloc[0]
        
        # Hacer predicción usando el modelo de negocio MLP con el cluster predicho
        mlp_result = business_model.predict_session(
            initial_amount=float(data['INITIAL_AMOUNT']),
            avg_bet=float(data['AVG_BET']),
            cluster=cluster_number,  # Usar el cluster predicho
            weekday=weekday,
            weekend=weekend,
            month=month
        )
        
        if 'error' in mlp_result:
            return jsonify({
                'error': f'Error en predicciones MLP: {mlp_result["error"]}',
                'cluster': {
                    'cluster_number': cluster_number,
                    'cluster_interpretation': cluster_interpretation
                }
            }), 500
        
        # PASO 3: Combinar resultados
        combined_result = {
            'cluster': {
                'cluster_number': int(cluster_number),  # Asegurar conversión a int nativo
                'cluster_interpretation': cluster_interpretation,
                'edad_calculada': int(edad) if edad is not None else None,
                'rango_edad': str(rango) if rango is not None else None
            },
            'predictions': mlp_result,
        }
        
        return jsonify(combined_result)
        
    except Exception as e:
        return jsonify({'error': f'Error en predicción combinada: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def health_check():
    """Verificar estado de la API"""
    return render_template('index.html')

@app.route('/predict-general', methods=['GET'])
def predict_general_information():
    return render_template('predict-general.html')

@app.route('/ok', methods=['GET'])
def return_ok():
    """Endpoint simple de verificación"""
    return jsonify({
        'msg': 'Ok', 
        'models_ready': len(business_model.models) > 0,
        'models_count': len(business_model.models)
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Información detallada sobre los modelos cargados"""
    info = {
        'mlp_models': {
            'loaded': list(business_model.models.keys()),
            'scalers': list(business_model.scalers.keys()),
            'pca_models': list(business_model.pca_models.keys()),
            'type': 'TensorFlow/Keras MLP',
            'format': '.h5 files',
            'features_tiempo': ['INITIAL_AMOUNT', 'AVG_BET', 'Cluster'],
            'features_bet': ['INITIAL_AMOUNT', 'AVG_BET', 'Cluster', 'tiempo_pred'],
            'features_win': ['INITIAL_AMOUNT', 'AVG_BET', 'Cluster', 'tiempo_pred', 'bet_pred', '+ business_features']
        },
        'cluster_model': {
            'loaded': model_cluster is not None,
            'type': 'KNN Pipeline (Scikit-learn)',
            'format': '.pkl file',
            'file': 'knn_pipeline_foliattiGeneral_v0.pkl',
            'features': ['AVG_BET', 'INITIAL_AMOUNT', 'GAMES_PLAYED_TOTAL', 'GAMES_WON_TOTAL', 'Rango_Edad_le'],
            'clusters': {
                0: "Jugadores mayores, baja actividad y bajo riesgo",
                1: "Alta actividad, consistencia moderada",
                2: "Jovenes, apuestas mas altas pero poca actividad",
                3: "Jugadores intensivos, alto volumen y constancia"
            }
        },
        'system_info': {
            'tensorflow_version': tf.__version__,
            'version': '3.0 - Hybrid: MLP TensorFlow + KNN Cluster',
            'business_logic': 'Enabled'
        }
    }
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)