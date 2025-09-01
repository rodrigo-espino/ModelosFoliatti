from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from flasgger import Swagger
import os

app = Flask(__name__)
swagger = Swagger(app)
# CORS(app)

class BusinessModelAPI:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self, model_dir="models"):
        """Cargar modelos y scalers entrenados"""
        try:
            # Cargar modelo TIEMPO
            self.models['tiempo'] = joblib.load(os.path.join(model_dir, "tiempo_model.pkl"))
            self.scalers['tiempo'] = joblib.load(os.path.join(model_dir, "tiempo_scaler.pkl"))
            print("✅ Modelo TIEMPO cargado")
            
            # Cargar modelo BET
            self.models['bet'] = joblib.load(os.path.join(model_dir, "bet_model.pkl"))
            self.scalers['bet'] = joblib.load(os.path.join(model_dir, "bet_scaler.pkl"))
            print("✅ Modelo BET cargado")
            
            # Cargar modelo WIN
            self.models['win'] = joblib.load(os.path.join(model_dir, "win_model.pkl"))
            self.scalers['win'] = joblib.load(os.path.join(model_dir, "win_scaler.pkl"))
            print("✅ Modelo WIN cargado")
            
            print("🎉 Todos los modelos cargados exitosamente!")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Error cargando modelos: {e}")
            return False
    
    def create_business_features(self, df, tiempo_pred=None, bet_pred=None, win_pred=None):
        """Crear features que reflejen la lógica real del negocio de casino"""
        features = df[['INITIAL_AMOUNT', 'AVG_BET', 'Cluster']].copy()
        
        if tiempo_pred is not None:
            features['tiempo_pred'] = tiempo_pred
            
        if bet_pred is not None and win_pred is not None:
            # Features que reflejan el comportamiento real del casino
            features['bet_pred'] = bet_pred
            features['win_pred'] = win_pred
            
            # FEATURES CLAVE PARA CASINO:
            
            # 1. Indicadores de comportamiento de juego
            features['total_money_handled'] = bet_pred  # Dinero total manejado
            features['house_edge_effect'] = bet_pred * 0.05  # Estimación de ventaja de la casa
            features['net_gaming_result'] = win_pred - bet_pred  # Resultado neto del juego
            
            # 2. Ratios de eficiencia y riesgo
            features['win_rate'] = win_pred / (bet_pred + 1)  # Tasa de ganancia
            features['money_multiplier'] = bet_pred / (df['INITIAL_AMOUNT'] + 1)  # Cuántas veces apostó su dinero inicial
            features['reinvestment_indicator'] = np.where(bet_pred > df['INITIAL_AMOUNT'], 1, 0)  # Si reinvirtió ganancias
            
            # 3. Patrones de gestión de dinero
            features['excess_betting'] = np.maximum(0, bet_pred - df['INITIAL_AMOUNT'])  # Apuestas con dinero ganado
            features['potential_redemptions'] = win_pred * 0.7  # Estimación de dinero que podría haber retirado
            features['money_at_risk'] = np.minimum(bet_pred, df['INITIAL_AMOUNT'] + win_pred)
            
            # 4. Indicadores de comportamiento de salida
            features['likely_loss_scenario'] = np.where(win_pred < bet_pred * 0.5, 1, 0)
            features['likely_win_scenario'] = np.where(win_pred > bet_pred * 1.2, 1, 0)
            features['breakeven_scenario'] = np.where(
                (win_pred >= bet_pred * 0.8) & (win_pred <= bet_pred * 1.2), 1, 0
            )
            
            # 5. Estimaciones de flujo de efectivo durante la sesión
            available_money_estimate = df['INITIAL_AMOUNT'] + win_pred * 0.6  # Asumiendo que retira 40% de ganancias
            features['estimated_available_money'] = available_money_estimate
            features['final_money_simple_estimate'] = available_money_estimate - bet_pred + win_pred * 0.4
            
            # 6. Features específicos por cluster (comportamiento por tipo de jugador)
            features['cluster_risk_adjusted'] = df['Cluster'] * features['money_multiplier']
            features['cluster_win_pattern'] = df['Cluster'] * features['win_rate']
            
        return features
    
    def predict_session(self, initial_amount, avg_bet, cluster, weekday=1, weekend=0, month=1):
        """Hacer predicción completa para una sesión usando los modelos en secuencia"""
        if not self.models:
            return {"error": "Modelos no cargados correctamente"}
        
        # Preparar datos base
        base_data = pd.DataFrame({
            'INITIAL_AMOUNT': [initial_amount],
            'AVG_BET': [avg_bet], 
            'Cluster': [cluster],
            'Weekday': [weekday],
            'Weekend': [weekend],
            'Month': [month]
        })
        
        try:
            # 1. Predecir TIEMPO
            X_tiempo = self.create_business_features(base_data)
            X_tiempo_scaled = self.scalers['tiempo'].transform(X_tiempo)
            tiempo_pred = float(self.models['tiempo'].predict(X_tiempo_scaled)[0])
            
            # 2. Predecir BET TOTAL
            X_bet = self.create_business_features(base_data, tiempo_pred=tiempo_pred)
            X_bet_scaled = self.scalers['bet'].transform(X_bet)
            bet_pred = float(self.models['bet'].predict(X_bet_scaled)[0])
            
            # 3. Predecir WIN TOTAL
            X_win = self.create_business_features(base_data, tiempo_pred=tiempo_pred, bet_pred=bet_pred)
            # Remover features de win para evitar data leakage
            X_win = X_win.drop(['win_pred', 'net_gaming_result', 'win_rate', 
                               'excess_betting', 'potential_redemptions',
                               'likely_loss_scenario', 'likely_win_scenario',
                               'breakeven_scenario', 'estimated_available_money',
                               'final_money_simple_estimate', 'cluster_win_pattern'], axis=1, errors='ignore')
            
            X_win_scaled = self.scalers['win'].transform(X_win)
            win_pred = float(self.models['win'].predict(X_win_scaled)[0])
            
            return {
                'tiempo_minutos': round(tiempo_pred, 2),
                'bet_total': round(bet_pred, 2),
                'win_total': round(win_pred, 2),
                'status': 'success'
            }
            
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
        # edad, rango, rango_encoded = calcular_edad_y_rango_encoded(data['DOB'])
        # input_data['Rango_Edad_le'] = rango_encoded

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
    Prediccion de Tiempo, Bet Total y Win Total con lógica de negocio
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
            tiempo_minutos:
              type: number
              example: 45.2
            bet_total:
              type: number
              example: 1250.5
            win_total:
              type: number
              example: 1100.3
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
        
        # Hacer predicción usando el modelo de negocio
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
        
        # Formatear respuesta
        response = {
            'Time on Device Prediction': f"{result['tiempo_minutos']} minutos",
            'Bet Total Prediction': f"${result['bet_total']}",
            'Win Total Prediction': f"${result['win_total']}",
            'Resultado Neto': f"${result['resultado_neto']}",
            'ROI': f"{result['roi_porcentaje']}%",
            'Comportamiento': result['comportamiento'],
            'raw_data': result  # Datos sin formatear para uso programático
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    """Verificar estado de la API"""
    model_status = {
        'tiempo': 'tiempo' in business_model.models,
        'bet': 'bet' in business_model.models,
        'win': 'win' in business_model.models,
        'cluster': model_cluster is not None
    }
    
    return jsonify({
        'status': 'API is running',
        'message': 'Flask ML API con lógica de negocio funcionando correctamente',
        'models_loaded': model_status,
        'endpoints': [
            '/predict-cluster',
            '/predict-time-bet-win',
            '/predict-complete-session',
            '/ok'
        ]
    })

@app.route('/ok', methods=['GET'])
def return_ok():
    """Endpoint simple de verificación"""
    return jsonify({'msg': 'Ok', 'models_ready': len(business_model.models) > 0})

@app.route('/model-info', methods=['GET'])
def model_info():
    """Información sobre los modelos cargados"""
    info = {
        'models_loaded': list(business_model.models.keys()),
        'scalers_loaded': list(business_model.scalers.keys()),
        'business_logic': 'Enabled',
        'version': '2.0 - Business Logic Enhanced'
    }
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)