from datetime import datetime
import numpy as np

def calcular_edad_y_rango_encoded(dob: str | datetime) -> tuple[int, str, int]:
    """
    Calcula la edad, rango de edad y el valor codificado a partir de la fecha de nacimiento (dob).
    Puede recibir un string en formato 'YYYY-MM-DD' o un datetime.
    Retorna (edad, rango, rango_encoded).
    """
    if isinstance(dob, str):
        dob = datetime.strptime(dob, "%Y-%m-%d")

    hoy = datetime.now()
    edad = hoy.year - dob.year
    # Ajustar si aún no ha cumplido años este año
    if (hoy.month, hoy.day) < (dob.month, dob.day):
        edad -= 1

    # Definir rangos
    bins = [18, 25, 35, 45, 55, 65, 200]  
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']

    rango = np.nan
    for i in range(len(bins) - 1):
        if bins[i] <= edad < bins[i+1]:
            rango = labels[i]
            break

    # Diccionario de codificación
    rango_edad_label = {
        '18-24': 0, '25-34': 1, '35-44': 2,
        '45-54': 3, '55-64': 4, '65+': 5, 
        np.nan: 6
    }

    rango_encoded = rango_edad_label.get(rango, 6)

    return edad, rango, rango_encoded
