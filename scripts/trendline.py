import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

def fit_trendlines(x, y):
    """
    Create a trendline based on a timeseries
    Args:
        x: np.arrange(df[column])
        y: variable y

    Returns:
        best_fit_type: Type of trendline
        best_fit_data: Data to plot trendline

    Example:
    best_fit_type, best_fit_data = fit_trendlines(np.arange(len(df_deposits_paid['DateTime'])), df_deposits_paid['Amount'])
    """

    # Convertir fechas a números para los cálculos
    x_num = np.arange(len(x))

    # Función para ajuste exponencial
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    # Calcular diferentes ajustes
    fits = {}

    # Lineal
    z = np.polyfit(x_num, y, 1)
    p = np.poly1d(z)
    fits['Linear'] = {
        'y_pred': p(x_num),
        'r2': r2_score(y, p(x_num)),
        'func': lambda x: p(x)
    }

    # Polinomial (grado 2)
    z = np.polyfit(x_num, y, 2)
    p = np.poly1d(z)
    fits['Polynomial'] = {
        'y_pred': p(x_num),
        'r2': r2_score(y, p(x_num)),
        'func': lambda x: p(x)
    }

    # Logarítmico
    try:
        z = np.polyfit(np.log(x_num + 1), y, 1)
        fits['Logarithmic'] = {
            'y_pred': z[0] * np.log(x_num + 1) + z[1],
            'r2': r2_score(y, z[0] * np.log(x_num + 1) + z[1]),
            'func': lambda x: z[0] * np.log(x + 1) + z[1]
        }
    except:
        pass

    # Exponencial
    try:
        popt, _ = curve_fit(exp_func, x_num, y, p0=[1, 0.1, 1])
        y_pred = exp_func(x_num, *popt)
        fits['Exponential'] = {
            'y_pred': y_pred,
            'r2': r2_score(y, y_pred),
            'func': lambda x: exp_func(x, *popt)
        }
    except:
        pass

    # Encontrar el mejor ajuste
    best_fit = max(fits.items(), key=lambda x: x[1]['r2'])
    return best_fit
