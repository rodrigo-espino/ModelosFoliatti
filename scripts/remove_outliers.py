def remove_outliers(df, column):
    """
    Elimina outliers de un DataFrame basándose en la desviación estándar.

    Args:
    df: DataFrame de pandas.
    column: Nombre de la columna que contiene los valores a analizar.

    Returns:
    DataFrame de pandas sin outliers.
    """
    # Calcula la media y la desviación estándar
    mean = df[column].mean()
    std = df[column].std()

    # Define los límites superior e inferior para identificar outliers
    upper_bound = mean + 3 * std
    lower_bound = mean - 3 * std

    # Filtra el DataFrame para eliminar outliers
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return filtered_df