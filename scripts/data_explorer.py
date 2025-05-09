import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from misc import graficador_bar_pie, graficador_hist


if __name__ == '__main__':
    isic_pd  = pd.read_csv('../data/ham10000_metadata_2025-03-26.csv')

    print(isic_pd.head())
    print(isic_pd.dtypes)

    numeric_columns =  list(isic_pd.select_dtypes(include=['number']).columns)
    print(numeric_columns)
    
    #se quitan columnas que no aportan información
    isic_pd = isic_pd.drop(columns=["lesion_id", "image_type", 'attribution', 'copyright_license'])
    
    start_column = "age_approx"

    start_index = isic_pd.columns.tolist().index(start_column)
    columnas_c_faltantes = []
    
    for column in isic_pd.columns[start_index:]:  
        columnas_c_faltantes.append(graficador_bar_pie(isic_pd,column, "exploracion_inicial"))

    print(columnas_c_faltantes)


    for column in numeric_columns:
        graficador_hist(isic_pd,column, "exploracion_inicial")

    isic_pd.to_csv('../data/ham10000_metadata_dropped_columns.csv', index=False)