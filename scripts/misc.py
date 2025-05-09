import pandas as pd
import matplotlib.pyplot as plt

def graficador_bar_pie(dataframe,column, dir):
    #columnas_c_faltantes = []

    valores = dataframe[column].value_counts()
    valores_faltantes = dataframe[column].isna().sum()
    name_column_aux = column.replace("/", "_")

    if valores_faltantes > 0:
        valores["Faltantes"] = valores_faltantes
        #columnas_c_faltantes.append(column, valores_faltantes)
    
    print(f'valores en la columna "{column}": {dataframe[column].unique()}\n')
    plt.figure(figsize=(17, 8)) 
    colores = plt.cm.Paired.colors
    valores.plot(kind='bar', color=colores)
    plt.title(column)
    plt.xlabel(column)
    plt.xticks(rotation=0) 
    plt.tight_layout() 
    plt.savefig(f'../graficos/{dir}/{name_column_aux}_bar.png')
    #plt.show()

    plt.figure(figsize=(17, 8)) 
    valores.plot(kind='pie')
    plt.title(column)
    plt.savefig(f'../graficos/{dir}/{name_column_aux}_pie.png')
    #plt.show()

    if valores_faltantes > 0 : return column, valores_faltantes

def graficador_hist(dataframe,column, dir):
    plt.hist(dataframe[column], bins=30, density=True, alpha=0.6, color='skyblue')
    plt.title("Distribución")
    plt.xlabel("Valor") 
    plt.ylabel("Densidad")
    plt.grid(True)
    plt.title(column)
    plt.savefig(f'../graficos/{dir}/{column}_hist.png')
    #plt.show()