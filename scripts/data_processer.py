import pandas as pd
import cv2  #opencv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
from misc import graficador_bar_pie, graficador_hist
import gc
import numpy as np

def one_hot_encoder(data_df:pd.DataFrame):
    #one-hot encoding
    columns_to_ignore  = ["benign_malignant", "concomitant_biopsy", "isic_id", "melanocytic", "sex", 
                        "age_approx", "diagnosis"]
    data_df = pd.get_dummies(data_df, columns=data_df.drop(columns = columns_to_ignore).columns[0:].tolist(), dtype=int)

    data_df.to_csv('../data/ham10000_metadata_one_hot_encoded.csv', index=False)

    return data_df

def balancer(data_df:pd.DataFrame):
    rus = RandomUnderSampler(random_state=42)
    for columna in ['benign_malignant', 'diagnosis']:  
        X = data_df.drop(columns= columna)
        y = data_df[columna]
        x_resampleado, y_resampleado = rus.fit_resample(X, y)

        df_balanced = pd.DataFrame(x_resampleado, columns=X.columns)
        df_balanced[columna] = y_resampleado

        df_leftover = data_df[~data_df['isic_id'].isin(df_balanced['isic_id'])].sample(frac=0.3)

        df_balanced.to_csv(f'../data/ham10000_metadata_balanced_{columna}.csv', index=False)
        df_leftover.to_csv(f'../data/ham10000_metadata_leftover_{columna}.csv', index=False)

        for column in df_balanced.columns[1:]:
            if df_balanced[column].dtype == 'number':
                graficador_hist(df_balanced, column, f'balanceo/{columna}')
                graficador_hist(df_leftover, column, f'leftover/{columna}')
            else:
                graficador_bar_pie(df_balanced, column, f'balanceo/{columna}')
                graficador_bar_pie(df_leftover, column, f'leftover/{columna}')

#este método rellena los faltantes con la moda de cada columna
def fill_na(data_df:pd.DataFrame):
    for columna in data_df.columns[1:]:
        #print(columna)
        if columna != None:
            data_df[columna] = data_df[columna].fillna("Faltante")
            moda = data_df[columna].mode()
            print(f'moda de la columna {columna} es {moda.iloc[0]}')
            if columna == "benign_malignant":
                data_df[columna] = data_df.apply(
                    lambda row: row['diagnosis_1'] if row[columna] == 'Faltante' or row['diagnosis_1'] != 'Indeterminate' 
                    else moda.iloc[0], axis=1)
                
            elif not moda.empty and moda.iloc[0] != 'Faltante':
                data_df[columna].replace("Faltante", moda.iloc[0], inplace=True)
            else:
                data_df.drop(columns=columna, inplace=True) 
    


    data_df.to_csv("../data/ham10000_metadata_no_nan.csv", index=False)

# esta función genera "nuevos" registros al voltear aleatoriamente todas la imágenes 
# y asignarles a estas los metadatos de las imágenes originales
def generador_de_registros_images(data_df:pd.DataFrame):

    #images_ids = list(data_df[data_df['benign_malignant'] == 0]['isic_id'])
    images_ids = list(data_df['isic_id'])
    for i, id in enumerate(images_ids):
        
        original_row = data_df[data_df['isic_id'] == id].iloc[0].copy()
        img  = cv2.imread(f'ISIC-images/{id}.jpg',cv2.IMREAD_COLOR)
        
        imgrot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        id_rotated_image = id+'_r'
        #print(id_rotated_image)
        cv2.imwrite(f'ISIC-images/{id_rotated_image}.jpg', imgrot)
        rotated_row = original_row.copy()
        rotated_row['isic_id'] = id_rotated_image
        
        imgblur = cv2.GaussianBlur(img, (15, 15), 0)
        id_blured_image = id+'_b'
        cv2.imwrite(f'ISIC-images/{id_blured_image}.jpg', imgblur)
        blured_row = original_row.copy()
        blured_row['isic_id'] = id_blured_image
        
        noise = np.random.normal(0, 1, img.shape).astype(np.uint8)
        imgnoisy = cv2.add(img, noise)
        id_noisy_image = id +'_n'
        cv2.imwrite(f'ISIC-images/{id_noisy_image}.jpg', imgnoisy)
        noised_row = original_row.copy()
        noised_row['isic_id'] = id_noisy_image
        

        new_rows = pd.DataFrame([rotated_row, blured_row, noised_row])
        data_df = pd.concat([data_df, new_rows], ignore_index=True)

        # Limpiar las variables de imágenes para liberar memoria
        del img
        del imgrot
        del imgblur
        del imgnoisy
        
        # #original_row['isic_id'] = id_rotated_image

        if i % 100 == 0:
            gc.collect()
            cv2.waitKey(1)
    
    return data_df



if __name__ == '__main__':
    standar_scaler  = StandardScaler()
    isic_pd  = pd.read_csv("../data/ham10000_metadata_dropped_columns.csv")

    isic_pd.loc[:, isic_pd.columns != 'isic_id'] = isic_pd.loc[:, isic_pd.columns != 'isic_id'].applymap(
    lambda x: x.lower() if isinstance(x, str) else x
    )
#todo en minusculas

    numeric_columns =  list(isic_pd.select_dtypes(include=['number']).columns)
    print(isic_pd.columns)
    fill_na(isic_pd)
    print(isic_pd.columns)
    for column in isic_pd.columns[1:]:  
        graficador_bar_pie(isic_pd,column, "no_nan")

    for column in numeric_columns[1:]:
        graficador_hist(isic_pd,column, "no_nan")

    #columnas boolenas se pasan a int
    for col in isic_pd.select_dtypes(include=["bool"]).columns:
        isic_pd[col] = isic_pd[col].astype(int)

     # #columnas binarias no bool se pasan a int
    isic_pd["sex"] = (isic_pd["sex"] == "male").astype(int) 
    isic_pd["benign_malignant"] = (isic_pd["benign_malignant"] == "benign").astype(int) #benigno 1, maligno 0

    #estandarizar columna age (real)
    isic_pd['age_approx'] =  standar_scaler.fit_transform(isic_pd[["age_approx"]])

    print(isic_pd.columns)
    #isic_pd = generador_de_registros_images(isic_pd)
    
    isic_pd.to_csv("../data/ham10000_metadata_preprocessed.csv", index=False)
    isic_pd =  one_hot_encoder(isic_pd)

    balancer(isic_pd)

