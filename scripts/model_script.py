import tensorflow as tf
from keras import Sequential, layers, Input, models, utils
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
import gc
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from data_processer import generador_de_registros_images
import os


l_encoder = LabelEncoder()
hot_encoder = OneHotEncoder()

def normalizar_images(data_df:pd.DataFrame, images:list):
    
    images_ids = (data_df['isic_id'])
    for i, id in enumerate(images_ids):
        img  = cv2.imread(f'ISIC-images/{id}.jpg',cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img,(32, 32))
            img = img.astype('float32') / 255.0 #normalizar   
            images.append(img)
            
        del img
        if i % 100 == 0:
            gc.collect()
            cv2.waitKey(1)

def preparar_data(data_df:pd.DataFrame, objetivo:str):
    
    X = data_df.drop(columns=[objetivo, 'benign_malignant' if objetivo == 'diagnosis' else 'diagnosis'])
    y  = data_df[objetivo]
    images=[]

    if y.dtype == 'object':
        y = l_encoder.fit_transform(y)

        clase_a_num = dict(zip(l_encoder.classes_, l_encoder.transform(l_encoder.classes_)))
        print("Diccionario de Clases: ", clase_a_num)
        y = utils.to_categorical(y)



    normalizar_images(data_df, images)

    X.drop(columns='isic_id', inplace=True)
    return X, y, images

def plot_history(trainning, metrics:list):
    history = trainning.history

    # Graficar la pérdida
    plt.figure(figsize=(12, 6))

    # Pérdida de entrenamiento y validación
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history['val_loss'], label='Pérdida de validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Graficar precisión
    plt.subplot(1, 2, 2)
    plt.plot(history[f'{metrics[0][0]}'], label='Precisión de entrenamiento')
    plt.plot(history[f'val_{metrics[0][0]}'], label='Precisión de validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Mostrar las gráficas
    plt.tight_layout()
    plt.savefig(f'graficos/model_performance/historial_{objetivo}.png')
    plt.show()


if __name__ == '__main__':

    # print(os.listdir())
    for objetivo in ['diagnosis']: 
        loss='categorical_crossentropy'
        metrics=['categorical_accuracy']
        output_activation = 'softmax'

        print(f'\n----Preparando Datos para {objetivo}----\n')
        data =  pd.read_csv(f'data/ham10000_metadata_balanced_{objetivo}.csv', engine = 'c')
        test_data = pd.read_csv(f'data/ham10000_metadata_leftover_{objetivo}.csv', engine = 'c').sample(frac=0.5)
        
        #missing_classes = ~val_data[val_data['diagnosis'].isin(test_data['diagnosis'])]
        #se toma una muestra del dataset balanceado y esa muestra se elimina del dataset original
        val_data =data.sample(frac=0.5, random_state=42)
        data = data.drop(val_data.index)
        
        data = generador_de_registros_images(data)
        val_data = generador_de_registros_images(val_data)
        test_data = generador_de_registros_images(test_data)

        missing_classes = val_data[~val_data['diagnosis'].isin(test_data['diagnosis'])]
        missing = missing_classes['diagnosis'].unique()

        print(missing)
        if missing.size > 0:
            print(f'Hay clases que no estan en test: {missing} ')
            for clase in missing:
                test_data = pd.concat([test_data, val_data[val_data['diagnosis']==clase]], ignore_index=True)

        repeated_registers = data[data['isic_id'].isin(val_data['isic_id'])]
        repeated_registers = repeated_registers['isic_id'].unique()
        print(f'\nNúmero de ids repetidos en datos de entrenamietno y validación: {repeated_registers}\n')

        data.to_csv(f'data/data_img_{objetivo}.csv', index=False)
        val_data.to_csv(f'data/val_data_{objetivo}.csv', index=False)

        print("\nDisposicion de los datos: \n")
        print(f'data columns: {data.columns.size}, test_columns: {test_data.columns.size} y val_columns: {val_data.columns.size}')
        print(f'data shape: {data.shape}, test_shape: {test_data.shape} y val_shape: {val_data.shape}')
        
        if objetivo == 'benign_malignant':
            loss= 'binary_crossentropy'
            metrics= ['accuracy']
            output_activation = 'sigmoid'
            data = pd.get_dummies(data, columns=['diagnosis'])
            test_data = pd.get_dummies(test_data, columns=['diagnosis'])
            val_data = pd.get_dummies(val_data, columns=['diagnosis'])


        X_train, y_train, images_train = preparar_data(data, objetivo)
        images_train = np.array(images_train)
        y_train = np.array(y_train)
        
        X_val, y_val, images_val = preparar_data(val_data, objetivo)
        images_val = np.array(images_val)
        y_val= np.array(y_val)

        X_tests, y_test, images_test = preparar_data(test_data, objetivo)
        images_test = np.array(images_test)
        y_test = np.array(y_test)

        num_output = np.unique(y_train).size
        
        if num_output <= 2:
            num_output = 1
        print(num_output)

        print("\nX_train shape: ", X_train.shape)
        print("images_train shape: ", images_train.shape)
        print("y_train shape :", y_train.shape)

        print("\nX_val shape:", X_val.shape)
        print("images_val shape:", images_val.shape)
        print("y_val shape:", y_val.shape)

        print("\nX_test shape: ", X_tests.shape)
        print("images_test shape: ", images_test.shape)
        print("y_test shape: ", y_test.shape)

        print(f'{set(X_train) - set(X_tests)}')
        num_input_columns = (X_train.columns.size)
        #print(num_input_columns) 
        #inputs 
        image_input =  Input(shape=(32, 32, 3), name= "image_input")
        metadata_input = Input(shape=(num_input_columns,), name = "metadata")

        x = layers.Conv2D(16, (3, 3), activation='leaky_relu')(image_input)
        x = layers.MaxPooling2D((2, 2))(x)
        # x = layers.Conv2D(32, (3, 3), activation='leaky_relu')(x)
        # x = layers.MaxPooling2D((2, 2))(x)
        # x = layers.Conv2D(16, (3, 3), activation='leaky_relu')(x)
        # x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)  
        
        x = layers.Concatenate()([x, metadata_input])        

        x = layers.Dense(16, activation='leaky_relu')(x)
        x = layers.Dropout(0.5)(x)
        # x = layers.Dense(16, activation='leaky_relu')(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(8, activation='leaky_relu')(x)
        # x = layers.Dropout(0.5)(x)

        output = layers.Dense(8, activation=output_activation)(x)

        model = models.Model(inputs = [image_input, metadata_input], outputs = output)
        
        model.compile(optimizer='adam',
              loss=loss,
              metrics=metrics)
        
        model.summary()

        trainning = model.fit(
            x = {
                    'image_input': images_train,
                    'metadata': X_train                
            },
            y = y_train,
            epochs=100,
            batch_size= 64, 
            validation_data=(
                {
                    'image_input': images_val,
                    'metadata': X_val
                
            },
            y_val)
        )

        plot_history(trainning, [metrics, loss])


        print("Evaluate on test data")
        results = model.evaluate(
            x={
                'image_input': images_test,
                'metadata': X_tests 
            },
            y=y_test)

        print("test loss, test acc:", results)

        print("Generar predicciones de 30 registros aleatorios de test")

        #sample  = data_test.sample(n=10, random_state=1)
        #print(sample)
        random_samples_ids = np.random.choice(len(images_test), size=1000, replace=False)
        predictions = model.predict(x ={
            'image_input': images_test[random_samples_ids],
            'metadata': X_tests.iloc[random_samples_ids] 
        })

        if output_activation == 'softmax':
            predicted_probabilities = tf.nn.softmax(predictions, axis=1)
            predicted_labels = np.argmax(predicted_probabilities, axis=1)  
            original_prob = tf.nn.softmax(y_test[random_samples_ids], axis=1)
            original_labels = np.argmax(original_prob,axis=1)     
            print(f"\ncomparacion predicciones {predicted_labels} vs. \noriginal: {original_labels}\n")
            cm = confusion_matrix(original_labels, predicted_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix= cm)
            disp.plot()
            plt.savefig(f'graficos/model_performance/confusion_matrix_{objetivo}.png')
            plt.show()
        else:
            y_pred_classes = (predictions > 0.5).astype("int")
            print(f"comparacion {y_pred_classes} vs. {y_test[random_samples_ids]} \n")
        
        print("predictions shape:", predictions.shape)



        # predictions = model.predict(x ={
        #     'image_input': images_val[random_samples_ids],
        #     'metadata': X_val.iloc[random_samples_ids] 
        # })


