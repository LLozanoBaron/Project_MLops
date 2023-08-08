<h1 align="center"> PROYECTO INDIVIDUAL MLOps </h1>

## **INTRODUCCIÓN**

En este proyecto individual vamos a utilizar la base de datos de videojuegos de Steam, en esta desarrollaremos varias consultas, asi como un modelo Machine Learning, las consultas y el modelo seran entregadas en una API desarrollada en el framework FastAPI.

## **OBJETIVOS**
Para este proyecto tenemos varios objetivos a cumplir en la API a desarrollar los cuales enemeraremos acontinuación:

+ Crear unan función en la cual al ingresar el año se dara como resultado los cinco(5) generos mas ofrecidos.

+ Crear unan función en la cual al ingresar un año devuelva un diccionario con los juegos lanzados en el año.

+ Crear unan función en la cual al ingresar un año devuelva un diccionario con los 5 specs que más se repiten en el mismo año en el orden correspondiente.

+ Crear unan función en la cual al ingresar un año devuelva un diccionario con la cantidad de juegos lanzados en un año con early access.

+ Crear unan función en la cual según el año de lanzamiento, se devuelve un diccionario con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento. 

+ Crear unan función en la cual se devuelve un diccionario con el top 5 juegos según año con mayor metascore.

+ Realizar la limpieza de datos correspondiente para poder realizar un análisis exploratorio de datos y un modelo de machine learning acorde con los datos.

+ Crear una funcion para el modelo de predicción elegido, en la cual al ingresar los parametros elegidos devuelva el precio y el RSME. 

+ Desplegar la API en render.com para que sea utilizada de forma libre.

## **Prueba de la API Proyecto ML**
En el siguiente link se puede encontrar la API realizada en el proyecto: https://games-app-and-ml.onrender.com/docs

En esta se podran observar las funicones realizadas y sus respectivos resultados, en las siguientes imagenes se podra observar no solo como es su aspecto sino algunos resultados:

+ Primera vista de la API ![API 1](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/4cdc7a2c-7344-4c05-bc0f-c9eaeb06f9c8)

+ Prueba de la función generos ![API 2](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/7868da8b-e671-4017-969c-6a49bf1e6c81)

+ Prueba de la función juegos ![API 3](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/a7425cf8-218d-4ac5-a714-cd9708c1ceac)

+ Prueba de la función specs ![API 4](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/1711d806-af32-42f9-a876-2a47a69e0ac6)

+ Prueba de la función earlyacces ![API 5](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/845af2ba-70f4-42b6-a638-ef184bfcf374)

+ Prueba de la función sentiment ![API 6](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/ba1a250f-8167-4979-8324-b508748f6db7)

+ Prueba de la función metascore ![API 7](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/ac2bb0d1-8441-40ae-b8c4-f854b620b39a)

+ Prueba del modelo predictivo ![API 8](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/f42c1463-acfb-43db-85cb-d41a3bf1dd42)

## **Caracteristicas de la API Proyecto ML***
Aqui se encuentran algunas de las mejores caracteristicas de la API:

+ Su uso es sencillo, solamente se deben ingresar los años requeridos de forma numérica y se dara el resultado.
+ El modelo tiene parametros sencillos de entender y rapidos de escribir.
+ La veracidad del modelo fue rectificada en multiples ocasiones , resultando en un RMSE menor de 10.
+ Se tienen los generos más conocidos para un mejor resultado predictivo.

## **Construcción de la API Proyecto ML***

la API fue construida con:
+ VSCode
+ Python 3.7.9
+ FastAPI 0.100.1
+ Uvicorn 0.22.0

Los siguientes codigos son parte de la estructura de la API:

+ Este codigo se uso para la lectura y limpieza de algunos datos antes de realizar las funciones
```import ast 
import pandas as pd
import json
from typing import Dict

rows = []
with open ('steam_games.json') as f: # f contains the data of the archive
     for line in f.readlines():
            rows.append(ast.literal_eval(line))
df1 = pd.DataFrame(rows)

# Change to NaT the data that is not in yyyy-mm-dd format
df1['release_date'] = pd.to_datetime(df1['release_date'], format='%Y-%m-%d', errors='coerce')

#Do a filter in release date to drop NaN
dff = df1.dropna(subset= ['release_date'])

#Change to datetime the release date
dff['release_date'] = pd.to_datetime(dff['release_date'])

#Unnest the colum genres in the dataframe
df_anid = dff.explode('genres') ```

+ Aqui se muestra el codigo de la función sentiment, en la cual se uso una mascara para determinar el sentimiento
```def get_sentiment( year : int):
    dfs = dff[['sentiment','release_date']]
    
    #Use isin to created a boolean series that indicate the value of the column
    #The ~ operator is used to invert the boolean series. 
    mask = ~dfs['sentiment'].isin(['Overwhelmingly Positive','Mostly Positive','Very Positive','Positive', 'Mixed', 'Negative','Mostly Negative','Very Negative','Overwhelmingly Negative'])
    
    #Select the rows where the mask is True and set that values in the column for None
    dfs.loc[mask, 'sentiment'] = 'None'
    years = pd.to_datetime(year,format = '%Y').to_period('Y')
    df_filter = dfs[dfs['release_date'].dt.to_period('Y') == years]    
    critics = df_filter['sentiment']
    num_critics = critics.value_counts() 
    return {year: num_critics.to_dict()} ```

+En este codigo se puede obsevar como se construyo la API
```from codigo import get_genero,get_juegos,get_specs,get_earlyaccess,get_sentiment,get_predict,get_metascore
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import ast

app = FastAPI()
app.title = 'PROYECTO ML' ```

+ En este codigo se puede observar la construcción de la función sentiment en la API
``` @app.get("/sentiment by year", tags=['games'])
async def sentiment(year: int):
    try:
        result = get_sentiment(year)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))```

+ En este codigo podemos observar el modelo machine learning utilizado
``` rf2 = RandomForestRegressor(n_estimators = 200, max_features = 'sqrt', max_depth = 5, random_state = 18).fit(X_train, y_train)```

+ El siguiente codigo nos muestra como se hallo el RMSE
```#Calculate the RMSE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import mean_squared_error

rmse_train2 = (mean_squared_error(y_train, y_train_pred, squared = False))
rmse_test2 = (mean_squared_error(y_test, y_test_pred, squared = False))
print(f'Raíz del error cuadrático medio en Train: {rmse_train}')
print(f'Raíz del error cuadrático medio en Test: {rmse_test}') ```

+ El codigo nos muestra como se guardo el modelo

```import pickle

# Save the trained model and RMSE values to a file
with open('model_and_rmse.pkl', 'wb') as file:
    pickle.dump((rf2, rmse_train2, rmse_test2), file) ```

+ En el siguiente codigo podemos observar la función utilizada para el modelo predictivo
```import pickle

from pandas import to_numeric

def get_predict(year, early_access, sentiment, genre):
    # Load the saved model from a file
    with open('model_and_rmse.pkl', 'rb') as file:
        data = pickle.load(file)
    
    # Unpack the tuple and extract the model
    model, rmse_train, rmse_test = data
    
    # Create a list of all possible genres
    all_genres = ['Indie','Action','Adventure','Casual','Simulation',
                  'Strategy','RPG','Early Access','Free to Play','Sports','Massively Multiplayer']
    
    # Create a one-hot encoded representation of the input genre
    genre_encoded = [1 if g == genre else 0 for g in all_genres]
    
    # Create input data for prediction
    X = [[year, early_access, sentiment] + genre_encoded]
    
    # Make prediction
    y_pred = model.predict(X)
    
    # Return prediction as a scalar value
    return {'predict price': round(to_numeric(y_pred[0]), 2), 'rmse_train': rmse_train, 'rmse_test': rmse_test} ```


## **Recursos**

+ https://github.com/soyHenry/PI_ML_OPS.git
+ En el siguiente link se puede encontrar un video en el cual se explica el funcionamiento y construcción de la API: 

## **AUTORA**

Laura Viviana Lozano Baron

## **Licencia**

This project is licensed under the MIT License







