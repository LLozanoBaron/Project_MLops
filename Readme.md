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

## **Prueba de la API Games app and ML**
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

## **Caracteristicas de la API**
Aqui se encuentran algunas de las mejores caracteristicas de la API:

+ Su uso es sencillo, solamente se deben ingresar los años requeridos de forma numérica y se dara el resultado.
+ El modelo tiene parametros sencillos de entender y rapidos de escribir.
+ La veracidad del modelo fue rectificada en multiples ocasiones , resultando en un RMSE menor de 10.
+ Se tienen los generos más conocidos para un mejor resultado predictivo.

## **Construcción de la API**

la API fue construida con:
+ VSCode
+ Python 3.7.9
+ FastAPI 0.100.1
+ Uvicorn 0.22.0

Los siguientes codigos son parte de la estructura de la API:

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

## **Recursos**
https://github.com/soyHenry/PI_ML_OPS.git








