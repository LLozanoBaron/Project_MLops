<h1 align="center"> PROJECT MLOps </h1>

## **INTRODUCTION**

In this individual project, we will use the Steam video game database. We will develop several queries and a Machine Learning model. The queries and the model will be delivered in an API developed in the FastAPI framework.

## **OBJECTIVES**
For this project, we have several objectives to achieve in the API to be developed, which we will enumerate below:

+ Create a function in which, when entering the year, the result will be the five (5) most offered genres. 

+  Create a function in which, when entering the year, the result will be a dictionary with all the games launched in that year.

+ Create a function in which, when entering a year, it returns a dictionary with the five (5) specs that are most repeated in the same year in the corresponding order.

+ Create a function in which, when entering a year, it returns a dictionary with the number of games launched in a year with early access.

+ Create a function in which, based on the release year, it returns a dictionary with the number of records that are categorized with a sentiment analysis. 

+ Create a function that returns a dictionary with the top 5 games by year with the highest metascore.

+ Perform the corresponding data cleaning to be able to carry out an exploratory data analysis and a machine learning model in accordance with the data.

+ Create a function for the chosen prediction model, in which, when entering the selected parameters, it returns the price and the RSME.

+ Deploy the API on render.com so that it can be used freely.

## **API TEST FOR ML PROJECT**
In the following link, you can find the API created in the project: https://games-app-and-ml.onrender.com/docs

In this, you will be able to see the functions performed and their respective results. In the following images, you will be able to see not only what they look like but also some results.

+ First view of the API ![API 1](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/4cdc7a2c-7344-4c05-bc0f-c9eaeb06f9c8)

+ Test of the genero function ![API 2](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/7868da8b-e671-4017-969c-6a49bf1e6c81)

+ Test of the juegos function ![API 3](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/a7425cf8-218d-4ac5-a714-cd9708c1ceac)

+ Test of the specs function ![API 4](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/1711d806-af32-42f9-a876-2a47a69e0ac6)

+ Test of the earlyacces function ![API 5](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/845af2ba-70f4-42b6-a638-ef184bfcf374)

+ Test of the sentiment function ![API 6](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/ba1a250f-8167-4979-8324-b508748f6db7)

+ Test of the metascore function ![API 7](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/ac2bb0d1-8441-40ae-b8c4-f854b620b39a)

+ Test of the predictive model ![API 8](https://github.com/LLozanoBaron/Proyecto_MLops/assets/125699712/f42c1463-acfb-43db-85cb-d41a3bf1dd42)

## **FEATURES OF THE ML PROJECT API***
Here are some of the best features of the API:

+ Its use is simple, you only need to enter a year numerically and the result will be given.
+ The model has parameters that are easy to understand and quick to write.
+ The accuracy of the model was verified on multiple occasions, resulting in an RMSE of less than 10.
+ The most well-known genres are available for a better predictive result.

## **CONSTRUCTION OF THE ML PROJECT API***

The API was built with:
+ VSCode
+ Python 3.7.9
+ FastAPI 0.100.1
+ Uvicorn 0.22.0

The following codes are part of the structure of the API:

+ This code was used for reading and cleaning some data before performing the functions.

``` python
import ast 
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
df_anid = dff.explode('genres')
```

+ With this code, the import of the functions and the construction of the API and its functions in main.py was carried out.

``` python
from codigo import get_genero,get_juegos,get_specs,get_earlyaccess,get_sentiment,get_predict,get_metascore
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import ast

app = FastAPI()
app.title = 'PROYECTO ML'

@app.get("/obtener generos", tags=['games'])
async def genero_top_5(year: int):
    try:
        result = get_genero(year)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

+ The following code shows how the get_genres function was built in codigo.py

``` python
def get_genero(year: int):
    df = df_anid[['release_date','genres']]
    
#Convert the time to datetime object.Then it converts the datetime object to period object with a yearly frequency
    years = pd.to_datetime(year,format = '%Y').to_period('Y')
    
#Filter the selectiong the rows only when the period is equal to the year stored in the variable
    df_filter = df[df['release_date'].dt.to_period('Y') == years]
    df_top = df_filter['genres'].value_counts()
    top_genres = df_top.head(5)
    return {year: top_genres.to_dict()}
```

+ The following code shows the chosen Random Forest Regression model

``` python
#Use the genres we selected had more frequency 
genres = ['Indie','Action','Adventure','Casual','Simulation','Strategy','RPG','Early Access','Free to Play','Sports','Massively Multiplayer']

# We select the predictor variables X and the variable to predict y

X = df_encoded[['year', 'early_access', 'sentiment']+genres]
y = df_encoded['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#Try again the model with the best estimator
rf2 = RandomForestRegressor(n_estimators = 200, max_features = 'sqrt', max_depth = 5, random_state = 18).fit(X_train, y_train)
```

+ In the following code, the calculation of the RMSE can be observed.

``` python
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

#Calculate the RMSE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import mean_squared_error

rmse_train = (mean_squared_error(y_train, y_train_pred, squared = False))
rmse_test = (mean_squared_error(y_test, y_test_pred, squared = False))
print(f'Raíz del error cuadrático medio en Train: {rmse_train}')
print(f'Raíz del error cuadrático medio en Test: {rmse_test}')
```

+ In the following code, it can be seen how the predictive model was saved.

```python
import pickle

# Save the trained model and RMSE values to a file
with open('model_and_rmse.pkl', 'wb') as file:
    pickle.dump((rf2, rmse_train2, rmse_test2), file)
```

+ In this section, we can observe the get_predict function.

```python
mport pickle

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
    return {'predict price': round(to_numeric(y_pred[0]), 2), 'rmse_train': rmse_train, 'rmse_test': rmse_test}
```
## **RESOURCES**

+ https://github.com/soyHenry/PI_ML_OPS.git
+ In the following link, you can find a video in which the operation and construction of the API is explained:https://youtu.be/_wFH1XkuKzk 

## **AUTHOR**

Laura Viviana Lozano Baron

## **LICENSE**

This project is licensed under the MIT License







