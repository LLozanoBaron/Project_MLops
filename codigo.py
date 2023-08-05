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

#Para la primera funcion 

def get_genero(year: int):
    df = df_anid[['release_date','genres']]
    
#Convert the time to datetime object.Then it converts the datetime object to period object with a yearly frequency
    years = pd.to_datetime(year,format = '%Y').to_period('Y')
    
#Filter the selectiong the rows only when the period is equal to the year stored in the variable
    df_filter = df[df['release_date'].dt.to_period('Y') == years]
    df_top = df_filter['genres'].value_counts()
    top_genres = df_top.head(5)
    return {year: top_genres.to_dict()}

#print(genero(2014))

    
def get_juegos(year: int):
    df = dff[['release_date','app_name']]
#Convert the time to datetime object.Then it converts the datetime object to period object with a yearly frequency
    years = pd.to_datetime(year,format = '%Y').to_period('Y')
    
#Filter the selectiong the rows only when the period is equal to the year stored in the variable
    df_filter = df[df['release_date'].dt.to_period('Y') == years]
    
#Converted the column into a list,transform to string and contain it in the variable
    juegos_lanzados = df_filter['app_name'].astype(str).tolist()
    juegos_dict = {juego for juego in juegos_lanzados}
    return {year: juegos_dict}

#print(juegos(2014))

def get_specs(year: int):
    df = dff[['release_date','specs']]
    years = pd.to_datetime(year,format = '%Y').to_period('Y')
    df_filter = df[df['release_date'].dt.to_period('Y') == years]
    df_filter = df_filter.explode('specs')
    df_top = df_filter['specs'].value_counts()
    top_specs = df_top.head(5)
    return {year: top_specs.to_dict()}

#print(specs(2014))

def get_earlyaccess(year: int):
    df = dff[['release_date','early_access']]
    years = pd.to_datetime(year,format = '%Y').to_period('Y')
    df_filter = df[df['release_date'].dt.to_period('Y') == years]    
    juegos_early = df_filter['early_access']
    num_early_access = juegos_early.value_counts() #Count the values True and False of the column
    return {year: num_early_access.to_dict()}

#print(earlyaccess(2014))

def get_sentiment( year : int):
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
    return {year: num_critics.to_dict()}

#print(sentiment(2014))

import pickle

def get_predict(year, early_access, sentiment,genre):
    
    # Load the saved model from a file
    with open('model_and_rmse.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Create input data for prediction
    X = [[year, early_access, sentiment,genre]]
    
    # Make prediction
    y_pred = model.predict(X)
    
    return y_pred
