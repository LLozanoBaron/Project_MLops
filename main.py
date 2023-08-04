from codigo import get_genero,get_juegos,get_specs,get_earlyaccess,get_sentiment,get_predict
import pandas as pd
from fastapi import FastAPI,HTTPException
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


@app.get("/games by year", tags=['games'])
async def juegos(year: int):
    try:
        result = get_juegos(year)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/specs by year", tags=['games'])
async def specs(year: int):
    try:
        result = get_specs(year)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/earlyacces by year", tags=['games'])
async def earlyaccess(year: int):
    try:
        result = get_earlyaccess(year)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/sentiment by year", tags=['games'])
async def sentiment(year: int):
    try:
        result = get_sentiment(year)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from fastapi import FastAPI, HTTPException, Query

app = FastAPI()

@app.get("/model", tags=['games'])
async def model(
    year,
    early_access: int = Query(..., ge=0, le=1),
    sentiment: int = Query(..., ge=1, le=9)
):
    # Validate input values
    if not 1 <= sentiment <= 9:
        raise HTTPException(status_code=400, detail="Sentiment must be between 1 and 9")
    if not 0 <= early_access <= 1:
        raise HTTPException(status_code=400, detail="Early access must be either 0 or 1")
    
    try:
        result = get_predict(year, early_access, sentiment)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

