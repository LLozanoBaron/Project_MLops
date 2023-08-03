import codigo
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
        result = codigo.genero(year)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/games by year", tags=['games'])
async def juegos(year: int):
    try:
        result = codigo.juegos(year)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/specs by year", tags=['games'])
async def specs(year: int):
    try:
        result = codigo.specs(year)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/earlyacces by year", tags=['games'])
async def earlyaccess(year: int):
    try:
        result = codigo.earlyaccess(year)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/sentiment by year", tags=['games'])
async def sentiment(year: int):
    try:
        result = codigo.sentiment(year)
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))