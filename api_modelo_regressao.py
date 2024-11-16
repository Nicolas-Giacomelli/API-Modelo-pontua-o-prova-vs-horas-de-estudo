from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

#Criar uma instancia no fastapi
app = FastAPI()

#Criar uma classe para validar os dados do request body da api
class request_body(BaseModel):
    horas_estudo : float

#Carregar o modelo para realizar a predição
modelo_pontuacao = joblib.load('./modelo_regressao.pkl')

#definindo metodo de exibição
@app.post('/predict')


def predict(data:request_body):
    #Preparar os dados para predicao
    input_feature = [[data.horas_estudo]]

    #Realizar a predição
    y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

    return {'pontuacao_teste': y_pred.tolist()}