from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("./model_lr.pkl")

class request_body(BaseModel):
    mean_integrated_profile: float
    std_integrated_profile: float
    kurtosis_integrated_profile: float
    skewness_integrated_profile: float
    mean_dmsnr_curve: float
    std_dmsnr_curve: float
    kurtosis_dmsnr_curve: float
    skewness_dmsnr_curve: float

@app.post("/classify")
def predict(data: request_body):
    input_features = [[
      data.mean_integrated_profile,
      data.std_integrated_profile,
      data.kurtosis_integrated_profile,
      data.skewness_integrated_profile,
      data.mean_dmsnr_curve,
      data.std_dmsnr_curve,
      data.kurtosis_dmsnr_curve,
      data.skewness_dmsnr_curve
    ]]

    y_pred = int(model.predict(input_features)[0])
    y_prob = model.predict_proba(input_features)[0]

    resposta = 'Estrela de Nêutron Pulsar' if y_pred == 1 else "Não Estrela de Nêutron Pulsar"
    probabilidade = y_prob[y_pred]

    return {
        "qualidade": resposta,
        "probabilidade_de_ter_qualidade_acima": probabilidade,
    }
