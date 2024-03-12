from fastapi import FastAPI, HTTPException
from src.ml.train import TrainingPipeline

from pydantic import BaseModel
import pandas as pd
import gunicorn


app = FastAPI()
MODEL = TrainingPipeline.load_pickle('pipeline/pipeline.pkl')

class PredictionInput(BaseModel):

    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    tenure: int
    monthlycharges: float
    totalcharges: float

    def to_df(self):

        data = self.dict()
        df = pd.DataFrame([data])
        return df


@app.post('/predict_churn')
async def predict_churn(data: PredictionInput):

    """Performing predictions and sending the requests through a web server
    The input will be our json file
    """

    #data_df = pd.DataFrame([data.dict()])
    y_pred_inference = MODEL.predict_proba(data.to_df())[0, 1]
    churn = int(y_pred_inference >= 0.5)
    return {"Probability":y_pred_inference,
            "Churn":churn}




