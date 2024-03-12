from sklearn.linear_model import LogisticRegression

import json
import pandas as pd
import toml
import sys
import os

def load_config():
    with open('config.toml', 'r') as file:
        config = toml.load(file)
    return config

config = load_config()
URL = config['paths']['URL']
FILENAME = config['paths']['FILENAME']
DIR = config['paths']['DIR']
JSON_PATH = config['paths']['JSON_PATH']
WORK_DIR = config['paths']['WORK_DIR']

sys.path.append(WORK_DIR)
os.chdir(WORK_DIR)

from src.ml.train import TrainingPipeline
from src.ml.preprocessing import DataLoader, DataPreprocessor
from src.ml.predict import PredictionPipeline

# backend
from src.backend import app
import uvicorn


if __name__ == '__main__':

    #URL = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    #FILENAME = 'customer_churn.csv'
    #DIR = 'data'
    #JSON_PATH = 'json/inference.json'


    data_loader = DataLoader(URL, FILENAME, DIR)
    data_preprocessor = DataPreprocessor(data_loader)

    # initial_df
    initial_df = data_preprocessor.preprocess()


    #saving train and test sets
    #data_preprocessor.split_and_save_data()

    train_set, test_set = data_loader.get_data(split=True)

    # building the training pipeline and choosing a pipeline to pass it there
    categorical, numerical = data_preprocessor.get_categorical_numerical()

    # Initialize the pipeline
    logistic_regression = LogisticRegression(C=0.1, max_iter=1000)

    # Initialize the instance and then build the pipeline
    train_ = TrainingPipeline(logistic_regression, categorical, numerical)
    training_pipeline = train_.build_pipeline()

    #train and predict using kfold
    final_pipeline, avg_score, std_score = train_.kfold(train_set,
                                                        training_pipeline)

    print("C:",logistic_regression.C," comes with values: ",(avg_score, std_score))

    # Results in test (auc score)
    #test = PredictionPipeline(train_)
    #print(test.roc_score_test(train_set,
     #                         test_set,
      #                        final_pipeline))


    #save the model as pickle file
    train_.save_pickle(final_pipeline, 'pipeline/pipeline.pkl')
    #loading the model
    inference_pipeline = train_.load_pickle('pipeline/pipeline.pkl')

    #loading the inference data
    with open(JSON_PATH, 'r') as file:
        inference_dict = json.load(file)

    inference_df = pd.DataFrame([inference_dict])
    y_pred_inference = inference_pipeline.predict_proba(inference_df)[0, 1]

    print('input:', inference_dict)
    print('output:', y_pred_inference)

    uvicorn.run(app, host = "0.0.0.0", port=8000)
























