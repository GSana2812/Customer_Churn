from typing import Type, List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

import numpy as np

from sklearn.metrics import roc_auc_score

import pickle

class TrainingPipeline:

    def __init__(self,
                 model:Type[BaseEstimator],
                 categorical: List[str],
                 numerical: List[str])->None:
        """
            Initialize the TrainingPipeline.

            Parameters:
                - pipeline (Type[BaseEstimator]): The machine learning pipeline.
                - categorical (List[str]): List of categorical feature names.
                - numerical (List[str]): List of numerical feature names.
        """

        self.model = model
        self.categorical = categorical
        self.numerical = numerical

    def build_pipeline(self) -> Pipeline:
        """
            Build and return a scikit-learn pipeline with preprocessing steps and the specified pipeline.

            Returns:
                - Pipeline: Scikit-learn pipeline.
        """
        categorical_pipeline = Pipeline([('one_hot_encoder', OneHotEncoder(handle_unknown="ignore"))])
        numerical_pipeline = Pipeline([('scaler', StandardScaler())])

        transformer = ColumnTransformer(transformers=[
            ("categorical", categorical_pipeline, self.categorical),
            ("numerical", numerical_pipeline, self.numerical)
        ],remainder='drop')

        pipeline = Pipeline([('transformer', transformer),
                             ('pipeline', self.model)])

        return pipeline

    def train_pipeline(self, df_train:pd.DataFrame,
              y_train: pd.Series,
              pipeline: Pipeline)-> None:
        """
            Train the pipeline on the provided training data.

            Parameters:
                - df_train (pd.DataFrame): Training feature DataFrame.
                - y_train (pd.Series): Training target variable.
                - pipeline (Pipeline): Scikit-learn pipeline.
        """

        #pipeline = self.build_pipeline()
        pipeline.fit(df_train, y_train)



    def predict(self,
                df:pd.DataFrame,
                pipeline:Pipeline)->pd.Series:
        """
            Make predictions using the trained pipeline.

            Parameters:
            - df (pd.DataFrame): DataFrame with features for prediction.
            - pipeline (Pipeline): Trained scikit-learn pipeline.

            Returns:
            - pd.Series: Predicted probabilities.
        """


        y_pred = pipeline.predict_proba(df)[:, 1]

        return y_pred


    def kfold(self,
              df_full_train: pd.DataFrame,
              pipeline:Pipeline,
              n_splits:int=5)->Tuple[float, float]:
        """
            Perform k-fold cross-validation and return the average AUC score.

            Parameters:
               - df_full_train (pd.DataFrame): Full training dataset.
               - pipeline (Pipeline): Scikit-learn pipeline.
               - n_splits (int): Number of splits for k-fold cross-validation.

            Returns:
               - Tuple[float, float]: Average AUC score, standard deviation of AUC scores.
        """

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        scores = []

        for train_idx, val_idx in kfold.split(df_full_train):
            df_train = df_full_train.iloc[train_idx]
            df_val = df_full_train.iloc[val_idx]

            y_train = df_train.churn.values
            y_val = df_val.churn.values

            self.train_pipeline(df_train, y_train, pipeline)
            y_pred = self.predict(df_val, pipeline)

            auc = roc_auc_score(y_val, y_pred)
            scores.append(auc)


        return pipeline, np.mean(scores), np.std(scores)

    @staticmethod
    def save_pickle(pipeline: Pipeline, file_path: str) -> None:
        with open(file_path, 'wb') as file:
            pickle.dump(pipeline, file)
    @staticmethod
    def load_pickle(file_path: str):
        with open(file_path, 'rb') as file:
            loaded_preprocessor = pickle.load(file)

        return loaded_preprocessor
