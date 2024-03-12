import pandas as pd

from .train import TrainingPipeline

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


class PredictionPipeline:

    def __init__(self, train_: TrainingPipeline):

        self.train_ = train_

    def roc_score_test(self, df_train:pd.DataFrame,
                             df_test: pd.DataFrame,
                             pipeline: Pipeline
                       )->float:

        self.train_.train_pipeline(df_train, df_train.churn.values, pipeline)
        y_pred = self.train_.predict(df_test, pipeline)

        auc = roc_auc_score(df_test.churn.values, y_pred)

        return auc




