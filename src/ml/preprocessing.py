import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from typing import Tuple, List


import requests

# to be saved in a config.toml file

class DataLoader:

    def __init__(self, url:str, filename:str, data_dir: str)->None:

        self.url = url
        self.filename = filename
        self.data_dir = data_dir
        self.full_path = f"{self.data_dir}/{self.filename}"

    def download_csv(self)->None:

        """
            Download a CSV file from a given URL and save it to a local file.

            Parameters:
                url (str): The URL of the CSV file.
                filename (str): The local filename to save the downloaded CSV file.

            Returns:
                None
        """

        response = requests.get(self.url)

        with open(self.full_path, 'wb') as file:
            file.write(response.content)

    def get_data(self, split=False)->pd.DataFrame:


        if split:
            # Load df_train and df_test if split is True
            train_filepath = f"{self.data_dir}/df_full_train.csv"
            test_filepath = f"{self.data_dir}/df_test.csv"

            df_train = pd.read_csv(train_filepath, index_col='customerid')
            df_test = pd.read_csv(test_filepath, index_col='customerid')

            return df_train, df_test

        else:
            # Download the initial dataset and return it
            #output_filename = f'{self.data_dir}/{self.filename}'

            self.download_csv()
            df = pd.read_csv(self.full_path)

            return df


class DataPreprocessor:

    def __init__(self, data_loader)->None:
        self.data_loader = data_loader


    def preprocess(self)->pd.DataFrame:

        """
            Preprocess the dataset by cleaning and transforming it.

            Returns:
                pd.DataFrame: The preprocessed dataset with 'customerid' as the row index.
        """


        df = self.data_loader.get_data(split=False)
        # remove the white space between columns and replace them with _
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
        # perform same operation with columns values
        for c in categorical_columns:
            df[c] = df[c].str.lower().str.replace(' ', '_')
        # impute missing values of totalcharges
        df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
        df.totalcharges = df.totalcharges.fillna(0)

        # convert from string to number for predictions
        df.churn = (df.churn == 'yes').astype(int)

        # make customerid as index for rows
        df.set_index('customerid', inplace=True)

        return df



    def get_categorical_numerical(self)-> Tuple[List[str], List[str]]:

        """
            Identify categorical and numerical columns in the preprocessed dataset.

            Returns:
                Tuple[List[str], List[str]]: Lists of categorical and numerical column names.
        """

        df = self.preprocess()

        # get numerical and categorical columns
        #categorical = list(set(df.dtypes[df.dtypes == 'object'].index) | set(col for col in df.columns if
                                                                         #df[col].nunique() <= 2 and col != 'churn'))

        # we get only categorical values, which are already object
        # categorical values like seniorcitizen or churn don't get included, we only want the 'str' types
        categorical = list(df.dtypes[df.dtypes == 'object'].index)

        numerical = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] if df[col].nunique() > 2 and
                 col != 'churn']

        return categorical, numerical

    def split_and_save_data(self)->None:

        """
            Split the preprocessed dataset into training and test sets and save them as CSV files.

            Returns:
                None
        """

        df = self.preprocess()

        # split it only on training and test data, so we can perform later some kfold
        df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

        df_full_train.to_csv('data/df_full_train.csv')
        df_test.to_csv('data/df_test.csv')










