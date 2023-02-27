import os
import boto3
import pandas as pd
from io import StringIO
from abc import ABC
from typing import List, Union
from pandas.core.frame import DataFrame
from numpy import ndarray

class TrainTestDataProvider(ABC):

    def __init__(self) -> None:
        self._loaded = False
        self._uri:str = None
        self._df: DataFrame = None
        self._train_column:str = None
        self._train_query: str = None
        self._test_query:str = None
        self._x_columns:List[str] = None
        self._y_columns:Union[str,List[str]] = None

    def _get_s3_resource(self):
        return boto3.resource(
            's3',
            endpoint_url = f'http://{os.environ.get("S3_HOST")}:{os.environ.get("S3_PORT")}',
            aws_access_key_id = os.environ.get("S3_AUTH_KEY_ID"),
            aws_secret_access_key = os.environ.get("S3_AUTH_SECRET_KEY")
        )

    def _get_s3_object(self):
        resource = self._get_s3_resource()
        return resource.Object(
            os.environ.get("S3_BUCKET"),
            self._uri.replace('s3://','')
        )

    def _get_df(self) -> DataFrame:
        if not self._loaded:
            self.load()
        return self._df

    def load(self) -> None:
        obj = self._get_s3_object().get()['Body']
        self._df = pd.read_csv(obj)
        self._loaded = True

    def save(self) -> None:
        csv_buffer = StringIO()
        self._get_df().to_csv(csv_buffer, index=False)
        self._get_s3_object().put(Body=csv_buffer.getvalue())

    def set_df(self, df):
        self._df = df
        self._loaded = True

    def can_be_x_column(self, column:str, allowed_dtypes:List[str] = ['float64','int64']) -> bool:
        if self._train_column and column == self._train_column:
            return False
        if type(self._y_columns) == str and column == self._y_columns:
            return False
        if type(self._y_columns) == list and column in self._y_columns:
            return False
        if column not in list(self._get_df().columns):
            return False
        if str(self._get_df()[column].dtype) not in allowed_dtypes:
            return False
        return True

    def get_train_column(self) -> str:
        return self._train_column

    def set_train_column(self, column:str) -> None:
        self._train_column = column

    def get_train_query(self) -> str:
        if self._train_query:
            return self._train_query
        if self._train_column:
            return f'`{self._train_column}` == 1'
        raise Exception('Can not build train query')

    def get_test_query(self) -> str:
        if self._train_query:
            return self._train_query
        if self._train_column:
            return f'`{self._train_column}` == 0'
        raise Exception('Can not build test query')

    def set_uri(self, uri: str):
        self._uri = uri

    def get_dataframe(self) -> DataFrame:
        return self._get_df()

    def get_train_dataframe(self) -> DataFrame:
        return self._get_df().query(self.get_train_query())

    def get_test_dataframe(self) -> DataFrame:
        return self._get_df().query(self.get_test_query())

    def set_x_columns(self, x_columns:List[str]) -> None:
        self._x_columns = x_columns

    def get_x_columns(self, columns:List[str] = None) -> List[str]:
        return columns or self._x_columns

    def set_y_columns(self, y_columns:Union[str,List[str]]) -> None:
        self._y_columns = y_columns

    def get_y_columns(self, columns:Union[str,List[str]] = None) -> Union[str,List[str]]:
        return columns or self._y_columns

    def get_x_train(self, columns:List[str] = None) -> ndarray:
        return self._get_df().query(self.get_train_query())[self.get_x_columns(columns)].to_numpy()

    def get_y_train(self, columns:Union[str,List[str]] = None) -> ndarray:
        return self._get_df().query(self.get_train_query())[self.get_y_columns(columns)].astype('float64').to_numpy()

    def get_x_test(self, columns:List[str] = None) -> ndarray:
        return self._get_df().query(self.get_test_query())[self.get_x_columns(columns)].to_numpy()

    def get_y_test(self, columns:Union[str,List[str]] = None) -> ndarray:
        return self._get_df().query(self.get_test_query())[self.get_y_columns(columns)].astype('float64').to_numpy()
