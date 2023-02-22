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

    def set_uri(self, uri: str):
        self._uri = uri

    def get_dataframe(self) -> DataFrame:
        return self._get_df()

    def get_x_columns(self) -> List[str]:
        return self._x_columns

    def get_y_columns(self) -> Union[str,List[str]]:
        return self._y_columns

    def get_x_train(self) -> ndarray:
        return self._get_df().query(self._train_query)[self.get_x_columns()].to_numpy()

    def get_y_train(self) -> ndarray:
        return self._get_df().query(self._train_query)[self.get_y_columns()].to_numpy()

    def get_x_test(self) -> ndarray:
        return self._get_df().query(self._test_query)[self.get_x_columns()].to_numpy()

    def get_y_test(self) -> ndarray:
        return self._get_df().query(self._test_query)[self.get_y_columns()].to_numpy()
