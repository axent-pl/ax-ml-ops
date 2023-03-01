import os
import boto3
import pandas as pd
from io import StringIO
from abc import ABC
from typing import Dict, List, Union
from pandas.core.frame import DataFrame
from numpy import ndarray
import json

class FeaturesDataProvider(ABC):

    def __init__(self) -> None:
        self._loaded = False
        self._base_uri:str = None
        self._features_class:str = None
        self._features:List[str] = []

    def _get_s3_resource(self):
        return boto3.resource(
            's3',
            endpoint_url = f'http://{os.environ.get("S3_HOST")}:{os.environ.get("S3_PORT")}',
            aws_access_key_id = os.environ.get("S3_AUTH_KEY_ID"),
            aws_secret_access_key = os.environ.get("S3_AUTH_SECRET_KEY")
        )

    def _get_s3_object(self, features_class:str):
        resource = self._get_s3_resource()
        return resource.Object(
            os.environ.get("S3_BUCKET"),
            self._base_uri.replace('s3://','') + f'/{features_class}.json'
        )

    def set_base_uri(self, base_uri: str):
        self._base_uri = base_uri

    def set_features(self, features_class:str, features:List[str]) -> None:
        buffer = StringIO()
        json.dump({'features':features}, buffer)
        self._get_s3_object(features_class=features_class).put(Body=buffer.getvalue())

    def get_features(self, features_class:str) -> List[str]:
        obj = self._get_s3_object(features_class=features_class).get()['Body']
        return json.load(obj)['features']