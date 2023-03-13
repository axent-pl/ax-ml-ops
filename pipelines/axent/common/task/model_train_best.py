
from typing import List, Union
from ..data import TrainTestDataProvider
from ..data import FeaturesDataProvider
from airflow.models.taskinstance import TaskInstance

class ModelTrainBest:

    def __init__(self, data_provider:TrainTestDataProvider, features_provider: FeaturesDataProvider = None) -> None:
        self._dp:TrainTestDataProvider = data_provider
        self._fp:FeaturesDataProvider = features_provider

    def execute(self, ti:TaskInstance, task_ids:Union[str,List[str]]):
        model_params = ti.xcom_pull(key="return_value", task_ids=task_ids)
        return model_params