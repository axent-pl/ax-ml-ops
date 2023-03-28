
from ..data import TrainTestDataProvider
from ..data import FeaturesDataProvider
from airflow.models.taskinstance import TaskInstance

class ModelTrainBestTask:

    def __init__(self, data_provider:TrainTestDataProvider, features_provider: FeaturesDataProvider = None) -> None:
        self._dp:TrainTestDataProvider = data_provider
        self._fp:FeaturesDataProvider = features_provider

    def execute(self, ti:TaskInstance):
        model_params = ti.xcom_pull(key='model_hyperparameter_tuning_result')
        return model_params