from ..data import TrainTestDataProvider
from ..data import FeaturesDataProvider
from airflow.models.taskinstance import TaskInstance

class ModelTrainBestTask:

    def __init__(self, data_provider:TrainTestDataProvider, features_provider: FeaturesDataProvider = None) -> None:
        self._dp:TrainTestDataProvider = data_provider
        self._fp:FeaturesDataProvider = features_provider

    def _get_best_result(self, ti:TaskInstance, model_results_key:str = 'model_hyperparameter_tuning_result') -> dict:
        model_results = ti.xcom_pull(key=model_results_key)
        best_result = None
        for result in model_results:
            if best_result is None or best_result['score'] < result['score']:
                best_result = result
        return best_result

    def execute(self, ti:TaskInstance, model_results_key:str = 'model_hyperparameter_tuning_result'):
        best_result = self._get_best_result(ti=ti, model_results_key=model_results_key)
        ti.xcom_push(key='model_train_best_result', value=best_result)