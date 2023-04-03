from axent.common.runtime import ModelTrainer
from axent.common.data import TrainTestDataProvider
from axent.common.data import FeaturesDataProvider
from airflow.models.taskinstance import TaskInstance

class ModelTrainBestTask:

    def __init__(self, data_provider:TrainTestDataProvider, features_provider: FeaturesDataProvider = None) -> None:
        self._trainer = ModelTrainer(data_provider=data_provider, features_provider=features_provider)

    def _get_best_result(self, ti:TaskInstance, model_results_key:str = 'model_hyperparameter_tuning_result') -> dict:
        model_results = ti.xcom_pull(key=model_results_key)
        best_result = None
        for result in model_results:
            if best_result is None or best_result['score'] < result['score']:
                best_result = result
        return best_result
    
    def _get_label(self, ti:TaskInstance):
        label = '-'.join([ti.dag_id,ti.task_id])
        return label

    def execute(self, ti:TaskInstance, model_results_key:str = 'model_hyperparameter_tuning_result'):
        label = self._get_label(ti)
        best_result = self._get_best_result(ti=ti, model_results_key=model_results_key)
        self._trainer.run(label=label, model_name=best_result['model_name'], params=best_result['params'], features_class=best_result['features_class'])
        ti.xcom_push(key='model_train_best_label', value=label)
        ti.xcom_push(key='model_train_best_result', value=best_result)