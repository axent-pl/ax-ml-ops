from ...common.data.features_data_provider import FeaturesDataProvider
from ...common.data.train_test_data_provider import TrainTestDataProvider
from ..runtime import ModelHyperparameterTuning
from airflow.models.taskinstance import TaskInstance

class ModelHyperparameterTuningTask:

    def __init__(self, data_provider:TrainTestDataProvider, features_provider:FeaturesDataProvider) -> None:
        self.tuner = ModelHyperparameterTuning(data_provider=data_provider, features_provider=features_provider)

    def execute(self, ti:TaskInstance, model_name: str, features_class: str, label: str = None, n_trials: int = 100, n_splits: int = 5, scoring: str = 'accuracy', direction = 'maximize', *args, **kwargs) -> None:
        best_params, best_value = self.tuner.run(model_name=model_name, features_class=features_class, label=label, n_trials=n_trials, n_splits=n_splits, scoring=scoring, direction=direction, *args, **kwargs)
        result = {
            "model_name": model_name,
            "features_class": features_class,
            "scoring": scoring,
            "score": best_value,
            "direction": direction,
            "params": best_params
        }
        ti.xcom_push(key='model_hyperparameter_tuning_result', value=result)