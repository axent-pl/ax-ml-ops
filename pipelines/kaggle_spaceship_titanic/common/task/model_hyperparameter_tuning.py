import os
from typing import List, Union
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from ...common_model import ModelClass, ModelTrain
from ...common_data import TrainTestDataProvider

class ModelHyperParameterTuning:

    def __init__(self, data_provider:TrainTestDataProvider) -> None:
        self._dp:TrainTestDataProvider = data_provider

    def run(
        self,
        model_class: ModelClass,
        x_columns: List[str] = None,
        y_columns: Union[str,List[str]] = None,
        label: str = None,
        n_trials: int = 100,
        n_splits: int = 5,
        scoring: str = 'accuracy',
        direction = 'maximize',
        *args,
        **kwargs
    ):
        mlflc = MLflowCallback(metric_name=scoring)

        @mlflc.track_in_mlflow()
        def objective(trial):
            x_train = self._dp.get_x_train(x_columns)
            y_train = self._dp.get_y_train(y_columns)
            scores = ModelTrain.get_cv_scores(model_class=model_class, x=x_train, y=y_train, trial=trial, n_splits=n_splits, scoring=scoring)
            mlflow.log_metric(f"train_min_{scoring}", scores["train_score"].min())
            mlflow.log_metric(f"train_mean_{scoring}", scores["train_score"].mean())
            mlflow.log_metric(f"train_max_{scoring}", scores["train_score"].max())
            mlflow.log_metric(f"test_min_{scoring}", scores["test_score"].min())
            mlflow.log_metric(f"test_mean_{scoring}", scores["test_score"].mean())
            mlflow.log_metric(f"test_max_{scoring}", scores["test_score"].max())
            return scores["test_score"].mean()

        study = optuna.create_study(
            study_name = label or model_class.value.lower(),
            load_if_exists = True,
            direction = direction,
            # storage=os.environ.get('MLFLOW_BACKEND_STORE_URI')
            storage=f'postgresql+psycopg2://optuna:{os.environ.get("DB_OPTUNA_PASS")}@db/optuna'
        )

        study.optimize(objective, n_trials=n_trials, callbacks=[mlflc])

        return study.best_params