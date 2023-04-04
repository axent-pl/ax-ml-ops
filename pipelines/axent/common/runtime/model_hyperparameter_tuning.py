import os
from typing import List, Union
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from ..model import ModelTrain
from ..data import TrainTestDataProvider
from ..data import FeaturesDataProvider

class ModelHyperparameterTuning:

    def __init__(self, data_provider:TrainTestDataProvider, features_provider: FeaturesDataProvider = None) -> None:
        self._dp:TrainTestDataProvider = data_provider
        self._fp:FeaturesDataProvider = features_provider

    def _get_experiment_id(self, experiment_name:str) -> str:
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        return experiment_id

    def run(self, model_name: str, features_class: str, label: str = None, n_trials: int = 100, n_splits: int = 5, scoring: str = 'accuracy', direction = 'maximize', *args, **kwargs):
        """Runs the model hyperparameter tuning

        Args:
            model_name (str): Model name (model class name)
            features_class (str): Features class name
            label (str, optional): Label of the MLFlow run name and Optuna study name. Defaults to None (model_name will be used).
            n_trials (int, optional): Number of trials in Optuna study. Defaults to 100.
            n_splits (int, optional): Number of splits in CV scores. Defaults to 5.
            scoring (str, optional): Scoring method. Defaults to 'accuracy'.
            direction (str, optional): Optuna study optimization direction. Defaults to 'maximize'.

        Returns:
            _type_: _description_
        """
        experimend_id = self._get_experiment_id(experiment_name=label)

        def objective(trial:optuna.trial.Trial) -> float:
            with mlflow.start_run(experiment_id=experimend_id) as run:
                x_train = self._dp.get_x_train(self._fp.get_features(features_class=features_class))
                y_train = self._dp.get_y_train()
                scores = ModelTrain.get_cv_scores(model_name=model_name, x=x_train, y=y_train, trial=trial, n_splits=n_splits, scoring=scoring)
                mlflow.log_params(trial.params)
                mlflow.set_tag('scoring', scoring)
                mlflow.set_tag('direction', direction)
                mlflow.set_tag('model_name', model_name)
                mlflow.set_tag('features_class', features_class)
                mlflow.log_metric(f"train_min_{scoring}", scores["train_score"].min())
                mlflow.log_metric(f"train_mean_{scoring}", scores["train_score"].mean())
                mlflow.log_metric(f"train_max_{scoring}", scores["train_score"].max())
                mlflow.log_metric(f"test_min_{scoring}", scores["test_score"].min())
                mlflow.log_metric(f"test_mean_{scoring}", scores["test_score"].mean())
                mlflow.log_metric(f"test_max_{scoring}", scores["test_score"].max())
                return scores["test_score"].mean()

        study = optuna.create_study(
            study_name = '-'.join([label, model_name.lower(), features_class.lower()]),
            load_if_exists = True,
            direction = direction,
            # storage=os.environ.get('MLFLOW_BACKEND_STORE_URI')
            storage=f'postgresql+psycopg2://optuna:{os.environ.get("DB_OPTUNA_PASS")}@db/optuna'
        )

        trials = study.get_trials()
        completed_trials = len([ 1 for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE ])
        todo_trials = max(0, n_trials-completed_trials)
        if todo_trials:
            study.optimize(objective, n_trials=todo_trials)

        return study.best_params, study.best_value