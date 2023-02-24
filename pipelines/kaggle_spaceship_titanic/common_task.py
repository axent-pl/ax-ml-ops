import os
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from .common_model import ModelClass, ModelTrain
from .common_data import TrainTestDataProvider

def model_hyperparameter_tuning_task(
    task_name: str,
    model_class: ModelClass,
    data_provider: TrainTestDataProvider,
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
        x_train = data_provider.get_x_train()
        y_train = data_provider.get_y_train()
        scores = ModelTrain.get_cv_scores(model_class=model_class, x=x_train, y=y_train, trial=trial, n_splits=n_splits, scoring=scoring)
        mlflow.log_metric(f"train_min_{scoring}", scores["train_score"].min())
        mlflow.log_metric(f"train_mean_{scoring}", scores["train_score"].mean())
        mlflow.log_metric(f"train_max_{scoring}", scores["train_score"].max())
        mlflow.log_metric(f"test_min_{scoring}", scores["test_score"].min())
        mlflow.log_metric(f"test_mean_{scoring}", scores["test_score"].mean())
        mlflow.log_metric(f"test_max_{scoring}", scores["test_score"].max())
        return scores["test_score"].mean()

    study = optuna.create_study(
        study_name=task_name,
        load_if_exists = True,
        direction = direction,
        storage=f'postgresql+psycopg2://optuna:{os.environ.get("DB_OPTUNA_PASS")}@db/optuna'
    )

    study.optimize(objective, n_trials=n_trials, callbacks=[mlflc])

    return study.best_params