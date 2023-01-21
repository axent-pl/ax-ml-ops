import os

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import StratifiedKFold, cross_validate
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.svm import SVC

from common import get_xy, get_xy_cols, load_and_transform_data

if __name__ == '__main__':
    mlflc = MLflowCallback(
        metric_name="accuracy",
    )

    df_train, df_test = load_and_transform_data()
    X_train, y_train = get_xy(df_train)
    X_cols, y_col = get_xy_cols(df_train)

    @mlflc.track_in_mlflow()
    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 0.1, 1000, log = True),
            "gamma": trial.suggest_float("gamma", 0.0001, 1),
            "random_state": 7
        }
        model = SVC(**params)

        kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 7)
        scores = cross_validate(model, X_train, y_train, cv = kf, scoring = "accuracy", n_jobs = -1, return_train_score=True)
        
        mlflow.log_metric("cv_train_min_score", scores["train_score"].min())
        mlflow.log_metric("cv_train_mean_score", scores["train_score"].mean())
        mlflow.log_metric("cv_train_max", scores["train_score"].max())
        mlflow.log_metric("cv_test_min_score", scores["test_score"].min())
        mlflow.log_metric("cv_test_mean_score", scores["test_score"].mean())
        mlflow.log_metric("cv_test_max_score", scores["test_score"].max())

        # fit and log model
        # model.fit(X_train, y_train)
        # mlflow.log_metric("train_score", model.score(X_train, y_train))
        # mlflow.sklearn.log_model(model, 'model')

        return scores["test_score"].mean()
    
    study = optuna.create_study(
        study_name=os.path.dirname(__file__).split(os.sep)[-1]+'-SVC',
        load_if_exists = True,
        direction = "maximize",
        storage=f'postgresql+psycopg2://optuna:{os.environ.get("DB_OPTUNA_PASS")}@db/optuna'
    )
    study.optimize(objective, n_trials=20, callbacks=[mlflc])