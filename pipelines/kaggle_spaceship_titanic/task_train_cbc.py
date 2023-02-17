from .common import model_hyperparameter_tuning_task
from catboost import CatBoostClassifier
from .data import get_xy, load_and_transform_data

def run(task_name: str, n_trials: int = 100, scoring: str = 'accuracy', direction = 'maximize', *args, **kwargs):
    print(task_name)
    def get_model(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 5000, step = 50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log = True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9, step = 0.05),
            "verbose": False,
            "random_state": 7
        }
        return CatBoostClassifier(**params)
    
    df_train, df_test = load_and_transform_data()
    X_train, y_train = get_xy(df_train)

    return model_hyperparameter_tuning_task(
        task_name = task_name,
        model_provider = get_model,
        x_train = X_train,
        y_train = y_train,
        n_trials = n_trials,
        scoring = scoring,
        direction = direction
    )