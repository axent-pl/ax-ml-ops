from .common import model_hyperparameter_tuning_task
from sklearn.svm import SVC
from .data import get_xy, load_and_transform_data

def run(task_name: str, n_trials: int = 100, scoring: str = 'accuracy', direction = 'maximize', *args, **kwargs):
    print(task_name)
    def get_model(trial):
        params = {
            "C": trial.suggest_float("C", 0.1, 1000, log = True),
            "gamma": trial.suggest_float("gamma", 0.0001, 1),
            "random_state": 7
        }
        return SVC(**params)
    
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