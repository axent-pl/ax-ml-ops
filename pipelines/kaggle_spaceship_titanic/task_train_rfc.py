from .common import model_hyperparameter_tuning_task
from sklearn.ensemble import RandomForestClassifier
from .data import get_xy, load_and_transform_data

def run(task_name: str, n_trials: int = 100, scoring: str = 'accuracy', direction = 'maximize', *args, **kwargs):
    print(task_name)
    def get_model(trial):
        params = {
            'bootstrap': trial.suggest_categorical("bootstrap", [True, False]),
            'max_depth': trial.suggest_int("max_depth", 10, 100, step = 10),
            'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 12),
            'min_samples_split': trial.suggest_int("min_samples_split", 2, 12),
            "n_estimators": trial.suggest_int("n_estimators", 5, 2000, log = True),
            "random_state": 7
        }
        return RandomForestClassifier(**params)
    
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