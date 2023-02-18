from enum import Enum
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ModelClass(Enum):
    CBC = 'CatBoostClassifier'
    RFC = 'RandomForestClassifier'
    SVC = 'SVC'

class ModelFactory:

    def _get_cbc(trial, params):
        _params = {}
        if trial:
            _params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 5000, step = 50),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log = True),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9, step = 0.05),
                "verbose": False,
                "random_state": 7
            }
        else:
            _params = params
        return CatBoostClassifier(**_params)

    def _get_rfc(trial, params):
        _params = {}
        if trial:
            _params = {
                'bootstrap': trial.suggest_categorical("bootstrap", [True, False]),
                'max_depth': trial.suggest_int("max_depth", 10, 100, step = 10),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 12),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 12),
                "n_estimators": trial.suggest_int("n_estimators", 5, 2000, log = True),
                "random_state": 7
            }
        else:
            _params = params
        return RandomForestClassifier(**_params)

    def _get_svc(trial, params):
        _params = {}
        if trial:
            _params = {
                "C": trial.suggest_float("C", 0.1, 1000, log = True),
                "gamma": trial.suggest_float("gamma", 0.0001, 1),
                "random_state": 7
            }
        else:
            _params = params
        return SVC(**_params)

    def get(model_class: ModelClass, trial = None, params = None):
        if model_class == ModelClass.CBC:
            return ModelFactory._get_cbc(trial, params)
        if model_class == ModelClass.RFC:
            return ModelFactory._get_rfc(trial, params)
        if model_class == ModelClass.SVC:
            return ModelFactory._get_svc(trial, params)
        
        raise Exception('Unsupported model class')