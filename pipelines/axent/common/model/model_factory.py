from enum import Enum
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from .model_class import ModelClass

class ModelFactory:

    def _get_CBC(trial, params):
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

    def _get_RFC(trial, params):
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

    def _get_SVC(trial, params):
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

    def _get_LRC(trial, params):
        _params = {}
        if trial:
            _params = {
                "C": trial.suggest_float("C", 0.1, 100, log = True),
                "max_iter": trial.suggest_int("max_iter", 5, 500),
                "random_state": 7
            }
        else:
            _params = params
        return LogisticRegression(**_params)

    def get(model_class: ModelClass, trial = None, params = None):
        if hasattr(ModelFactory, f'_get_{model_class.name}'):
            return getattr(ModelFactory, f'_get_{model_class.name}')(trial, params)
        raise NotImplementedError(f'Model class {model_class.name} not implemented')


class ModelTrain:

    def get_cv_scores(model_class: ModelClass, x, y, trial = None, params = None, n_splits:int = 5, scoring:str = 'accuracy',):
        model = ModelFactory.get(model_class, trial=trial, params=params)
        kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 7)
        return cross_validate(model, x, y, cv = kf, scoring = scoring, n_jobs = -1, return_train_score=True)