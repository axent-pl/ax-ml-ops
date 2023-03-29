from catboost import CatBoostClassifier
from ...model.model_base import ModelBase

class CatBoostClassifierModel(ModelBase):

    def initialize(self,trial, params) -> CatBoostClassifier:
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