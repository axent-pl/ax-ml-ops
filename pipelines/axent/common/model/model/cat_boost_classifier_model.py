from catboost import CatBoostClassifier
from ..model_base import ModelBase


class CatBoostClassifierModel(CatBoostClassifier,ModelBase):
    
    def __init__(self, **kwargs):
        if 'trial' in kwargs and kwargs['trial'] is not None:
            _params = {
                "n_estimators": kwargs['trial'].suggest_int("n_estimators", 100, 5000, step = 50),
                "learning_rate": kwargs['trial'].suggest_float("learning_rate", 1e-4, 0.3, log = True),
                "max_depth": kwargs['trial'].suggest_int("max_depth", 3, 9),
                "subsample": kwargs['trial'].suggest_float("subsample", 0.5, 0.9, step = 0.05),
                "verbose": False,
                "random_state": 7
            }
            CatBoostClassifier.__init__(self, **_params)
        else:
            CatBoostClassifier.__init__(self, **kwargs)