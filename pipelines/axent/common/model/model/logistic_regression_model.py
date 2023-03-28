from sklearn.linear_model import LogisticRegression
from ...model.model_base import ModelBase

class LogisticRegressionModel(ModelBase):

    def initialize(self, trial, params) -> LogisticRegression:
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