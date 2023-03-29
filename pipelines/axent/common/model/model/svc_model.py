from sklearn.svm import SVC
from ...model.model_base import ModelBase

class SVCModel(ModelBase):

    def initialize(self, trial, params) -> SVC:
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