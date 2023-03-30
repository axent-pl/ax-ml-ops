from sklearn.linear_model import LogisticRegression
from ..model_base import ModelBase

class LogisticRegressionModel(LogisticRegression, ModelBase):

    def __init__(self, **kwargs):
        if 'trial' in kwargs:
            _params = {
                "C": kwargs['trial'].suggest_float("C", 0.1, 100, log = True),
                "max_iter": kwargs['trial'].suggest_int("max_iter", 5, 500),
                "random_state": 7
            }
            LogisticRegression.__init__(self, **_params)
        else:
            LogisticRegression.__init__(self, **kwargs)