from sklearn.svm import SVC
from ..model_base import ModelBase

class SVCModel(SVC, ModelBase):

    def __init__(self, **kwargs):
        if 'trial' in kwargs:
            _params = {
                "C": kwargs['trial'].suggest_float("C", 0.1, 1000, log = True),
                "gamma": kwargs['trial'].suggest_float("gamma", 0.0001, 1),
                "random_state": 7
            }
            SVC.__init__(self, **_params)
        else:
            SVC.__init__(self, **kwargs)
        