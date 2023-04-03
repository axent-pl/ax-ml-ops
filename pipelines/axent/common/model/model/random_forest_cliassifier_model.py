import mlflow
from sklearn.ensemble import RandomForestClassifier
from ..model_base import ModelBase

class RandomForestClassifierModel(RandomForestClassifier, ModelBase):

    def __init__(self, **kwargs):
        if 'trial' in kwargs and kwargs['trial'] is not None:
            _params = {
                'bootstrap': kwargs['trial'].suggest_categorical("bootstrap", [True, False]),
                'max_depth': kwargs['trial'].suggest_int("max_depth", 10, 100, step = 10),
                'min_samples_leaf': kwargs['trial'].suggest_int("min_samples_leaf", 1, 12),
                'min_samples_split': kwargs['trial'].suggest_int("min_samples_split", 2, 12),
                "n_estimators": kwargs['trial'].suggest_int("n_estimators", 5, 2000, log = True),
                "random_state": 7
            }
            RandomForestClassifier.__init__(self, **_params)
        else:
            RandomForestClassifier.__init__(self, **kwargs)

    def get_serializer(self):
        return mlflow.sklearn
        