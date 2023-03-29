from sklearn.ensemble import RandomForestClassifier
from ...model.model_base import ModelBase

class RandomForestClassifierModel(ModelBase):

    def initialize(self,trial,params) -> RandomForestClassifier:
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