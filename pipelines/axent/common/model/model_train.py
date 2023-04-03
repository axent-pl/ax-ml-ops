from .model_base import ModelBase
from .model_factory import ModelFactory
from sklearn.model_selection import cross_validate, StratifiedKFold


class ModelTrain:

    def get_cv_scores(model_name: str, x, y, trial = None, params = None, n_splits:int = 5, scoring:str = 'accuracy',):
        model = ModelFactory.get(model_name, trial=trial, params=params)
        kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 7)
        return cross_validate(model, x, y, cv = kf, scoring = scoring, n_jobs = -1, return_train_score=True)
    
    def train(model_name: str, x, y, params = None, scoring:str = 'accuracy') -> ModelBase:
        model = ModelFactory.get(model_name, params=params)
        model.fit(x, y)
        return model