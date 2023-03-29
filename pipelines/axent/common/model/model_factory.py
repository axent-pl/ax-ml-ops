from .model_base import ModelBase


class ModelFactory:

    def get(model_name: str, trial = None, params = None):
        if ModelBase.is_registered(model_name):
            return ModelBase.get_model(model_name).initialize(trial=trial, params=params)
        raise NotImplementedError(f'Model class {model_name} not implemented')