from .model_registry import ModelRegistry

class ModelFactory:

    def get(model_name: str, trial = None, params = None):
        if ModelRegistry().is_registered(model_name):
            return ModelRegistry().get(model_name)(trial=trial, params=params)
        raise NotImplementedError(f'Model class "{model_name}" not registered')