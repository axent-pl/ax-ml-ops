from .model_registry import ModelRegistry

class ModelFactory:

    def get(model_name: str, trial = None, params = None):
        if ModelRegistry().is_registered(model_name):
            if trial is not None:
                return ModelRegistry().get(model_name)(trial=trial)
            else:
                return ModelRegistry().get(model_name)(**params)
        raise NotImplementedError(f'Model class "{model_name}" not registered')