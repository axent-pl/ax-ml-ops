from abc import abstractmethod


class ModelBase:
    models = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.models[cls.__name__] = cls

    def is_registered(model_name: str) -> bool:
        return model_name in ModelBase.models
    
    def get_model(model_name: str):
        return ModelBase.models[model_name]()

    @abstractmethod
    def initialize(self, trial, params):
        pass