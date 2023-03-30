from .model_registry import ModelRegistry


class ModelBase:
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelRegistry().register(cls)
