from abc import abstractmethod
from .model_registry import ModelRegistry


class ModelBase:
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelRegistry().register(cls)

    def get_base_class(self):
        """Returns the base class of the model. We expect that each model is subclass of ModelBase classs and a base class.

        Raises:
            Exception: When the model is a subclass of more than one base class.

        Returns:
            class: Model base class.
        """
        super_classes = [ cls for cls in self.__class__.__bases__ if cls != ModelBase ]
        if len(super_classes) == 1:
            return super_classes[0]
        raise Exception(f"Multiple base classes found: {super_classes}")
    
    def get_base_module(self):
        return self.get_base_class().__module__.split('.')[0]

    @abstractmethod
    def get_serializer(self):
        pass
