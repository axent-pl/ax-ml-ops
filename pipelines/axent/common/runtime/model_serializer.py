
import mlflow
from axent.common.model import ModelBase


class ModelSerializer:

    def serialize(self, model:ModelBase, path:str, registered_model_name:str):
        model_base_class = model.get_base_class()
        model_class_module = model.get_base_module()
        model.__class__ = model_base_class
        getattr(mlflow, model_class_module).log_model(model, artifact_path=path, registered_model_name=registered_model_name)