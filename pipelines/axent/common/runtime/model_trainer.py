import mlflow
from axent.common.data import FeaturesDataProvider
from axent.common.data import TrainTestDataProvider
from axent.common.model import ModelTrain
from axent.common.model import ModelBase
from .model_serializer import ModelSerializer


class ModelTrainer:

    def __init__(self, data_provider:TrainTestDataProvider, features_provider: FeaturesDataProvider = None) -> None:
        self._dp:TrainTestDataProvider = data_provider
        self._fp:FeaturesDataProvider = features_provider

    def run(self, label:str, model_name:str, params:dict, features_class:str, scoring:str = 'accuracy'):
        x = self._dp.get_x_train(self._fp.get_features(features_class=features_class))
        y = self._dp.get_y_train()
        with mlflow.start_run(run_name=label) as run:
            model:ModelBase = ModelTrain.train(model_name=model_name, x=x, y=y, params=params, scoring=scoring)
            ModelSerializer().serialize(model=model, path=label, registered_model_name=label)
            mlflow.set_tag('model_name', model_name)
            mlflow.set_tag('features_class', features_class)
        return model