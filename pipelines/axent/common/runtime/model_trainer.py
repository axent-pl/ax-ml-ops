import os
import mlflow
import numpy as np
from axent.common.data import FeaturesDataProvider
from axent.common.data import TrainTestDataProvider
from axent.common.model import ModelTrain
from axent.common.model import ModelBase
from .model_serializer import ModelSerializer
from tempfile import TemporaryDirectory

class ModelTrainer:

    def __init__(self, data_provider:TrainTestDataProvider, features_provider: FeaturesDataProvider = None) -> None:
        self._dp:TrainTestDataProvider = data_provider
        self._fp:FeaturesDataProvider = features_provider

    def _log_model(self, model:ModelBase, path:str, registered_model_name:str) -> None:
        ModelSerializer().serialize(model=model, path=path, registered_model_name=registered_model_name)

    def _log_predictions(self, preds:np.ndarray, path:str) -> None:
        with TemporaryDirectory() as tmpdirname:
            tmpfilepath = os.path.join(tmpdirname, 'y_test.npy')
            with open(tmpfilepath, 'wb') as f:
                np.save(f, preds)
                mlflow.log_artifact(tmpfilepath, artifact_path=path)

    def _predict(self, model:ModelBase, features_class:str) -> np.ndarray:
        x_test = self._dp.get_x_test(self._fp.get_features(features_class=features_class))
        return model.predict(x_test)

    def run(self, label:str, model_name:str, params:dict, features_class:str, scoring:str = 'accuracy', log_model:bool = True, predict:bool = True) -> ModelBase:
        x = self._dp.get_x_train(self._fp.get_features(features_class=features_class))
        y = self._dp.get_y_train()
        with mlflow.start_run(run_name=label) as run:
            model:ModelBase = ModelTrain.train(model_name=model_name, x=x, y=y, params=params, scoring=scoring)
            if predict:
                y_test = self._predict(model=model, features_class=features_class)
                self._log_predictions(preds=y_test, path=label)
            if log_model:
                self._log_model(model=model, path=label, registered_model_name=label)
            mlflow.set_tag('model_name', model_name)
            mlflow.set_tag('features_class', features_class)
        return model