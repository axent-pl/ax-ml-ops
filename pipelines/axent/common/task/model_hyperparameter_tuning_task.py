from ...common.model.model_class import ModelClass
from ...common.data.features_data_provider import FeaturesDataProvider
from ...common.data.train_test_data_provider import TrainTestDataProvider
from ..runtime import ModelHyperParameterTuning

class ModelHyperParameterTuningTask:

    def __init__(self, data_provider:TrainTestDataProvider, features_provider:FeaturesDataProvider):
        self.tuner = ModelHyperParameterTuning(data_provider=data_provider, features_provider=features_provider)

    def execute(self, model_class: str, features_class: str, label: str = None, n_trials: int = 100, n_splits: int = 5, scoring: str = 'accuracy', direction = 'maximize', *args, **kwargs):
        best_params, best_value = self.tuner.run(model_class=ModelClass(model_class), features_class=features_class, label=label, n_trials=n_trials, n_splits=n_splits, scoring=scoring, direction=direction, *args, **kwargs)

        return {
            "model_class": model_class,
            "features_class": features_class,
            "scoring": scoring,
            "score": best_value,
            "params": best_params
        }