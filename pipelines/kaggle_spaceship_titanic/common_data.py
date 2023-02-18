from abc import ABC, abstractmethod
from typing import List, Union
from pandas.core.frame import DataFrame
from numpy import ndarray

class TrainTestDataProvider(ABC):

    def __init__(self) -> None:
        self._loaded = False
        self._df: DataFrame = None
        self._train_query: str = None
        self._test_query:str = None
        self._x_columns:List[str] = None
        self._y_columns:Union[str,List[str]] = None

    def _get_df(self) -> DataFrame:
        if not self._loaded:
            self._load()
        return self._df

    @abstractmethod
    def _load(self) -> None:
        self._loaded = True
        pass

    def get_dataframe(self) -> DataFrame:
        return self._get_df()

    def get_x_columns(self) -> List[str]:
        return self._x_columns

    def get_y_columns(self) -> Union[str,List[str]]:
        return self._y_columns

    def get_x_train(self) -> ndarray:
        return self._get_df().query(self._train_query)[self.get_x_columns()].to_numpy()

    def get_y_train(self) -> ndarray:
        return self._get_df().query(self._train_query)[self.get_y_columns()].to_numpy()

    def get_x_test(self) -> ndarray:
        return self._get_df().query(self._test_query)[self.get_x_columns()].to_numpy()

    def get_y_test(self) -> ndarray:
        return self._get_df().query(self._test_query)[self.get_y_columns()].to_numpy()
