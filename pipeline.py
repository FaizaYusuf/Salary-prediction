import pandas as pd
import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union


class DropColumns(BaseEstimator, TransformerMixin):
    """
    this transformer drops the specified columns in a dataframe
    === parameters ===
    -> features = list of columns to drop
    """

    def __init__(self, *, features: list) -> None:
        """ this contains parameters to pass"""
        self.features = features

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        X = X.copy()
        return X.drop(columns=self.features)


class ReplaceQuestionMark(BaseEstimator, TransformerMixin):
    """"
       This transformer checks to see if a variable contains contains ? and then replace the label with mode of the variable
    """

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        X = X.copy()
        columns = X.select_dtypes(include="O").columns
        for i in columns:
            get_mode = X[i].mode()[0]
            X[i] = np.where(X[i].str.contains("\?"), get_mode, X[i])
        return X


class Discretize(BaseEstimator, TransformerMixin):
    """
       this transformer accepts four parameters and return the decritized variable
       ==== parameters ====
        -> data = the dataframe
        -> variabe = the column to be descritize
        -> bins = number of bins to used
        -> labels = label to used
    """

    def __init__(self, *, variable: str, bins: int, labels: list[Union[str, int]]) -> None:
        """ this contains parameters to pass"""
        self.variable = variable
        self.bins = bins
        self.labels = labels

    def discetize_var(self, *, data: pd.DataFrame) -> pd.DataFrame:
        data[f"{self.variable}_binned"] = pd.qcut(
            x=data[self.variable], q=self.bins, labels=self.labels, duplicates="drop"
        )

        # dropping the old column
        data.drop(columns=[self.variable], inplace=True)

        return data

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        X = X.copy()
        return self.discetize_var(data=X)


class ReplaceValues(BaseEstimator, TransformerMixin):
    """
        this transformer accepts three parameters and return the decritized variable
        ==== parameters ====
        -> variabe = the column to be replace
        -> search = the label(s) to search in a list
        -> replace = label to repace
        """

    def __init__(self, *, variable: str, search: list, replace: str) -> None:
        """ this contains parameters to pass"""
        self.variable = variable
        self.search = search
        self.replace = replace

    def replace_values(self, *, data: pd.DataFrame) -> pd.DataFrame:
        """ This is used to replace the labes"""
        data[self.variable] = np.where(data[self.variable].isin(self.search), self.replace, data[self.variable])
        return data

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        X = X.copy()
        return self.replace_values(data=X)


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """This transformer is used to apply One-Hot
    Encoding to the variables."""

    def __init__(self) -> None:
        pass

    def _one_hot_encode(self, *, X: pd.DataFrame) -> pd.DataFrame:
        """This is used to one-hot encode the categorical variables."""
        data = pd.get_dummies(X)
        return data

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        X = X.copy()
        X = self._one_hot_encode(X=X)
        return X
