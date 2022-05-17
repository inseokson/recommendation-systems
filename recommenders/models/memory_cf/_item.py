from typing import Callable, TypeVar, Union

import pandas as pd

from ._base import BaseMemoryCF

ItemCFType = TypeVar("ItemCFType", bound="ItemCF")


class ItemCF(BaseMemoryCF):
    """Item-based Collaborative Filtering"""

    def __init__(
        self,
        fillna: float,
        func_similarity: Callable,
        top_n: int = None,
    ):
        """Inits ItemCF

        Args:
            fillna (float): 유사도를 구할 때 NA를 대신할 값
            func_similarity (Callable): 유사도 계산 함수 (Pairwise)
            top_n (int): 양의 정수가 주어질 경우 상위 `top_n`개의 유사도를 이용. Defaults to None.
        """

        super().__init__(
            fillna,
            func_similarity,
            top_n,
        )

    def fit(
        self,
        df: pd.DataFrame,
    ) -> ItemCFType:
        """Fit model

        Args:
            df (pd.DataFrame): User-Item Rating 행렬(m users x n items)

        Returns:
            ItemCFType: Fitted model
        """

        self._fit(df, True)

        return self

    def predict(
        self,
        user: Union[str, int, float],
        item: Union[str, int, float],
    ) -> float:
        """Predict using model

        Args:
            user (Union[str, int, float]): User ID
            item (Union[str, int, float]): Item ID

        Returns:
            float: 예측 점수
        """

        mask = self.mask.loc[user, :]
        ratings = self.df.loc[user, mask]
        weights = self.similarity.loc[mask, item]

        rating_pred = self._predict(mask, ratings, weights)

        return rating_pred
