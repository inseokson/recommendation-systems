from typing import Callable

import pandas as pd


class BaseMemoryCF(object):
    """Base class for memory-based collaborative filtering"""

    def __init__(
        self,
        fillna: float,
        func_similarity: Callable,
        top_n: int = None,
    ):
        """Inits BaseMemoryCF

        Args:
            fillna (float): 유사도를 구할 때 NA를 대신할 값
            func_similarity (Callable): 유사도 계산 함수 (Pairwise)
            top_n (int): 양의 정수가 주어질 경우 상위 `top_n`개의 유사도를 이용. Defaults to None.
        """

        self.fillna = fillna
        self.func_similarity = func_similarity
        self.top_n = top_n

    def _fit(
        self,
        df: pd.DataFrame,
        transpose: bool,
    ):
        """Fit model

        Args:
            df (pd.DataFrame): User-Item Rating 행렬(m users x n items)
            transpose (bool): 유사도 계산시 전치(transpose) 여부. 만약 False일 경우 행을 기준으로, True일 경우 열을 기준으로 유사도 계산
        """

        self.df = df
        self.mask = df.notna()

        _df = df.copy().fillna(self.fillna)

        if not transpose:
            self.similarity = pd.DataFrame(self.func_similarity(_df), index=df.index, columns=df.index)
        else:
            self.similarity = pd.DataFrame(self.func_similarity(_df.T), index=df.columns, columns=df.columns)

    def _predict(
        self,
        mask: pd.Series,
        ratings: pd.Series,
        weights: pd.Series,
    ) -> float:
        """Predict using model

        Args:
            mask (pd.Series): Not-NA 위치를 나타내는 Mask
            ratings (pd.Series): 각 비중에 해당되는 점수
            weights (pd.Series): 유사도 기반으로 계산된 비중

        Returns:
            float: 가중 평균 점수
        """

        if not any(mask):
            return 3.0

        if isinstance(self.top_n, int):
            idx = weights.nlargest(self.top_n + 1).index
            weights, ratings = weights.loc[idx], ratings.loc[idx]

        rating_pred = (ratings * weights / weights.sum()).sum()

        return rating_pred
