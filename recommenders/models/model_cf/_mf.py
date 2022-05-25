from typing import Callable, TypeVar, Union

import pandas as pd
import torch

MFType = TypeVar("MFType", bound="MF")


class MF(object):
    """Matrix Factorization"""

    def __init__(
        self,
        df: pd.DataFrame,
        latent_dim: int,
    ):
        """Inits MF

        Args:
            df (pd.DataFrame): User-Item Rating 행렬(m users x n items)
            latent_dim (int): Latent Matrix 차원
        """

        self.df = df
        self.latent_dim = latent_dim

        self.users, self.items = self.df.index, self.df.columns
        self.loss = []

    def _init_params(
        self,
    ):
        """Initialize Parameters"""

        self.P = torch.randn(len(self.users), self.latent_dim, requires_grad=True)
        self.Q = torch.randn(self.latent_dim, len(self.items), requires_grad=True)

        self.bias = self.df.mean().mean()
        self.user_bias = torch.zeros(len(self.users), 1, requires_grad=True)
        self.item_bias = torch.zeros(len(self.items), 1, requires_grad=True)

        self.trainable_params = [self.P, self.Q, self.user_bias, self.item_bias]

    def _update_prediction(
        self,
    ):
        """Update Predicted Rating Matrix"""

        pred = torch.mm(self.P, self.Q) + self.bias + self.user_bias + self.item_bias.view(1, -1)

        self.pred = pd.DataFrame(
            pred.detach().numpy(),
            index=self.users,
            columns=self.items,
        )

    def fit(
        self,
        optimizer: Callable,
        criterion: Callable,
        n_epochs: int,
        lr: float = 0.01,
        beta: float = 0.01,
    ) -> MFType:
        """Latent Matrix 학습

        Args:
            optimizer (Callable): Optimizer
            criterion (Callable): Criterion(Loss)
            n_epochs (int): Number of epochs
            lr (float, optional): Learning rate. Defaults to 0.01.
            beta (float, optional): Regularization rate. Defaults to 0.01.

        Returns:
            MFType: Fitted Model
        """

        self.optimizer = optimizer(self.trainable_params, lr=lr)
        self.criterion = criterion()

        y_true = torch.from_numpy(self.df.to_numpy()).float()
        mask = ~torch.isnan(y_true)
        pos_mask = mask.nonzero(as_tuple=True)
        pos_user_mask, pos_item_mask = pos_mask[0], pos_mask[1]

        for e in range(n_epochs):
            y_pred = (
                torch.mm(self.P, self.Q)[mask]
                + self.bias
                + self.user_bias[pos_user_mask].view(-1)
                + self.item_bias[pos_item_mask].view(-1)
            )

            penalty = beta * (self.P.pow(2).sum() + self.Q.pow(2).sum())
            loss = self.criterion(y_true[mask], y_pred) + penalty

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.loss.append(loss.item())

        self._update_prediction()

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

        if user not in self.users:
            return 3.0

        if item not in self.items:
            return 3.0

        return self.pred.loc[user, item]
