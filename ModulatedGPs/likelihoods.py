from typing import Optional, Any, Callable

import gpflow.utilities
import numpy as np
import tensorflow as tf
from gpflow import logdensities
from gpflow.base import TensorType, MeanAndVariance, Parameter
from gpflow.likelihoods import ScalarLikelihood
from gpflow.utilities.parameter_or_function import ParameterOrFunction

from ModulatedGPs.utils import inv_probit


class Gaussian(ScalarLikelihood):
    r"""
    The Gaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with constant
    variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    def __init__(
            self,
            variance=1e-0,
            D: int = None,
            **kwargs: Any,
    ) -> None:
        """
        :param variance: The noise variance;
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """
        super().__init__(**kwargs)

        if D is not None:
            variance = variance * np.ones((1, D))

        variance = np.maximum(variance - 1e-6, np.finfo(np.float64).eps)
        variance = variance + np.log(-np.expm1(-variance))
        self.variance: Optional[ParameterOrFunction] = Parameter(variance, transform=gpflow.utilities.positive())

    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return logdensities.gaussian(Y, F, self.variance)

    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        shape = tf.shape(F)
        return tf.broadcast_to(self.variance, shape)

    def _predict_mean_and_var(self, X: TensorType, Fmu: TensorType, Fvar: TensorType) -> MeanAndVariance:
        return tf.identity(Fmu), Fvar + self.variance

    def _predict_log_density(self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType) -> tf.Tensor:
        return tf.reduce_sum(logdensities.gaussian(Y, Fmu, Fvar + self.variance), axis=-1)

    # NOTE: Even though the SVGP elbo uses this, it's never called from the SMGP model
    def _variational_expectations(self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType) -> tf.Tensor:
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(self.variance) - 0.5 * (
                (Y - Fmu) ** 2 + Fvar) / self.variance


class Bernoulli(ScalarLikelihood):
    def __init__(
            self, invlink: Callable[[tf.Tensor], tf.Tensor] = inv_probit, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.invlink = invlink

    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return logdensities.bernoulli(Y, self.invlink(F))

    def _predict_mean_and_var(
            self, X: TensorType, Fmu: TensorType, Fvar: TensorType
    ) -> MeanAndVariance:
        if self.invlink is inv_probit:
            p = inv_probit(Fmu / tf.sqrt(1 + Fvar))
            return p, p - tf.square(p)
        else:
            # for other invlink, use quadrature
            return super()._predict_mean_and_var(X, Fmu, Fvar)

    def _predict_log_density(
            self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        p = self.predict_mean_and_var(X, Fmu, Fvar)[0]
        return tf.reduce_sum(logdensities.bernoulli(Y, p), axis=-1)

    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return self.invlink(F)

    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        p = self.conditional_mean(X, F)
        return p - (p ** 2)
