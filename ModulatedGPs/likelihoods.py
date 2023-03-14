from typing import Optional, Any

import numpy as np
import tensorflow as tf
from gpflow import logdensities
from gpflow.base import TensorType, MeanAndVariance
from gpflow.likelihoods import ScalarLikelihood
from gpflow.utilities.parameter_or_function import ConstantOrFunction, ParameterOrFunction, \
    prepare_parameter_or_function


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
            variance: Optional[ConstantOrFunction] = None,
            D: int = None,
            **kwargs: Any,
    ) -> None:
        """
        :param variance: The noise variance;
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """
        super().__init__(**kwargs)

        if variance is None:
            variance = 1.0
        if D is not None:
            variance = variance * np.ones((1, D))
        self.variance: Optional[ParameterOrFunction] = prepare_parameter_or_function(variance)

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

    def _variational_expectations(self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType) -> tf.Tensor:
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(self.variance) - 0.5 * (
                    (Y - Fmu) ** 2 + Fvar) / self.variance
