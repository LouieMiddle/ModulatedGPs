from typing import Optional, Any

import gpflow.utilities
import numpy as np
import tensorflow as tf
from gpflow import logdensities
from gpflow.base import TensorType, MeanAndVariance, Parameter
from gpflow.likelihoods import ScalarLikelihood
from gpflow.utilities.parameter_or_function import ParameterOrFunction


class GaussianModified(ScalarLikelihood):
    def __init__(self, variance=1e-0, D: int = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if D is not None:
            variance = variance * np.ones((1, D))

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

    # NOTE: Even though the SVGP elbo uses this, the SVGP elbo is never called in the SMGP model
    # This has been changed to match the original SMGP
    def _variational_expectations(self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType) -> tf.Tensor:
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(self.variance) - 0.5 * (
                (Y - Fmu) ** 2 + Fvar) / self.variance
