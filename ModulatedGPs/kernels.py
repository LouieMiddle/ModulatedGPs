from typing import Any

import tensorflow as tf
from check_shapes import inherit_check_shapes, check_shapes
from gpflow.base import TensorType, Parameter
from gpflow.kernels import IsotropicStationary
from gpflow.utilities import positive


class SquaredExponentialModified(IsotropicStationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is

        k(r) = σ² exp{-½ r²}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ²  is the variance parameter

    Functions drawn from a GP with this kernel are infinitely differentiable!
    """

    @check_shapes(
        "variance: []",
        "lengthscales: [broadcast n_active_dims]",
    )
    def __init__(
            self, variance: TensorType = 1.0, lengthscales: TensorType = 1.0, **kwargs: Any
    ) -> None:
        """
        :param variance: the (initial) value for the variance parameter.
        :param lengthscales: the (initial) value for the lengthscale
            parameter(s), to induce ARD behaviour this must be initialised as
            an array the same length as the number of active dimensions
            e.g. [1., 1., 1.]. If only a single value is passed, this value
            is used as the lengthscale of each dimension.
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")

        super().__init__(**kwargs)
        self.variance = Parameter(variance, transform=positive())
        self.lengthscales = Parameter(lengthscales, transform=positive(), trainable=False)
        self._validate_ard_active_dims(self.lengthscales)

    @inherit_check_shapes
    def K_r2(self, r2: TensorType) -> tf.Tensor:
        return self.variance * tf.exp(-0.5 * r2)
