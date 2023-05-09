import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import inherit_check_shapes
from gpflow import config, covariances, default_jitter, posteriors
from gpflow.base import TensorType, MeanAndVariance, Module
from gpflow.conditionals import base_conditional
from gpflow.likelihoods import Likelihood
from gpflow.models import SVGP, GPModel
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin, Data
from gpflow.posteriors import IndependentPosterior

from .broadcasting_lik import BroadcastingLikelihood
from .utils import reparameterize

float_type = config.default_float()
jitter_level = config.default_jitter()


# S: The number of Monte Carlo samples used in the model.
# N: The number of data points.
# D: The dimensionality of the output.
class SGP(Module, ExternalDataTrainingLossMixin):
    """
    Scalable Gaussian process (SGP)
    X -> Xt = integrate(X) -> GP -> Y
    """

    def __init__(self, likelihood: Likelihood, pred_layer: GPModel, num_samples=1, num_data=None):
        self.num_samples = num_samples
        self.num_data = num_data
        self.likelihood: BroadcastingLikelihood = BroadcastingLikelihood(likelihood)
        self.pred_layer: GPModel = pred_layer

    def integrate(self, X, S=1):
        return tf.tile(X[None, :, :], [S, 1, 1]), None

    def predict_y(self, Xnew, S=1):
        Xnewt = self.integrate(Xnew, S)[0]
        Fmean, Fvar = self.pred_layer.predict_f(Xnewt, full_cov=False)
        return self.likelihood.predict_mean_and_var([], Fmean, Fvar)


class SMGP(SGP):
    '''
    Mixture of Gaussian processes, used for regression, density estimation, data association, etc
    '''

    def __init__(self, likelihood: Likelihood, pred_layer: GPModel,
                 assign_layer: GPModel, K=3, num_samples=1, num_data=None):
        SGP.__init__(self, likelihood, pred_layer, num_samples, num_data)
        self.assign_layer: GPModel = assign_layer
        self.K = K

    def W_dist(self, Xt):
        logassign_mean, logassign_var = self.assign_layer.predict_f(Xt, full_cov=False)
        z = tf.random.normal(tf.shape(logassign_mean), dtype=float_type)
        log_assign = reparameterize(logassign_mean, logassign_var, z)
        log_assign = tf.reshape(log_assign, [tf.shape(Xt)[0] * tf.shape(Xt)[1], self.K])
        W_dist = tfp.distributions.RelaxedOneHotCategorical(temperature=1e-2, logits=log_assign)
        return W_dist

    def E_log_p_Y(self, Xt, Y, W_SND):
        Fmean, Fvar = self.pred_layer.predict_f(Xt, full_cov=False)
        var_exp = self.likelihood.variational_expectations(Xt, Fmean, Fvar, tf.cast(Y, dtype=tf.float64))
        var_exp *= tf.cast(W_SND, dtype=float_type)
        return tf.reduce_logsumexp(tf.reduce_sum(var_exp, 2), 0) - np.log(self.num_samples)

    def _build_likelihood(self, X, Y):
        Xt = self.integrate(X, self.num_samples)[0]
        # sample from q(w)
        W_dist = self.W_dist(Xt)
        W = W_dist.sample(1)[0, :, :]
        W_SND = tf.reshape(W, [self.num_samples, tf.shape(Xt)[1], self.K])
        # Expectation of lik
        L = tf.reduce_mean(self.E_log_p_Y(Xt, Y, W_SND))
        # ELBO
        # KL Divergence
        return L - (self.pred_layer.prior_kl() + self.assign_layer.prior_kl()) / self.num_data

    def _training_loss(self, data: Data) -> tf.Tensor:
        X, Y = data
        return -self._build_likelihood(X, Y)

    def predict_assign(self, Xnew, S=1):
        Xt = self.integrate(Xnew, S)[0]
        logassign_mean, logassign_var = self.assign_layer.predict_f(Xt)
        assign = tf.nn.softmax(tf.reduce_mean(logassign_mean, 0))
        return assign

    def predict_samples(self, Xnew, S=1):
        Xt = self.integrate(Xnew, S)[0]
        W_dist = self.W_dist(Xt)
        W = W_dist.sample(1)[0, :, :]
        W_SND = tf.cast(tf.reshape(W, [S, tf.shape(Xt)[1], self.K]), dtype=float_type)
        Fmean, Fvar = self.pred_layer.predict_f(Xt, full_cov=False)
        mean, var = self.likelihood.predict_mean_and_var([], Fmean, Fvar)
        z = tf.random.normal(tf.shape(Fmean), dtype=float_type)
        samples_y = reparameterize(mean, var, z)
        samples_y = tf.reduce_sum(samples_y * W_SND, 2, keepdims=True)
        samples_f = reparameterize(Fmean, Fvar, z)
        samples_f = tf.reduce_sum(samples_f * W_SND, 2, keepdims=True)
        return samples_y, samples_f


class SMGPModified(SMGP):
    def __init__(self, likelihood: Likelihood, assign_likelihood: Likelihood, pred_layer: GPModel,
                 assign_layer: GPModel, K=3, num_samples=1, num_data=None):
        SMGP.__init__(self, likelihood, pred_layer, assign_layer, K, num_samples, num_data)
        self.assign_likelihood = BroadcastingLikelihood(assign_likelihood)

    def E_log_p_Y(self, Xt, Y, W_SND):
        Fmean, Fvar = self.assign_layer.predict_f(Xt, full_cov=False)
        var_exp = self.assign_likelihood.variational_expectations(Xt, Fmean, Fvar, tf.cast(Y, dtype=tf.float64))
        var_exp *= tf.cast(W_SND, dtype=float_type)
        E_log_p_A = tf.reduce_sum(var_exp, 2) - np.log(self.num_samples)

        Fmean, Fvar = self.pred_layer.predict_f(Xt, full_cov=False)
        var_exp = self.likelihood.variational_expectations(Xt, Fmean, Fvar, tf.cast(Y, dtype=tf.float64))
        var_exp *= tf.cast(W_SND, dtype=float_type)
        E_log_p_y = tf.reduce_sum(var_exp, 2) - np.log(self.num_samples)

        return tf.reduce_logsumexp(E_log_p_A, 0) + tf.reduce_logsumexp(E_log_p_y, 0)


class IndependentPosteriorSingleOutputModified(IndependentPosterior):
    # could almost be the same as IndependentPosteriorMultiOutput ...
    @inherit_check_shapes
    def _conditional_fused(
            self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following line:
        Knn = self.kernel(Xnew, full_cov=full_cov)

        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]

        # Same as IndependentPosteriorSingleOutput except for following line:
        # This change was made to match the original ModulatedGPs
        Kmn = self.kernel.K(self.X_data.Z, Xnew)

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)


class SVGPModified(SVGP):
    def posterior(
            self,
            precompute_cache: posteriors.PrecomputeCacheType = posteriors.PrecomputeCacheType.TENSOR,
    ) -> posteriors.BasePosterior:
        return IndependentPosteriorSingleOutputModified(
            self.kernel,
            self.inducing_variable,
            self.q_mu,
            self.q_sqrt,
            whiten=self.whiten,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )
