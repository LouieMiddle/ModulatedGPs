import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import config, Module

from .broadcasting_lik import BroadcastingLikelihood
from .utils import reparameterize

float_type = config.default_float()
jitter_level = config.default_jitter()


# SND likely stands for Samples by N by D, where:
#
# Samples: The number of Monte Carlo samples used in the model.
# N: The number of data points.
# D: The dimensionality of the output.

class SGP(Module):
    """
    Scalable Gaussian process (SGP)
    X -> Xt = integrate(X) -> GP -> Y
    """

    def __init__(self, likelihood, pred_layer, num_samples=1, num_data=None):
        self.num_samples = num_samples
        self.num_data = num_data
        self.likelihood = BroadcastingLikelihood(likelihood)
        self.pred_layer = pred_layer

    def integrate(self, X, S=1):
        return tf.tile(X[None, :, :], [S, 1, 1]), None

    def propagate(self, Xt, full_cov=False):
        Fmean, Fvar = self.pred_layer.predict_y(Xt, full_cov=full_cov)
        return Fmean, Fvar

    def _build_predict(self, Xt, full_cov=False):
        Fmeans, Fvars = self.propagate(Xt, full_cov=full_cov)
        return Fmeans, Fvars

    def E_log_p_Y(self, Xt, Y):
        Fmean, Fvar = self._build_predict(Xt, full_cov=False)
        var_exp = self.likelihood.variational_expectations(Xt, Fmean, Fvar, Y)
        return tf.reduce_mean(tf.reduce_sum(var_exp, 2), 0)

    def _build_likelihood(self, X, Y):
        Xt = self.integrate(X, self.num_samples)[0]
        L = tf.reduce_mean(self.E_log_p_Y(Xt, Y))
        return L - self.pred_layer.KL() / self.num_data

    def predict_f(self, Xnew, S=1):
        Xnewt = self.integrate(Xnew, S)[0]
        return self._build_predict(Xnewt, full_cov=False)

    def predict_y(self, Xnew, S=1):
        Xnewt = self.integrate(Xnew, S)[0]
        Fmean, Fvar = self._build_predict(Xnewt, full_cov=False)
        return self.likelihood.predict_mean_and_var([], Fmean, Fvar)

    def predict_samples(self, Xnew, S=1):
        Fmean, Fvar = self.predict_f(Xnew, S)
        mean, var = self.likelihood.predict_mean_and_var([], Fmean, Fvar)
        z = tf.random_normal(tf.shape(Fmean), dtype=float_type)
        samples_y = reparameterize(mean, var, z)
        samples_f = reparameterize(Fmean, Fvar, z)
        return samples_y, samples_f

    def predict_density(self, Xnew, Ynew, S):
        Fmean, Var = self.predict_y(Xnew, S)
        l = self.likelihood.predict_density(Fmean, Var, Ynew)
        log_num_samples = tf.log(tf.cast(S, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

    def get_inducing_Z(self):
        return self.pred_layer.Z


class SMGP(SGP):
    '''
    Mixture of Gaussian processes, used for regression, density estimation, data association, etc
    '''

    def __init__(self, likelihood, pred_layer, assign_layer, K=3, num_samples=1, num_data=None):
        SGP.__init__(self, likelihood, pred_layer, num_samples, num_data)
        self.assign_layer = assign_layer
        self.K = K

    def propagate_logassign(self, Xt, full_cov=False):
        logassign_mean, logassign_var = self.assign_layer.predict_y(Xt, full_cov=full_cov)
        return logassign_mean, logassign_var

    def _build_predict_logassign(self, Xt, full_cov=False):
        logassign_mean, logassign_var = self.propagate_logassign(Xt, full_cov=full_cov)
        return logassign_mean, logassign_var

    def W_dist(self, Xt):
        logassign_mean, logassign_var = self._build_predict_logassign(Xt)
        z = tf.random.normal(tf.shape(logassign_mean), dtype=float_type)
        log_assign = reparameterize(logassign_mean, logassign_var, z)
        log_assign = tf.reshape(log_assign, [tf.shape(Xt)[0] * tf.shape(Xt)[1], self.K])
        W_dist = tfp.distributions.RelaxedOneHotCategorical(temperature=1e-2, logits=log_assign)
        return W_dist

    def E_log_p_Y(self, Xt, Y, W_SND):
        Fmean, Fvar = self._build_predict(Xt, full_cov=False)
        var_exp = self.likelihood.variational_expectations(Xt, Fmean, Fvar, Y)
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

    def predict_assign(self, Xnew, S=1):
        Xt = self.integrate(Xnew, S)[0]
        logassign_mean, logassign_var = self._build_predict_logassign(Xt)
        assign = tf.nn.softmax(tf.exp(tf.reduce_mean(logassign_mean, 0)))
        return assign

    def predict_samples(self, Xnew, S=1):
        Xt = self.integrate(Xnew, S)[0]
        W_dist = self.W_dist(Xt)
        W = W_dist.sample(1)[0, :, :]
        W_SND = tf.cast(tf.reshape(W, [S, tf.shape(Xt)[1], self.K]), dtype=float_type)
        Fmean, Fvar = self._build_predict(Xt, full_cov=False)
        mean, var = self.likelihood.predict_mean_and_var([], Fmean, Fvar)
        z = tf.random.normal(tf.shape(Fmean), dtype=float_type)
        samples_y = reparameterize(mean, var, z)
        samples_y = tf.reduce_sum(samples_y * W_SND, 2, keepdims=True)
        samples_f = reparameterize(Fmean, Fvar, z)
        samples_f = tf.reduce_sum(samples_f * W_SND, 2, keepdims=True)
        return samples_y, samples_f
