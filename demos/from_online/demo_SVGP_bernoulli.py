import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.models import VGP
from matplotlib.ticker import MaxNLocator


# From https://towardsdatascience.com/variational-gaussian-process-what-to-do-when-things-are-not-gaussian-41197039f3d4

def optimise(model, maxiter=2000):
    """
    Perform parameter learning for model given training data.
    :param model: model to be optimised.
    :param data: training data.
    :param maxiter: max number of optimisation steps.
    """
    print(f'Optimising model {model}')
    opt = gpflow.optimizers.Scipy()
    gpflow.utilities.print_summary(model)

    objective_closure = model.training_loss_closure()

    print(f'After optimisation')
    try:
        opt_logs = opt.minimize(objective_closure,
                                model.trainable_variables,
                                options=dict(maxiter=maxiter))
        print(opt_logs)
    finally:
        gpflow.utilities.print_summary(model)


# Construct training data.
X = tf.convert_to_tensor([2, 4, 7, 9, 17, 19, 21], dtype=default_float())
X = tf.expand_dims(X, axis=1)

Y = tf.convert_to_tensor([1, 1, 1, 1, 0, 0, 0], dtype=default_float())
Y = tf.expand_dims(Y, axis=1)

# Create VGP model with squared exponential kernel and Bernoulli likelihood.
data = (X, Y)
kern = gpflow.kernels.SquaredExponential()
model = VGP(data=data, kernel=kern, likelihood=gpflow.likelihoods.Bernoulli())

# Parameter learning
optimise(model)

# Predictions.
predicted_f_mean, predicted_f_var = model.predict_f(X)
predicted_Y_mean, predicted_Y_var = model.predict_y(X)

# Plotting.
predicted_f_mean = predicted_f_mean.numpy().flatten()
predicted_f_var = predicted_f_var.numpy().flatten()
predicted_Y_mean = predicted_Y_mean.numpy().flatten()

X = X.numpy().flatten()
markersize = 8
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(X, predicted_f_mean, marker='x', markersize=markersize, color='black')
ax1.fill_between(X,
                 predicted_f_mean - 1.96 * np.sqrt(predicted_f_var),
                 predicted_f_mean + 1.96 * np.sqrt(predicted_f_var),
                 color='C0', alpha=0.2)

ax2.plot(X, predicted_Y_mean, marker='x', markersize=markersize, color='blue')
ax3.scatter(X, Y, marker='x', color='red', s=45)

ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_ylabel('f(x)', fontsize=18, color='black')
ax2.set_ylabel('g(x)', fontsize=18, color='blue')
ax3.set_ylabel('Y', fontsize=18, color='red')
plt.xlabel('X', fontsize=18)
plt.show()
