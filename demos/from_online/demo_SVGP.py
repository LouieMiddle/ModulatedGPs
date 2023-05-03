import gpflow
import numpy as np
import tensorflow as tf

# From https://towardsdatascience.com/sparse-and-variational-gaussian-process-what-to-do-when-data-is-large-2d3959f430e7

# Define input data
X = tf.random.uniform(shape=[10, 1], minval=-2.0, maxval=2.0, dtype=tf.float64)
Y = tf.sin(X) + 0.2 * tf.random.normal(shape=[10, 1], dtype=tf.float64)

# Define kernel and inducing points
kernel = gpflow.kernels.SquaredExponential()
inducing_points = tf.random.uniform(shape=[5, 1], minval=-2.0, maxval=2.0)

# Define SVGP model
vgp_model = gpflow.models.SVGP(kernel=kernel, inducing_variable=inducing_points,
                               likelihood=gpflow.likelihoods.Gaussian(), num_data=X.shape[0])

# Optimize model
optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(vgp_model.training_loss_closure((X, Y)), variables=vgp_model.trainable_variables)

# Generate posterior samples
num_samples = 5
X_test = np.linspace(-3.0, 3.0, num=100, dtype=np.float64)[:, None]
f_samples = vgp_model.predict_f_samples(X_test, num_samples=num_samples)

# Plot samples
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(X, Y, "kx", mew=2)
plt.plot(X_test, f_samples[0, :, :], "b", linewidth=2, alpha=0.3)
plt.plot(X_test, f_samples[1, :, :], "g", linewidth=2, alpha=0.3)
plt.plot(X_test, f_samples[2, :, :], "r", linewidth=2, alpha=0.3)
plt.plot(X_test, f_samples[3, :, :], "y", linewidth=2, alpha=0.3)
plt.plot(X_test, f_samples[4, :, :], "m", linewidth=2, alpha=0.3)
plt.show()
