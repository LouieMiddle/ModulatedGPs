import gpflow.kernels
import matplotlib.colors as mcolors
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans

from ModulatedGPs.likelihoods import Gaussian
from ModulatedGPs.models import SMGP

print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

colors = [mcolors.TABLEAU_COLORS[key] for key in mcolors.TABLEAU_COLORS.keys()]

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

N, Ns, lambda_ = 600, 100, 0.1

x_min = -6.0
x_max = 6.0
Xtrain = np.random.uniform(low=x_min, high=x_max, size=(N, 1))

Ytrain = np.where(Xtrain < 0.0, 1.0, 0.0)
outlier_indices = np.random.choice(N, size=int(N * lambda_), replace=False)
Ytrain[outlier_indices] = 1 - Ytrain[outlier_indices]

Xtest = np.linspace(x_min, x_max, Ns).reshape(Ns, 1)

# Model configuration
num_iter = 100  # Optimization iterations
lr = 0.005  # Learning rate for Adam opt
num_minibatch = N  # Batch size for stochastic opt
num_samples = 25  # Number of MC samples
num_predict_samples = 100  # Number of prediction samples
num_data = Xtrain.shape[0]  # Training size
dimX = Xtrain.shape[1]  # Input dimensions
dimY = 1  # Output dimensions
num_ind = 25  # Inducing size for f
K = 2

lik = Gaussian(D=K)

input_dim = dimX
pred_kernel = gpflow.kernels.SquaredExponential(variance=0.5, lengthscales=0.5)
assign_kernel = gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=1.0)
Z, Z_assign = kmeans(Xtrain, num_ind)[0], kmeans(Xtrain, num_ind)[0]

pred_layer = gpflow.models.SVGP(kernel=pred_kernel, likelihood=lik, inducing_variable=Z, num_latent_gps=K)
assign_layer = gpflow.models.SVGP(kernel=assign_kernel, likelihood=lik, inducing_variable=Z_assign, num_latent_gps=K)

# model definition
model = SMGP(likelihood=lik, pred_layer=pred_layer, assign_layer=assign_layer, K=K, num_samples=num_samples,
             num_data=num_data)

optimizer = tf.optimizers.Adam(lr)

print('{:>5s}'.format("iter") + '{:>24s}'.format("ELBO:"))
iters = []
elbos = []
for i in range(1, num_iter + 1):
    try:
        with tf.GradientTape() as tape:
            # Record gradients of the loss with respect to the trainable variables
            elbo = model._build_likelihood(Xtrain, Ytrain)
            loss_value = -elbo
            # gpflow.utilities.print_summary(model)
            gradients = tape.gradient(loss_value, model.trainable_variables)

        # Use the optimizer to apply the gradients to update the trainable variables
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if i % 100 == 0 or i == 0:
            print('{:>5d}'.format(i) + '{:>24.6f}'.format(elbo))
            iters.append(i)
            elbos.append(elbo)
    except KeyboardInterrupt as e:
        print("stopping training")
        break

samples_y, samples_f = model.predict_samples(Xtest, S=num_predict_samples)
mu_avg, fmu_avg = np.mean(samples_y, 0), np.mean(samples_f, 0)
samples_y_stack = np.reshape(samples_y, (num_predict_samples * Xtest.shape[0], -1))
samples_f_stack = np.reshape(samples_f, (num_predict_samples * Xtest.shape[0], -1))
Xt_tiled = np.tile(Xtest, [num_predict_samples, 1])

f, ax = plt.subplots(2, 2, figsize=(14, 8))

ax[0, 0].scatter(Xt_tiled.flatten(), samples_y_stack.flatten(), marker='+', alpha=0.01,
                 color=mcolors.TABLEAU_COLORS['tab:red'])
ax[0, 0].scatter(Xt_tiled.flatten(), samples_f_stack.flatten(), marker='+', alpha=0.01,
                 color=mcolors.TABLEAU_COLORS['tab:blue'])
ax[0, 0].scatter(Xtrain, Ytrain, marker='x', color='black', alpha=0.1)
ax[0, 0].set_title("Many GPs")
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel('y')
ax[0, 0].set_ylim(1.2 * min(Ytrain), 1.2 * max(Ytrain))
ax[0, 0].grid()

ax[0, 1].plot(iters, elbos, 'o-', ms=8, alpha=0.5)
ax[0, 1].set_xlabel('Iterations')
ax[0, 1].set_ylabel('ELBO')
ax[0, 1].grid()

assign_ = model.predict_assign(Xtrain)
ax[1, 0].plot(Xtrain, assign_, 'o')
ax[1, 0].set_xlabel('x')
ax[1, 0].set_ylabel('softmax(assignment)')
ax[1, 0].grid()

fmean_, fvar_ = np.mean(model.predict_y(Xtest), 0), np.mean(model.predict_y(Xtest), 0)
lb, ub = (fmean_ - 2 * fvar_ ** 0.5), (fmean_ + 2 * fvar_ ** 0.5)
I = np.argmax(assign_, 1)
for i in range(K):
    test = fmean_[:, :, i]
    ax[1, 1].plot(Xtest.flatten(), fmean_[:, :, i], '-', alpha=1., color=colors[i])
    ax[1, 1].fill_between(Xtest.flatten(), lb[:, i], ub[:, i], alpha=0.3, color=colors[i])
ax[1, 1].scatter(Xtrain, Ytrain, marker='x', color='black', alpha=0.5)
ax[1, 1].set_xlabel('x')
ax[1, 1].set_ylabel('Pred. of GP experts')
ax[1, 1].grid()

plt.tight_layout()
plt.savefig('figs/test_toy.png')
plt.show()
