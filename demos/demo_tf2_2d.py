import gpflow.kernels
import matplotlib.colors as mcolors
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans

from MixtureGPs.likelihoods import GaussianModified
from MixtureGPs.models import SMGP, SVGPModified
from utils.dataset_utils import load_toy_2d_data
from utils.training_utils import run_adam

print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

colors = [mcolors.TABLEAU_COLORS[key] for key in mcolors.TABLEAU_COLORS.keys()]

seed = 0
tf.random.set_seed(seed)
rng = np.random.default_rng(seed=seed)

N, Xtrain, Ytrain, Xtest = load_toy_2d_data(rng)

# Model configuration
num_iter = 2000  # Optimization iterations
lr = 0.005  # Learning rate for Adam opt
num_minibatch = 500  # Batch size for stochastic opt
num_samples = 25  # Number of MC samples
num_predict_samples = 100  # Number of prediction samples
num_data = Xtrain.shape[0]  # Training size
dimX = Xtrain.shape[1]  # Input dimensions
dimY = 1  # Output dimensions
num_ind = 25  # Inducing size for f
K = 3

input_dim = dimX
pred_kernel = gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=1.0)
assign_kernel = gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=1.0)
Z, Z_assign = kmeans(Xtrain, num_ind, seed=0)[0], kmeans(Xtrain, num_ind, seed=1)[0]

lik = GaussianModified(variance=0.5, D=K)

pred_layer = SVGPModified(kernel=pred_kernel, likelihood=lik, inducing_variable=Z, num_latent_gps=K,
                          whiten=True)
assign_layer = SVGPModified(kernel=assign_kernel, likelihood=lik, inducing_variable=Z_assign, num_latent_gps=K,
                            whiten=True)

# model definition
model = SMGP(likelihood=lik, pred_layer=pred_layer,
             assign_layer=assign_layer, K=K, num_samples=num_samples,
             num_data=num_data)

gpflow.utilities.print_summary(model)

dataset = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
dataset = dataset.shuffle(buffer_size=num_data, seed=seed)
dataset = dataset.batch(num_minibatch).repeat()
train_iter = iter(dataset)

iters, elbos = run_adam(model, num_iter, train_iter, lr, compile=True)

gpflow.utilities.print_summary(model)

n_batches = max(int(Xtrain.shape[0] / 500), 1)
Ss_y, Ss_f = [], []
for X_batch in np.array_split(Xtrain, n_batches):
    samples_y, samples_f = model.predict_samples(X_batch, S=num_predict_samples)
    Ss_y.append(samples_y)
    Ss_f.append(samples_f)
samples_y, samples_f = np.hstack(Ss_y), np.hstack(Ss_f)
mu_avg, fmu_avg = np.mean(samples_y, 0), np.mean(samples_f, 0)
samples_y_stack = np.reshape(samples_y, (num_predict_samples * Xtrain.shape[0], -1))
samples_f_stack = np.reshape(samples_f, (num_predict_samples * Xtrain.shape[0], -1))
Xt_tiled = np.tile(Xtrain, [num_predict_samples, 1])

# Plotting results
fig_3d, fig = plt.figure(figsize=(14, 8)), plt.figure(figsize=(14, 8))
ax_3d, ax = [], []
for i in range(1, 5):
    ax_3d.append(fig_3d.add_subplot(2, 2, i, projection='3d'))

for i in range(1, 6):
    ax.append(fig.add_subplot(2, 3, i))

ax_3d[0].scatter(Xtrain[:, 0], Xtrain[:, 1], Ytrain, s=1)
ax_3d[0].set_title("Raw Data")
ax_3d[0].set_xlabel('x1')
ax_3d[0].set_ylabel('x2')
ax_3d[0].set_zlabel('y')
ax_3d[0].grid()

ax_3d[1].scatter(Xt_tiled[:, 0:1], Xt_tiled[:, 1:2], samples_y_stack.flatten(), marker='+', alpha=0.01,
                 color=mcolors.TABLEAU_COLORS['tab:red'])
ax_3d[1].scatter(Xt_tiled[:, 0:1], Xt_tiled[:, 1:2], samples_f_stack.flatten(), marker='+', alpha=0.01,
                 color=mcolors.TABLEAU_COLORS['tab:blue'])
ax_3d[1].scatter(Xtrain[:, 0], Xtrain[:, 1], Ytrain, marker='x', color='black', alpha=0.1)
ax_3d[1].set_title("Mixture of GPs")
ax_3d[1].set_xlabel('x1')
ax_3d[1].set_ylabel('x2')
ax_3d[1].set_zlabel('y')
ax_3d[1].set_ylim(1.2 * min(Ytrain), 1.2 * max(Ytrain))
ax_3d[1].grid()

assign_ = model.predict_assign(Xtrain)
for i in range(K):
    ax_3d[2].scatter(Xtrain[:, 0], Xtrain[:, 1], assign_[:, i], color=colors[i], s=1)
ax_3d[2].set_title("Assignment Plot")
ax_3d[2].set_xlabel('x1')
ax_3d[2].set_ylabel('x2')
ax_3d[2].set_zlabel('y')
ax_3d[2].grid()

fmean, _ = model.predict_y(Xtrain)
fmean_ = np.mean(fmean, 0)
for i in range(K):
    ax_3d[3].scatter(Xtrain[:, 0], Xtrain[:, 1], fmean_[:, i], color=colors[i], s=1)
ax_3d[3].set_title("Prediction Plot")
ax_3d[3].set_xlabel('x1')
ax_3d[3].set_ylabel('x2')
ax_3d[3].set_zlabel('y')
ax_3d[3].grid()

ax[0].plot(iters, elbos, 'o-', ms=8, alpha=0.5)
ax[0].set_xlabel('Iterations')
ax[0].set_ylabel('ELBO')
ax[0].grid()

stumpsX_const_value = -0.25
stumpsY_const_value = 0.75

Xtest_stumpsX = np.c_[Xtest[:, 0], stumpsY_const_value * np.ones(len(Xtest[:, 0]))]
Xtest_stumpsY = np.c_[stumpsX_const_value * np.ones(len(Xtest[:, 1])), Xtest[:, 1]]

Xtests = [Xtest_stumpsX, Xtest_stumpsY]

for i in range(2):
    if i == 0:
        ax[i + 1].set_title("x2 Constant Value = " + str(stumpsY_const_value))
        ax[i + 1].set_xlabel('x1')
    elif i == 1:
        ax[i + 1].set_title("x1 Constant Value = " + str(stumpsX_const_value))
        ax[i + 1].set_xlabel('x2')

    assign_ = model.predict_assign(Xtests[i])
    ax[i + 1].plot(Xtests[i][:, i], assign_, 'o', markersize=1)
    ax[i + 1].set_ylabel('softmax(assignment)')
    ax[i + 1].grid()

for i in range(2):
    if i == 0:
        ax[i + 3].set_title("x2 Constant Value = " + str(stumpsY_const_value))
        ax[i + 3].set_xlabel('x1')
    elif i == 1:
        ax[i + 3].set_title("x1 Constant Value = " + str(stumpsX_const_value))
        ax[i + 3].set_xlabel('x2')

    fmean, fvar = model.predict_y(Xtests[i])
    fmean_, fvar_ = np.mean(fmean, 0), np.mean(fvar, 0)

    X_sorted = np.zeros_like(Xtests[i])
    sort_indices = np.argsort(Xtests[i][:, i])
    X_sorted[:, i] = Xtests[i][sort_indices, i]
    fmean_sorted = fmean_[sort_indices]
    fvar_sorted = fvar_[sort_indices]

    lb, ub = (fmean_sorted - 2 * fvar_sorted ** 0.5), (fmean_sorted + 2 * fvar_sorted ** 0.5)

    for k in range(K):
        ax[i + 3].plot(X_sorted[:, i], fmean_sorted[:, k], '-', alpha=1., color=colors[k])
        ax[i + 3].fill_between(X_sorted[:, i], lb[:, k], ub[:, k], alpha=0.3, color=colors[k])

    ax[i + 3].set_ylabel('Pred. of GP experts')
    ax[i + 3].grid()

plt.tight_layout()
plt.show()
fig_3d.savefig('../figs/demo_tf2_2d_1.png')
fig.savefig('../figs/demo_tf2_2d_2.png')
