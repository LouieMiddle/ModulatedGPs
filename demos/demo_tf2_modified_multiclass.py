import gpflow.kernels
import matplotlib.colors as mcolors
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans

from MixtureGPs.likelihoods import GaussianModified
from MixtureGPs.models import SVGPModified, SMGPModified
from utils.dataset_utils import load_toy_data_categorical
from utils.training_utils import run_adam

print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

colors = [mcolors.TABLEAU_COLORS[key] for key in mcolors.TABLEAU_COLORS.keys()]

seed = 0
tf.random.set_seed(seed)
rng = np.random.default_rng(seed=seed)

N, Xtrain, Ytrain, Xtest = load_toy_data_categorical(rng)

Xplot = rng.uniform(min(Xtrain[:, 0]) - 2, max(Xtrain[:, 0]) + 2, (200, 1))

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
K = 2

input_dim = dimX
pred_kernel = gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=1.0)
assign_kernel = gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=1.0)
Z, Z_assign = kmeans(Xtrain, num_ind, seed=0)[0], kmeans(Xtrain, num_ind, seed=1)[0]

inv_link = gpflow.likelihoods.RobustMax(num_classes=K)
lik = gpflow.likelihoods.MultiClass(num_classes=K, invlink=inv_link)
assign_lik = GaussianModified(variance=0.5, D=K)

pred_layer = SVGPModified(kernel=pred_kernel, likelihood=lik, inducing_variable=Z, num_latent_gps=K,
                          whiten=True)
assign_layer = SVGPModified(kernel=assign_kernel, likelihood=assign_lik, inducing_variable=Z_assign, num_latent_gps=K,
                            whiten=True)

# model definition
model = SMGPModified(likelihood=lik, assign_likelihood=assign_lik, pred_layer=pred_layer,
                     assign_layer=assign_layer, K=K, num_samples=num_samples,
                     num_data=num_data)

gpflow.utilities.print_summary(model)

dataset = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
dataset = dataset.shuffle(buffer_size=num_data, seed=seed)
dataset = dataset.batch(num_minibatch).repeat()
train_iter = iter(dataset)

iters, elbos = run_adam(model, num_iter, train_iter, lr, compile=True)

gpflow.utilities.print_summary(model)

n_batches = max(int(Xplot.shape[0] / 500), 1)
Ss_y, Ss_f = [], []
for X_batch in np.array_split(Xplot, n_batches):
    samples_y, samples_f = model.predict_samples(X_batch, S=num_predict_samples)
    Ss_y.append(samples_y)
    Ss_f.append(samples_f)
samples_y, samples_f = np.hstack(Ss_y), np.hstack(Ss_f)
mu_avg, fmu_avg = np.mean(samples_y, 0), np.mean(samples_f, 0)
samples_y_stack = np.reshape(samples_y, (num_predict_samples * Xplot.shape[0], -1))
samples_f_stack = np.reshape(samples_f, (num_predict_samples * Xplot.shape[0], -1))
Xt_tiled = np.tile(Xplot, [num_predict_samples, 1])

# Plotting results
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

fmean, fvar = model.predict_y(Xtest)
fmean_, fvar_ = np.mean(fmean, 0), np.mean(fvar, 0)
lb, ub = (fmean_ - 2 * fvar_ ** 0.5), (fmean_ + 2 * fvar_ ** 0.5)
I = np.argmax(assign_, 1)
for i in range(K):
    ax[1, 1].plot(Xtest.flatten(), fmean_[:, i], '-', alpha=1., color=colors[i])
    ax[1, 1].fill_between(Xtest.flatten(), lb[:, i], ub[:, i], alpha=0.3, color=colors[i])
ax[1, 1].scatter(Xtrain, Ytrain, marker='x', color='black', alpha=0.5)
ax[1, 1].set_xlabel('x')
ax[1, 1].set_ylabel('Pred. of GP experts')
ax[1, 1].grid()

plt.tight_layout()
plt.savefig('../figs/demo_tf2_modified_multiclass.png')
plt.show()
