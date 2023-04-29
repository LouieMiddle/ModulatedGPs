import os

import gpflow.kernels
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans
from sklearn.model_selection import train_test_split

from ModulatedGPs.likelihoods import GaussianModified
from ModulatedGPs.models import SMGP, SVGPModified


# TODO: Need to group boundaries and non boundaries together
#  Or do multiclass classifiers

def filter_by_pitch_x_pitch_y(data):
    data = data[(data['pitchX'] >= -2) & (data['pitchX'] <= 2)]
    data = data[(data['pitchY'] >= 0) & (data['pitchY'] <= 14)]
    return data


def load_csv_data_mipl():
    csv_path = os.path.join("./", "mensIPLHawkeyeStats.csv")
    return pd.read_csv(csv_path)


print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

colors = [mcolors.TABLEAU_COLORS[key] for key in mcolors.TABLEAU_COLORS.keys()]

seed = 0
tf.random.set_seed(seed)
rng = np.random.default_rng(seed=seed)

mipl_csv = load_csv_data_mipl()
mipl_csv = filter_by_pitch_x_pitch_y(mipl_csv)

seam = ['FAST_SEAM', 'MEDIUM_SEAM', 'SEAM']
mipl_csv = mipl_csv[mipl_csv['batter'] == 'Jos Buttler']
mipl_csv = mipl_csv[mipl_csv['bowlingStyle'].isin(seam)]
mipl_csv = mipl_csv[mipl_csv['rightArmedBowl'] == True]

categorical_attributes = []
numerical_attributes = ['stumpsX', 'stumpsY']
# numerical_attributes = ['stumpsX', 'stumpsY', 'pitchX', 'pitchY']
all_columns = numerical_attributes + ['runs']

mipl_csv = mipl_csv[all_columns]
mipl_csv = mipl_csv[(mipl_csv['runs'] == 0) | (mipl_csv['runs'] == 6)]
mipl_csv = mipl_csv.tail(1000)

features = mipl_csv.drop(['runs'], axis=1)
targets = mipl_csv['runs']

name = 'JosButtler_RightArmSeam_'

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, targets, test_size=0.2)
Xtrain, Xtest, Ytrain, Ytest = Xtrain.to_numpy(), Xtest.to_numpy(), Ytrain.to_numpy(), Ytest.to_numpy()
Ytrain, Ytest = Ytrain.reshape((len(Ytrain), 1)), Ytest.reshape((len(Ytest), 1))

Xplot = rng.uniform([min(Xtrain[:, 0]) - 1, min(Xtrain[:, 1]) - 1], [max(Xtrain[:, 0]) + 1, max(Xtrain[:, 1]) + 1],
                    (200, 2))

# Model configuration
num_iter = 1000  # Optimization iterations
lr = 0.005  # Learning rate for Adam opt
num_minibatch = 500  # Batch size for stochastic opt
num_samples = 25  # Number of MC samples
num_predict_samples = 100  # Number of prediction samples
num_data = Xtrain.shape[0]  # Training size
dimX = Xtrain.shape[1]  # Input dimensions
dimY = 1  # Output dimensions
num_ind = 25  # Inducing size for f
K = 2

# bernoulli_lik = Bernoulli()
gaussian_lik = GaussianModified(D=K)

input_dim = dimX
pred_kernel = gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=1.0)
assign_kernel = gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=1.0)
Z, Z_assign = kmeans(Xtrain, num_ind, seed=0)[0], kmeans(Xtrain, num_ind, seed=1)[0]
# Z, Z_assign = rng.uniform(-2 * np.pi, 2 * np.pi, size=(num_ind, 1)), rng.uniform(-2 * np.pi, 2 * np.pi,
#                                                                                  size=(num_ind, 1))

pred_layer = SVGPModified(kernel=pred_kernel, likelihood=gaussian_lik, inducing_variable=Z, num_latent_gps=K,
                          whiten=True)
assign_layer = SVGPModified(kernel=assign_kernel, likelihood=gaussian_lik, inducing_variable=Z_assign, num_latent_gps=K,
                            whiten=True)

# model definition
model = SMGP(likelihood=gaussian_lik, pred_layer=pred_layer, assign_layer=assign_layer, K=K, num_samples=num_samples,
             num_data=num_data)

dataset = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
dataset = dataset.shuffle(buffer_size=num_data, seed=seed)
dataset = dataset.batch(num_minibatch)

optimizer = tf.optimizers.Adam(lr)

print('{:>5s}'.format("iter") + '{:>24s}'.format("ELBO:"))
iters = []
elbos = []
for i in range(1, num_iter + 1):
    try:
        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                # Record gradients of the loss with respect to the trainable variables
                elbo = model._build_likelihood(x_batch, y_batch)
                loss_value = -elbo
                gradients = tape.gradient(loss_value, model.trainable_variables)

            # Use the optimizer to apply the gradients to update the trainable variables
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if i % 5 == 0 or i == 0:
                print('{:>5d}'.format(i) + '{:>24.6f}'.format(elbo))
                # gpflow.utilities.print_summary(model)
                iters.append(i)
                elbos.append(elbo)
    except KeyboardInterrupt as e:
        print("stopping training")
        break

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
fig = plt.figure(figsize=(16, 10))
ax = []
for i in range(1, 10):
    if i > 4:
        ax.append(fig.add_subplot(5, 2, i))
        continue
    ax.append(fig.add_subplot(5, 2, i, projection='3d'))

ax[0].scatter(Xtrain[:, 0], Xtrain[:, 1], Ytrain, s=1)
ax[0].set_title("Raw Data")
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[0].set_zlabel('z')
ax[0].grid()

ax[1].scatter(Xt_tiled[:, 0:1], Xt_tiled[:, 1:2], samples_y_stack.flatten(), marker='+', alpha=0.01,
              color=mcolors.TABLEAU_COLORS['tab:red'])
ax[1].scatter(Xt_tiled[:, 0:1], Xt_tiled[:, 1:2], samples_f_stack.flatten(), marker='+', alpha=0.01,
              color=mcolors.TABLEAU_COLORS['tab:blue'])
ax[1].scatter(Xtrain[:, 0], Xtrain[:, 1], Ytrain, marker='x', color='black', alpha=0.1)
ax[1].set_title("Many GPs")
ax[1].set_xlabel('x1')
ax[1].set_ylabel('x2')
ax[1].set_zlabel('z')
ax[1].set_ylim(1.2 * min(Ytrain), 1.2 * max(Ytrain))
ax[1].grid()

assign_ = model.predict_assign(Xplot)
for i in range(K):
    ax[2].scatter(Xplot[:, 0], Xplot[:, 1], assign_[:, i], color=colors[i], s=1)
ax[2].set_title("Assignment 3D")
ax[2].set_xlabel('x1')
ax[2].set_ylabel('x2')
ax[2].set_zlabel('z')
ax[2].grid()

fmean, _ = model.predict_y(Xplot)
fmean_ = np.mean(fmean, 0)
for i in range(K):
    ax[3].scatter(Xplot[:, 0], Xplot[:, 1], fmean_[:, i], color=colors[i], s=1)
ax[3].set_title("Prediction 3D")
ax[3].set_xlabel('x1')
ax[3].set_ylabel('x2')
ax[3].set_zlabel('z')
ax[3].grid()

ax[4].plot(iters, elbos, 'o-', ms=8, alpha=0.5)
ax[4].set_xlabel('Iterations')
ax[4].set_ylabel('ELBO')
ax[4].grid()

stumpsX_const_value = 0.0
stumpsY_const_value = 1.0

Xtest_stumpsX = np.c_[Xplot[:, 0], stumpsY_const_value * np.ones(len(Xplot[:, 0]))]
Xtest_stumpsY = np.c_[stumpsX_const_value * np.ones(len(Xplot[:, 1])), Xplot[:, 1]]

Xtests = [Xtest_stumpsX, Xtest_stumpsY]

for i in range(2):
    assign_ = model.predict_assign(Xtests[i])
    ax[i + 5].plot(Xtests[i][:, i], assign_, 'o', markersize=1)
    ax[i + 5].set_xlabel('x' + str(i + 1))
    ax[i + 5].set_ylabel('softmax(assignment)')
    ax[i + 5].grid()

for i in range(2):
    fmean, fvar = model.predict_y(Xtests[i])
    fmean_, fvar_ = np.mean(fmean, 0), np.mean(fvar, 0)

    X_sorted = np.zeros_like(Xtests[i])
    sort_indices = np.argsort(Xtests[i][:, i])
    X_sorted[:, i] = Xtests[i][sort_indices, i]
    fmean_sorted = fmean_[sort_indices]
    fvar_sorted = fvar_[sort_indices]

    lb, ub = (fmean_sorted - 2 * fvar_sorted ** 0.5), (fmean_sorted + 2 * fvar_sorted ** 0.5)

    for k in range(K):
        ax[i + 7].plot(X_sorted[:, i], fmean_sorted[:, k], '-', alpha=1., color=colors[k])
        ax[i + 7].fill_between(X_sorted[:, i], lb[:, k], ub[:, k], alpha=0.3, color=colors[k])

    ax[i + 7].scatter(Xtrain[:, i], Ytrain, marker='x', color='black', alpha=0.5)
    ax[i + 7].set_xlabel('x' + str(i + 1))
    ax[i + 7].set_ylabel('Pred. of GP experts')
    ax[i + 7].grid()

plt.tight_layout()
plt.show()
fig.savefig('figs/demo_' + name + "_".join(numerical_attributes) + '.png')
