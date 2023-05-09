import warnings

warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import set_trainable

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]


# Sourced from GPFlow notebooks https://gpflow.github.io/GPflow/develop/notebooks/advanced/multiclass_classification.html

def plot_posterior_predictions(m, X, Y):
    f = plt.figure(figsize=(12, 6))
    a1 = f.add_axes([0.05, 0.05, 0.9, 0.6])
    a2 = f.add_axes([0.05, 0.7, 0.9, 0.1])
    a3 = f.add_axes([0.05, 0.85, 0.9, 0.1])

    xx = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    mu, var = m.predict_f(xx)
    p, _ = m.predict_y(xx)

    a3.set_xticks([])
    a3.set_yticks([])

    for c in range(m.likelihood.num_classes):
        x = X[Y.flatten() == c]

        color = colors[c]
        a3.plot(x, x * 0, ".", color=color)
        a1.plot(xx, mu[:, c], color=color, lw=2, label="%d" % c)
        a1.plot(xx, mu[:, c] + 2 * np.sqrt(var[:, c]), "--", color=color)
        a1.plot(xx, mu[:, c] - 2 * np.sqrt(var[:, c]), "--", color=color)
        a2.plot(xx, p[:, c], "-", color=color, lw=2)

    a2.set_ylim(-0.1, 1.1)
    a2.set_yticks([0, 1])
    a2.set_xticks([])

    a3.set_title("inputs X")
    a2.set_title(
        "predicted mean label value \
                 $\mathbb{E}_{q(\mathbf{u})}[y^*|x^*, Z, \mathbf{u}]$"
    )
    a1.set_title(
        "posterior process \
                $\int d\mathbf{u} q(\mathbf{u})p(f^*|\mathbf{u}, Z, x^*)$"
    )

    handles, labels = a1.get_legend_handles_labels()
    a1.legend(handles, labels)
    f.tight_layout()
    plt.show()


# reproducibility:
np.random.seed(0)
tf.random.set_seed(123)

# Number of functions and number of data points
C = 3
N = 100

# Lengthscale of the SquaredExponential kernel (isotropic -- change to `[0.1] * C` for ARD)
lengthscales = 0.1

# Jitter
jitter_eye = np.eye(N) * 1e-6

# Input
X = np.random.rand(N, 1)

# SquaredExponential kernel matrix
kernel_se = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
K = kernel_se(X) + jitter_eye

# Latents prior sample
f = np.random.multivariate_normal(mean=np.zeros(N), cov=K, size=(C)).T

# Hard max observation
Y = np.argmax(f, 1).flatten().astype(int)

# One-hot encoding
Y_hot = np.zeros((N, C), dtype=bool)
Y_hot[np.arange(N), Y] = 1

data = (X, Y)

plt.figure(figsize=(12, 6))
order = np.argsort(X.flatten())

for c in range(C):
    plt.plot(X[order], f[order, c], ".", color=colors[c], label=str(c))
    plt.plot(X[order], Y_hot[order, c], "-", color=colors[c])

plt.legend()
plt.xlabel("$X$")
plt.ylabel("Latent (dots) and one-hot labels (lines)")
plt.title("Sample from the joint $p(Y, \mathbf{f})$")
plt.grid()
plt.show()

# sum kernel: Matern32 + White
kernel = gpflow.kernels.Matern32() + gpflow.kernels.White(variance=0.01)

# Robustmax Multiclass Likelihood
invlink = gpflow.likelihoods.RobustMax(C)  # Robustmax inverse link function
likelihood = gpflow.likelihoods.MultiClass(
    3, invlink=invlink
)  # Multiclass likelihood
Z = X[::5].copy()  # inducing inputs

m = gpflow.models.SVGP(
    kernel=kernel,
    likelihood=likelihood,
    inducing_variable=Z,
    num_latent_gps=C,
    whiten=True,
    q_diag=True,
)

# Only train the variational parameters
set_trainable(m.kernel.kernels[1].variance, False)
set_trainable(m.inducing_variable, False)
gpflow.utilities.print_summary(m)

opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(
    m.training_loss_closure(data),
    m.trainable_variables,
    options=dict(maxiter=reduce_in_tests(1000)),
)
gpflow.utilities.print_summary(m)

plot_posterior_predictions(m, X, Y)
