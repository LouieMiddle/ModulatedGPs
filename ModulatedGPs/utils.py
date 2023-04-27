import matplotlib.colors as mcolors
import numpy as np
import tensorflow as tf
from gpflow import config
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

float_type = config.default_float()


def reparameterize(mean, var, z, full_cov=False):
    """
    Implements the 'reparameterization trick' for the Gaussian, either full rank or diagonal

    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)

    If full_cov=True then var must be of shape S,U,N,N,D and the full covariance is used. Otherwise
    var must be S,N,D and the operation is elementwise

    :param mean: mean of shape S,N,D
    :param var: covariance of shape S,N,D or S,N,N,D
    :param z: samples form unit Gaussian of shape S,N,D
    :param full_cov: bool to indicate whether var is of shape S,N,N,D or S,N,D
    :return sample from N(mean, var) of shape S,N,D
    """
    if var is None:
        return mean

    if full_cov is False:
        return mean + z * (var + config.default_jitter()) ** 0.5
    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2]  # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = config.default_jitter() * tf.eye(N, dtype=float_type)[None, None, :, :]  # 11NN
        chol = tf.cholesky(var + I)  # SDNN
        z_res = tf.transpose(z, [0, 2, 1])[:, :, :, None]  # SND->SDN1
        f = mean + tf.matmul(chol, z_res)[:, :, :, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1))  # SND


def inv_probit(x: tf.Tensor) -> tf.Tensor:
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


def load_categorical_data(rng: np.random.Generator):
    N, Ns, lambda_ = 500, 100, 0.1

    x_min = -6.0
    x_max = 6.0
    Xtrain = rng.uniform(low=x_min, high=x_max, size=(N, 1))

    Ytrain = np.where(Xtrain < 0.0, 1, 0)
    outlier_indices = rng.choice(N, size=int(N * lambda_), replace=False)
    Ytrain[outlier_indices] = 1 - Ytrain[outlier_indices]

    Xtest = np.linspace(x_min, x_max, Ns).reshape(Ns, 1)

    return N, Xtrain, Ytrain, Xtest


def load_multimodal_data(rng: np.random.Generator):
    N, Ns = 3000, 500

    epsilon = rng.normal(0, 0.2, (N // 3, 1))

    Xtrain = rng.uniform(low=-2 * np.pi, high=2 * np.pi, size=(N, 1))

    Ytrain1 = np.sin(Xtrain[0:N // 3]) + epsilon
    Ytrain2 = np.sin(Xtrain[N // 3:2 * N // 3]) - 2 * np.exp(-0.5 * pow(Xtrain[N // 3:2 * N // 3] - 2, 2)) + epsilon
    Ytrain3 = -2 - (3 / (8 * np.pi)) * Xtrain[2 * N // 3:N] + (3 / 10) * np.sin(2 * Xtrain[2 * N // 3:N]) + epsilon
    # Ytrain1 = np.sin(Xtrain[0:N // 3])
    # Ytrain2 = np.sin(Xtrain[N // 3:2 * N // 3]) - 2 * np.exp(-0.5 * pow(Xtrain[N // 3:2 * N // 3] - 2, 2))
    # Ytrain3 = -2 - (3 / (8 * np.pi)) * Xtrain[2 * N // 3:N] + (3 / 10) * np.sin(2 * Xtrain[2 * N // 3:N])
    Ytrain = np.concatenate((Ytrain1, Ytrain2, Ytrain3))

    Xtest = np.linspace(-2 * np.pi, 2 * np.pi, Ns)[:, None]

    return N, Xtrain, Ytrain, Xtest


def load_data_assoc():
    N, Ns, lambda_ = 1000, 500, .4
    delta = np.random.binomial(1, lambda_, size=(N, 1))
    noise = np.random.randn(N, 1) * .15
    epsilon = np.random.uniform(low=-1., high=3., size=(N, 1))
    Xtrain = np.random.uniform(low=-3., high=3., size=(N, 1))
    Ytrain = (1. - delta) * (np.cos(.5 * np.pi * Xtrain) * np.exp(-.25 * Xtrain ** 2) + noise) + delta * epsilon
    Xtest = np.linspace(-3, 3, Ns)[:, None]
    return N, Xtrain, Ytrain, Xtest


def load_2d_data(rng: np.random.Generator):
    N, Ns = 1000, 100

    x_min = [-12.0, -12.0]
    x_max = [12.0, 12.0]
    Xtrain = rng.uniform(low=x_min, high=x_max, size=(N, 2))

    # gaussian = np.exp(-((Xtrain[:, 0] - 0.5) ** 2 + (Xtrain[:, 1] - 0.5) ** 2) / (2 * 0.1 ** 2))
    # sine = np.sin(2 * np.pi * 5 * np.sqrt((Xtrain[:, 0] - 0.5) ** 2 + (Xtrain[:, 1] - 0.5) ** 2))
    radial = np.sqrt((Xtrain[:, 0] - 0.5) ** 2 + (Xtrain[:, 1] - 0.5) ** 2)
    radial2 = np.sqrt((Xtrain[:, 0] - 0.5) ** 2 + (Xtrain[:, 1] - 0.5) ** 2) + 10.0
    # himmelblaus = (Xtrain[:, 0] ** 2 + Xtrain[:, 1] - 11) ** 2 + (Xtrain[:, 0] + Xtrain[:, 1] ** 2 - 7) ** 2 + 10.0

    # fig = plt.figure()
    # ax1, ax2 = fig.add_subplot(2, 1, 1, projection='3d'), fig.add_subplot(2, 1, 2, projection='3d')
    # ax1.set_title("radial")
    # ax1.scatter(Xtrain[0: N // 2][:, 0], Xtrain[0: N // 2][:, 1], radial[0: N // 2])
    # ax2.set_title("radial 2")
    # ax2.scatter(Xtrain[N // 2: N][:, 0], Xtrain[N // 2: N][:, 1], radial2[N // 2: N])
    # plt.show()

    Ytrain = np.concatenate((radial[0: N // 2], radial2[N // 2: N])).reshape((N, 1))
    # Ytrain = radial.reshape((N, 1))

    Xtest = np.linspace(x_min, x_max, Ns)

    return N, Xtrain, Ytrain, Xtest


def load_2d_data_categorical(rng: np.random.Generator):
    N, Ns, lambda_ = 500, 100, 0.1

    x_min = [-6.0, -6.0]
    x_max = [6.0, 6.0]
    Xtrain = rng.uniform(low=x_min, high=x_max, size=(N, 2))

    Ytrain = np.where((Xtrain[:, 0] < 0) & (Xtrain[:, 1] < 0), 1, 0)

    # to add occasional outliers
    outlier_indices = rng.choice(N, size=int(N * lambda_), replace=False)
    Ytrain[outlier_indices] = 1 - Ytrain[outlier_indices]

    # Ytrain = Ytrain[:, 0:1]
    Ytrain = Ytrain.reshape((N, 1))

    Xtest = np.linspace(x_min, x_max, Ns)

    return N, Xtrain, Ytrain, Xtest


def plot_2d_kernel_prediction(axes, svgp) -> None:
    Xplot = np.linspace([-6.0, -6.0], [6.0, 6.0], 100)
    X = (tf.tile(Xplot[None, :, :], [1, 1, 1]), None)[0]

    f_mean, f_var = svgp.predict_f(X, full_cov=False)
    f_lower, f_upper = (f_mean - 2 * f_var ** 0.5), (f_mean + 2 * f_var ** 0.5)

    colors = [mcolors.TABLEAU_COLORS[key] for key in mcolors.TABLEAU_COLORS.keys()]
    for i in range(3):
        axes[0].plot(Xplot[:, 0], f_mean[:, i], '-', alpha=1., color=colors[i])
        axes[0].fill_between(Xplot[:, 0], f_lower[:, i], f_upper[:, i], alpha=0.3, color=colors[i])
        axes[1].plot(Xplot[:, 1], f_mean[:, i], '-', alpha=1., color=colors[i])
        axes[1].fill_between(Xplot[:, 1], f_lower[:, i], f_upper[:, i], alpha=0.3, color=colors[i])

    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('Pred.')
    axes[1].set_xlabel('x2')
    axes[1].set_ylabel('Pred.')
    axes[0].grid()
    axes[1].grid()


def plot_2d_kernel(svgp) -> None:
    _, (pred_ax_x1, pred_ax_x2) = plt.subplots(nrows=1, ncols=2)
    plot_2d_kernel_prediction([pred_ax_x1, pred_ax_x2], svgp)


def plot_kernel_samples(ax: Axes, svgp) -> None:
    Xplot = np.linspace(-6.0, 6.0, 100)[:, None]
    n_samples = 3
    # predict_f_samples draws n_samples examples of the function f, and returns their values at Xplot.
    fs = svgp.predict_f_samples(Xplot, n_samples)
    ax.plot(Xplot, fs[:, :, 0].numpy().T)
    ax.set_title("Example $f$s")


def plot_kernel_prediction(ax: Axes, svgp) -> None:
    Xplot = np.linspace(-6.0, 6.0, 100)[:, None]

    f_mean, f_var = svgp.predict_f(Xplot, full_cov=False)
    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)

    mean_lines = ax.plot(Xplot, f_mean, "-")
    for i, mean_line in enumerate(mean_lines):
        color = mean_line.get_color()
        ax.plot(Xplot, f_lower, lw=0.1, color=color)
        ax.plot(Xplot, f_upper, lw=0.1, color=color)
        ax.fill_between(Xplot[:, 0], f_lower[:, i], f_upper[:, i], color=color, alpha=0.1)

    ax.set_title("Example data fit")


def plot_kernel(svgp) -> None:
    _, (samples_ax, prediction_ax) = plt.subplots(nrows=1, ncols=2)
    plot_kernel_samples(samples_ax, svgp)
    plot_kernel_prediction(prediction_ax, svgp)
