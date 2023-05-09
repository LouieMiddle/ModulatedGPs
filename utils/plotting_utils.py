import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes


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
