import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_john_doe_runs():
    def load_john_doe_data():
        csv_path = os.path.join("../data", "john_doe_dataset.csv")
        return pd.read_csv(csv_path)

    john_doe = load_john_doe_data()

    john_doe = john_doe[
        (john_doe['batterRuns'] == 0) | (john_doe['batterRuns'] == 1) | (john_doe['batterRuns'] == 4) | (
                    john_doe['batterRuns'] == 6)]

    seam = ['FAST_SEAM', 'MEDIUM_SEAM', 'SEAM']
    john_doe = john_doe[john_doe['bowlingStyle'].isin(seam)]
    john_doe = john_doe[john_doe['rightArmedBowl'] == True]

    categorical_attributes = []
    numerical_attributes = ['stumpsX', 'stumpsY']
    # numerical_attributes = ['stumpsX', 'stumpsY', 'pitchX', 'pitchY']
    all_columns = numerical_attributes + ['batterRuns']

    john_doe = john_doe[all_columns]

    features = john_doe.drop(['batterRuns'], axis=1)
    targets = john_doe['batterRuns']

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, targets, test_size=0.2)
    Xtrain, Xtest, Ytrain, Ytest = Xtrain.to_numpy(), Xtest.to_numpy(), Ytrain.to_numpy(), Ytest.to_numpy()
    Ytrain, Ytest = Ytrain.reshape((len(Ytrain), 1)), Ytest.reshape((len(Ytest), 1))

    return len(Xtrain), Xtrain, Ytrain, Xtest, numerical_attributes


def load_john_doe():
    def load_john_doe_data():
        csv_path = os.path.join("../data", "john_doe_dataset.csv")
        return pd.read_csv(csv_path)

    def boundary_mapper(run_value):
        if run_value in [0, 1]:
            return 0
        elif run_value in [4, 6]:
            return 1
        else:
            raise ValueError("Invalid batterRuns value")

    john_doe = load_john_doe_data()

    john_doe = john_doe[
        (john_doe['batterRuns'] == 0) | (john_doe['batterRuns'] == 1) | (john_doe['batterRuns'] == 4) | (
                john_doe['batterRuns'] == 6)]
    john_doe['boundary'] = john_doe['batterRuns'].apply(boundary_mapper)

    seam = ['FAST_SEAM', 'MEDIUM_SEAM', 'SEAM']
    john_doe = john_doe[john_doe['bowlingStyle'].isin(seam)]
    john_doe = john_doe[john_doe['rightArmedBowl'] == True]

    categorical_attributes = []
    numerical_attributes = ['stumpsX', 'stumpsY']
    # numerical_attributes = ['stumpsX', 'stumpsY', 'pitchX', 'pitchY']
    all_columns = numerical_attributes + ['boundary']

    john_doe = john_doe[all_columns]

    features = john_doe.drop(['boundary'], axis=1)
    targets = john_doe['boundary']

    unique, counts = np.unique(targets, return_counts=True)
    percent = counts[0] / (counts[0] + counts[1])

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, targets, test_size=0.2)
    Xtrain, Xtest, Ytrain, Ytest = Xtrain.to_numpy(), Xtest.to_numpy(), Ytrain.to_numpy(), Ytest.to_numpy()
    Ytrain, Ytest = Ytrain.reshape((len(Ytrain), 1)), Ytest.reshape((len(Ytest), 1))

    return len(Xtrain), Xtrain, Ytrain, Xtest, numerical_attributes


def load_toy_data_categorical(rng: np.random.Generator):
    N, Ns, lambda_ = 500, 100, 0.1

    x_min = -6.0
    x_max = 6.0
    Xtrain = rng.uniform(low=x_min, high=x_max, size=(N, 1))

    Ytrain = np.where(Xtrain < 0.0, 1, 0)
    outlier_indices = rng.choice(N, size=int(N * lambda_), replace=False)
    Ytrain[outlier_indices] = 1 - Ytrain[outlier_indices]

    Xtest = np.linspace(x_min, x_max, Ns).reshape(Ns, 1)

    return N, Xtrain, Ytrain, Xtest


def load_toy_multimodal_data(rng: np.random.Generator):
    N, Ns = 1500, 100

    epsilon = rng.normal(0, 0.1, (N // 3, 1))

    Xtrain = rng.uniform(low=-2 * np.pi, high=2 * np.pi, size=(N, 1))

    Ytrain1 = np.sin(Xtrain[0:N // 3]) + epsilon
    Ytrain2 = np.sin(Xtrain[N // 3:2 * N // 3]) - 2 * np.exp(-0.5 * pow(Xtrain[N // 3:2 * N // 3] - 2, 2)) + epsilon
    Ytrain3 = -2 - (3 / (8 * np.pi)) * Xtrain[2 * N // 3:N] + (3 / 10) * np.sin(2 * Xtrain[2 * N // 3:N]) + epsilon
    Ytrain = np.concatenate((Ytrain1, Ytrain2, Ytrain3))

    Xtest = np.linspace(-2 * np.pi, 2 * np.pi, Ns)[:, None]

    return N, Xtrain, Ytrain, Xtest


def load_toy_data_assoc():
    N, Ns, lambda_ = 500, 100, .4
    delta = np.random.binomial(1, lambda_, size=(N, 1))
    noise = np.random.randn(N, 1) * .15
    epsilon = np.random.uniform(low=-1., high=3., size=(N, 1))
    Xtrain = np.random.uniform(low=-3., high=3., size=(N, 1))
    Ytrain = (1. - delta) * (np.cos(.5 * np.pi * Xtrain) * np.exp(-.25 * Xtrain ** 2) + noise) + delta * epsilon
    Xtest = np.linspace(-3, 3, Ns)[:, None]
    return N, Xtrain, Ytrain, Xtest


def load_toy_2d_data(rng: np.random.Generator):
    N, Ns = 500, 100

    x_min = [-12.0, -12.0]
    x_max = [12.0, 12.0]
    Xtrain = rng.uniform(low=x_min, high=x_max, size=(N, 2))

    # gaussian = np.exp(-((Xtrain[:, 0] - 0.5) ** 2 + (Xtrain[:, 1] - 0.5) ** 2) / (2 * 0.1 ** 2))
    # sine = np.sin(2 * np.pi * 5 * np.sqrt((Xtrain[:, 0] - 0.5) ** 2 + (Xtrain[:, 1] - 0.5) ** 2))
    radial = np.sqrt((Xtrain[:, 0] - 0.5) ** 2 + (Xtrain[:, 1] - 0.5) ** 2)
    radial2 = np.sqrt((Xtrain[:, 0] - 0.5) ** 2 + (Xtrain[:, 1] - 0.5) ** 2) + 10.0
    # himmelblaus = (Xtrain[:, 0] ** 2 + Xtrain[:, 1] - 11) ** 2 + (Xtrain[:, 0] + Xtrain[:, 1] ** 2 - 7) ** 2 + 10.0

    Ytrain = np.concatenate((radial[0: N // 2], radial2[N // 2: N])).reshape((N, 1))
    # Ytrain = radial.reshape((N, 1))

    Xtest = np.linspace(x_min, x_max, Ns)

    return N, Xtrain, Ytrain, Xtest


def load_toy_2d_data_categorical(rng: np.random.Generator):
    N, Ns, lambda_ = 500, 100, 0.1

    x_min = [-6.0, -6.0]
    x_max = [6.0, 6.0]
    Xtrain = rng.uniform(low=x_min, high=x_max, size=(N, 2))

    Ytrain = np.where((Xtrain[:, 0] < 0) & (Xtrain[:, 1] < 0), 1, 0)

    # to add occasional outliers
    outlier_indices = rng.choice(N, size=int(N * lambda_), replace=False)
    Ytrain[outlier_indices] = 1 - Ytrain[outlier_indices]
    Ytrain = Ytrain.reshape((N, 1))

    Xtest = np.linspace(x_min, x_max, Ns)

    return N, Xtrain, Ytrain, Xtest
