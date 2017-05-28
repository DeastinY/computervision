import logging
import argparse
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from matplotlib import pyplot as plt


logging.basicConfig(level=logging.INFO)


def pca(data, k):
    mean = np.mean(data, axis=0)  # the 32 x 32 images are shaped 1 x 1024 so mean is of length 1024
    dmean = data-mean
    if dmean.shape[0] > dmean.shape[1]:
        eigenvalues, eigenvectors = np.linalg.eigh(dmean.T @ dmean)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(dmean @ dmean.T)
        eigenvectors = np.dot(dmean.T, eigenvectors).T
        for i in range(len(dmean)):
            eigenvectors[i] = eigenvectors[i] / np.linalg.norm(eigenvectors[i])
    eigenpairs = [(eigenvalues[i], eigenvectors[i]) for i in range(len(eigenvalues))]
    eigenpairs.sort(key=lambda x: x[0], reverse=True)
    return mean, eigenpairs[:k]


def plot_eigenface(value, vector):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(vector.T, cmap="Greys")
    plt.title(f"Eigenvalue: {value}")
    plt.waitforbuttonpress()


def eigen_analysis(data, labels, k, plot):
    mean, eigenpairs = pca(data, k)
    [plot_eigenface(value, vector.reshape((32, 32))) for value, vector in eigenpairs if plot]


def process_data(data, train):
    labels, data = np.array(data["gnd"]), np.array(data["fea"])
    data = data / 255.  # normalize
    # All credits to Christoph Emunds (@Fallscout) for figuring that out:
    # Need to subtract 1 from the indices, because matrices
    # have originally been created in MATLAB and
    # MATLAB indexing starts at 1 as opposed to 0
    dtrain = data[raw_train["trainIdx"].flatten() - 1]
    ltrain = labels[raw_train["trainIdx"].flatten() - 1].flatten()
    dtest = data[raw_train["testIdx"].flatten() - 1]
    lttest = labels[raw_train["testIdx"].flatten() - 1].flatten()
    return dtrain, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # The train file contains the images for
    parser.add_argument("-data", default="images/ORL_32x32.mat", help="The file with all the image data.")
    parser.add_argument("-train", default="images/3Train/3.mat", help="The file with training indices. (3/5/7Train)")
    parser.add_argument("-k", default=20, help="How many principal components to use.")
    parser.add_argument("--pc", action='store_true', help="Set flag to plot the first k principal components")
    args = vars(parser.parse_args())

    raw_data, raw_train = loadmat(args["data"]), loadmat(args["train"])
    train_data, nlabels = process_data(raw_data, raw_train)

    eigen_analysis(train_data, nlabels, args["k"], args["pc"])
