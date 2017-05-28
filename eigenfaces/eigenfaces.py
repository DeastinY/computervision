import logging
import argparse
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from matplotlib import pyplot as plt


logging.basicConfig(level=logging.INFO)


def pca(data, mean, k):
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
    return eigenpairs[:k]


def plot_eigenface(value, vector):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(vector.T, cmap="Greys")
    plt.title(f"Eigenvalue: {value}")
    plt.waitforbuttonpress()


def plot_facecompare(a, b, title_a, title_b):
    a, b = a.reshape(32, 32), b.reshape(32, 32)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(a.T, cmap="Greys")
    plt.title(title_a)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(b.T, cmap="Greys")
    plt.title(title_b)
    plt.waitforbuttonpress()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-data", default="images/ORL_32x32.mat", help="The file with all the image data.")
    parser.add_argument("-train", default="images/3Train/3.mat", help="The file with training indices. (3/5/7Train)")
    parser.add_argument("-k", default=10, help="How many principal components to use.")
    parser.add_argument("--pc", action='store_true', help="Set flag to plot the first k principal components.")
    parser.add_argument("--compare", action='store_true', help="Set flag to plot the comparison of pca and original.")
    args = vars(parser.parse_args())

    # Handle data
    raw_data, raw_train = loadmat(args["data"]), loadmat(args["train"])
    labels, data = np.array(raw_data["gnd"]), np.array(raw_data["fea"])
    data = data / 255.  # normalize
    # All credits to Christoph Emunds (@Fallscout) for figuring that out:
    # Need to subtract 1 from the indices, because matrices
    # have originally been created in MATLAB and
    # MATLAB indexing starts at 1 as opposed to 0
    dtrain = data[raw_train["trainIdx"].flatten() - 1]
    ltrain = labels[raw_train["trainIdx"].flatten() - 1].flatten()
    dtest = data[raw_train["testIdx"].flatten() - 1]
    ltest = labels[raw_train["testIdx"].flatten() - 1].flatten()

    # Execute PCA
    dtmean = np.mean(dtrain, axis=0)  # the 32 x 32 images are shaped 1 x 1024 so mean is of length 1024
    eigenpairs = pca(dtrain, dtmean, args['k'])
    eigenvectors = np.array([vector for _, vector in eigenpairs])
    [plot_eigenface(value, vector.reshape((32, 32))) for value, vector in eigenpairs if args['pc']]
    # Project Training-Data to new space
    ntdata = (dtrain - dtmean) @ eigenvectors.T
    [
        plot_facecompare(dtmean + (ws @ eigenvectors), dtrain[i], "Reconstruct", "Original")
        for i, ws in enumerate(ntdata) if args['compare']
    ]

