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


def eigen_analysis(data, labels, k, plot=True):
    mean, eigenpairs = pca(data, k)
    for value, vector in eigenpairs:
        vector = vector.reshape((32, 32))
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(vector.T, cmap="Greys")
            plt.title(f"Eigenvalue: {value}")
            plt.waitforbuttonpress()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # The train file contains the images for
    parser.add_argument("-data", default="images/ORL_32x32.mat", help="The file with all the image data.")
    parser.add_argument("-train", default="images/3Train/3.mat", help="The file with training indices. 3Train, 5Train or 7Train.")
    parser.add_argument("-k", default=20, help="How many principal components to use.")
    args = vars(parser.parse_args())

    raw_data, raw_train = loadmat(args["data"]), loadmat(args["train"])
    nlabels, ndata = np.array(raw_data["gnd"]), np.array(raw_data["fea"])
    ndata = ndata / 255.  # normalize

    # All credits to Christoph Emunds (@Fallscout) for figuring that out:
    # Need to subtract 1 from the indices, because matrices
    # have originally been created in MATLAB and
    # MATLAB indexing starts at 1 as opposed to 0
    train_data = ndata[raw_train["trainIdx"].flatten() - 1]
    train_labels = nlabels[raw_train["trainIdx"].flatten() - 1].flatten()
    test_data = ndata[raw_train["testIdx"].flatten() - 1]
    test_labels = nlabels[raw_train["testIdx"].flatten() - 1].flatten()

    eigen_analysis(train_data, nlabels, args["k"])
