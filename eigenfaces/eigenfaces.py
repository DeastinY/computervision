import logging
import argparse
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score


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


def plot_compare(original, compare_to):
    for i, ws in enumerate(compare_to):
        plot_facecompare(train_mean + (ws @ eigenvectors), original[i], "Reconstruct", "Original")


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
    parser.add_argument("--comptrain", action='store_true', help="Compare training data to PCA reconstruct.")
    parser.add_argument("--comptest", action='store_true', help="Compare test data to PCA reconstruct.")
    args = vars(parser.parse_args())

    logging.info(f"Executing eigenface analysis with {args['train']} and k={args['k']}")
    # Handle data
    raw_data, raw_train = loadmat(args["data"]), loadmat(args["train"])
    labels, data = np.array(raw_data["gnd"]), np.array(raw_data["fea"])
    data = data / 255.  # normalize
    # All credits to Christoph Emunds (@Fallscout) for figuring that out:
    # Need to subtract 1 from the indices, because matrices
    # have originally been created in MATLAB and
    # MATLAB indexing starts at 1 as opposed to 0
    train_data = data[raw_train["trainIdx"].flatten() - 1]
    train_label = labels[raw_train["trainIdx"].flatten() - 1].flatten()
    test_data = data[raw_train["testIdx"].flatten() - 1]
    test_label = labels[raw_train["testIdx"].flatten() - 1].flatten()

    # Execute PCA
    train_mean = np.mean(train_data, axis=0)  # the 32 x 32 images are shaped 1 x 1024 so mean is of length 1024
    eigenpairs = pca(train_data, train_mean, int(args['k']))
    eigenvectors = np.array([vector for _, vector in eigenpairs])
    [plot_eigenface(value, vector.reshape((32, 32))) for value, vector in eigenpairs if args['pc']]
    # Project Training-Data to new space
    train_pca = (train_data - train_mean) @ eigenvectors.T
    if args['comptrain']:
        plot_compare(train_data, train_pca)
    # Project Sample-Data to new space
    test_pca = (test_data - train_mean) @ eigenvectors.T
    if args['comptest']:
        plot_compare(test_data, test_pca)
    # Find matches using nearest neighbor
    neighbors = NearestNeighbors(n_neighbors=1).fit(train_pca)
    distances, indices = neighbors.kneighbors(test_pca)
    predict_label = train_label[indices].flatten()
    accuracy = accuracy_score(test_label, predict_label)
    logging.info(f"Accurracy : {accuracy}")





