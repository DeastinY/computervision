import logging
import argparse
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
PATH_OUT = Path("report/images")
PATH_OUT.mkdir(exist_ok=True)
FILE_STATS = Path("stats.json")


def pca(data, mean, k):
    """
    Executes principal component analysis on data.
    :param data: The dataset to perform pca on.
    :param mean: The mean of this dataset.
    :param k: How many principal components to return.
    :return: The k first principal components.
    """
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


counter = 0


def plot_eigenface(value, vector, *, save_image=False):
    """
    Plots the passed eigenface.
    :param value: The eigenvalue will be in the title.
    :param vector: The eigenface.
    :param save_image: Set to True to save the image to eigenface_x.png.
    """
    global counter
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(vector.T, cmap="Greys")
    plt.title(f"Eigenvalue: {value}")
    if not save_image:
        plt.waitforbuttonpress()
    else:
        plt.savefig(str(PATH_OUT / f"eigenface_{counter}.png"))
        counter += 1
    plt.close()


def plot_compare(original, compare_to, *, save_image=False):
    """
    Plots the comparison of images in the eigenface space and original images.
    :param original: The images in the original representation.
    :param compare_to: The images represented in the eigenface space.
    :param save_image: Set to True to save the image to ReconstructX_OriginalY.png.
    """
    for i, ws in enumerate(compare_to):
        plot_facecompare(train_mean + (ws @ eigenvectors), original[i],
                         f"Reconstruct {i}", f"Original {i}", save_image=save_image)


def plot_facecompare(a, b, title_a, title_b, *, save_image=False):
    """
    Utilized by plot_compare to plot two images side by side.
    :param a: The first image.
    :param b: The second image.
    :param title_a: Title for the first image.
    :param title_b: Title for the second image.
    :param save_image: Set to True to save the image to ReconstructX_OriginalY.png.
    """
    a, b = a.reshape(32, 32), b.reshape(32, 32)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(a.T, cmap="Greys")
    plt.title(title_a)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(b.T, cmap="Greys")
    plt.title(title_b)
    if not save_image:
        plt.waitforbuttonpress()
    else:
        plt.savefig(str(PATH_OUT / f"{title_a}_{title_b}.png"))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-data", default="images/ORL_32x32.mat", help="The file with all the image data.")
    parser.add_argument("-train", default="images/3Train/3.mat", help="The file with training indices. (3/5/7Train)")
    parser.add_argument("-k", default=10, help="How many principal components to use.")
    parser.add_argument("--pc", action='store_true', help="Set flag to plot the first k principal components.")
    parser.add_argument("--comptrain", action='store_true', help="Compare training data to PCA reconstruct.")
    parser.add_argument("--comptest", action='store_true', help="Compare test data to PCA reconstruct.")
    parser.add_argument("--images", action='store_true', help="Save all images.")
    args = vars(parser.parse_args())

    if args['images']:
        args['comptrain'], args['comptest'], args['pc'] = True, True, True
        PATH_OUT = PATH_OUT / f"{args['k']}_{args['train'][-5]}"
        PATH_OUT.mkdir(exist_ok=True)

    logging.info(f"Executing eigenface analysis with {args['train']} and k={args['k']}")

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

    # Execute PCA and collect eigenvectors
    train_mean = np.mean(train_data, axis=0)  # the 32 x 32 images are shaped 1 x 1024 so mean is of length 1024
    eigenpairs = pca(train_data, train_mean, int(args['k']))
    eigenvectors = np.array([vector for _, vector in eigenpairs])
    if args['pc']:
        [plot_eigenface(value, vector.reshape((32, 32)), save_image=args['images']) for value, vector in eigenpairs]

    # Project Training-Data to new space
    train_new = (train_data - train_mean) @ eigenvectors.T
    if args['comptrain']:
        plot_compare(train_data, train_new, save_image=args['images'])

    # Project Sample-Data to new space
    test_new = (test_data - train_mean) @ eigenvectors.T
    if args['comptest']:
        plot_compare(test_data, test_new, save_image=args['images'])

    # Find matches using nearest neighbor
    neighbors = NearestNeighbors(n_neighbors=1).fit(train_new)
    distances, indices = neighbors.kneighbors(test_new)
    predict_label = train_label[indices].flatten()
    accuracy = accuracy_score(test_label, predict_label)

    logging.info(f"Accurracy : {accuracy}")




