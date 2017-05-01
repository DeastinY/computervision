#!/usr/bin/python3
import cv2
import math
import tkinter
import logging
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

logging.basicConfig(level=logging.DEBUG)


def load_images():
    """Loads all images into a dictionary with their names."""
    return {
        "image": cv2.imread('images/image.png'),
        "lab1a" : cv2.imread('images/lab1a.png'),
        "lab1b": cv2.imread('images/lab1b.png'),
        "lab2a": cv2.imread('images/lab2a.png'),
        "lab2b": cv2.imread('images/lab2b.png'),
        "lena": cv2.imread('images/Lena.jpg'),
        "unequalized_h": cv2.imread('images/Unequalized_H.jpg')
    }


def show(images, cols=4, gray=False):
    """
    Visualizes the passed images. Assumes a dict with names is passed.
    :type cols: Amount of columns to visualize the images in.
    """
    rows = int(math.ceil(len(images)/cols))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()
    for idx, name in enumerate(sorted(images)):
        ax = fig.add_subplot(gs[idx])
        if gray:
            ax.imshow(images[name], cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.axis("off")
    #fig.tight_layout()
    plt.show()


def show_cv(images):
    """Visualizes the passed images. Assumes a dict with names is passed. Uses OpenCV."""
    for name, img in images.items():
        cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hsv_split(image):
    """Converts a color image into three subimages containing the different channels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return {
        "hue": h,
        "saturation": s,
        "value": v
    }


def to_binary(image, threshold):
    """Converts a color image to a binary image using the defined threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold*255, 255, cv2.THRESH_BINARY)
    return {
        "Binary : {}".format(threshold): thresh
    }


def naive_convolve(f, g):
    # http://docs.cython.org/en/latest/src/tutorial/numpy.html
    # f is an image and is indexed by (v, w)
    # g is a filter kernel and is indexed by (s, t),
    #   it needs odd dimensions
    # h is the output image and is indexed by (x, y),
    #   it is not cropped
    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
    # smid and tmid are number of pixels between the center pixel
    # and the edge, ie for a 5x5 filter they will be 2.
    #
    # The output size is calculated by adding smid, tmid to each
    # side of the dimensions of the input image.
    vmax = f.shape[0]
    wmax = f.shape[1]
    smax = g.shape[0]
    tmax = g.shape[1]
    smid = smax // 2
    tmid = tmax // 2
    xmax = vmax + 2*smid
    ymax = wmax + 2*tmid
    # Allocate result image.
    h = np.zeros([xmax, ymax], dtype=np.float64)
    # Do convolution
    for x in range(xmax):
        for y in range(ymax):
            # Calculate pixel value for h at (x,y). Sum one component
            # for each pixel (s, t) of the filter g.
            s_from = max(smid - x, -smid)
            s_to = min((xmax - x) - smid, smid + 1)
            t_from = max(tmid - y, -tmid)
            t_to = min((ymax - y) - tmid, tmid + 1)
            value = 0
            for s in range(s_from, s_to):
                for t in range(t_from, t_to):
                    v = x - smid + s
                    w = y - tmid + t
                    value += g[smid - s, tmid - t] * f[v, w]
            h[x, y] = value
    return h


def gradient_magnitude(image, filter, direction):
    """
    Applies a filter with thresholds 0.05, 0.25, 0.5, 0.75 and 1 to the image.
    :param filter: Can either be sobel or prewitt
    :param direction: Defines the direction the filter is applied. Either horizontal, vertical or both.
    """
    if filter == "sobel":
        filter_m = {
            "x": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64),
            "y": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
        }
    elif filter == "prewitt":
        filter_m = {
            "x": np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64),
            "y": np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)
        }
    else:
        logging.error("Wrong filter name. Must bei either sobel or prewitt.")

    result = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if direction == "horizontal":
        result = naive_convolve(gray, filter_m['x'])
    elif direction == "vertical":
        result = naive_convolve(gray, filter_m['y'])
    elif direction == "both":
        result = naive_convolve(gray, filter_m['x']) + naive_convolve(gray, filter_m['y'])
        result *= 255.0/result.max()
    else:
        logging.error("Wrong direction. Must bei either horizontal, vertical or both.")

    return {
        "Gradient Magnitude:  {} {}".format(filter, direction): result
    }


def zero_crossings(image, size):
    """
    Returns the edges at zero-crossings of the image. Uses the Laplacian operator of size 3x3, 5x5 or 17x17.
    :param size: Size of the Laplacian filter. Either 3, 5 or 17. 
    """
    pass


def enhance_contrast(image, *args):
    """Enhances the contrast of an image."""
    pass


def enhance_clip(image, *args):
    """Applies clipping to the image."""
    pass


def histogram_equalization(image, bins):
    """Performs histogram equalization with the provided number of bins."""
    pass


def blur_1d(image, direction):
    """
    Blurs the image with an 1D filter of size 10. 
    :type direction: Defines the direction the filter is applied. Either horizontal, vertical or both.
    """
    pass


def blur_2d(image):
    """Applies a 10x10 blur filter to the image."""
    pass


def highpass(image):
    """Applies a high pass filter to the image."""
    filter = [0, -1, 0, -1, 4, -1, 0, -1, 0]
    pass


def dft_2d(image):
    """Performs a discrete fourier transform on a grayscale image. The components are shifted to the center."""
    pass


if __name__ == "__main__":
    def test_binaries(images):
        binaries = [to_binary(images['lab1b'], i/10) for i in range(10)]
        res = {}
        for b in binaries:
            res.update(b)
        show(res, 5, True)
        test_binaries(images)

    def test_gradient_magnitude(image):
        res = {}
        for f in ["sobel", "prewitt"]:
            for d in ["horizontal", "vertical", "both"]:
                res.update(gradient_magnitude(image, f, d))
        res.update({"original": image})
        show(res, 3, True)


    images = load_images()
    test_gradient_magnitude(images['lena'])
    #show_cv(hsv_split(images['lab1a']))
    #show(images)
