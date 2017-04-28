#!/usr/bin/python3

def load_images():
    """Loads all images into a dictionary with their names."""
    return {
        "image": ""
    }


def show(image):
    """Uses matplotlib to visualize the passed image."""
    pass


def hsv_split(image):
    """Converts a color image into three subimages containing the different channels."""
    pass


def to_binary(image, threshold):
    """Converts a color image to a binary image using the defined threshold."""
    pass


def convolve(image, filter):
    """Applies 2D covolution with zero-padding with the passed filter to the image."""
    pass


def gradient_magnitude(image, filter, direction):
    """
    Applies a filter with thresholds 0.05, 0.25, 0.5, 0.75 and 1 to the image.
    :param filter: Can either be sobel or prewitt
    :param direction: Defines the direction the filter is applied. Either horizontal, vertical or both.
    """
    pass


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
    pass
