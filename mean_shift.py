# coding: utf-8

# # Mean Shift
# #### Basic Implementation
# The following two functions find_peak and meanshift execute the basic mean shift algorithm.

# In[332]:

import math
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import requests
from IPython.core.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
from progress.bar import Bar
from scipy import spatial
from scipy.io import loadmat
from sklearn.datasets import *
from util import log_progress

# get_ipython().magic('matplotlib inline')
pylab.rcParams['figure.figsize'] = 16, 12
sys.setrecursionlimit(10000)

# ##### Load  and visualize sample data
# The matrix is loaded into a numpy array of dimensions (2000, 3)

# In[333]:

"""
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/181091.html
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/55075.html
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/368078.html
"""
SAMPLE_DATA = loadmat("pts.mat")['data'].transpose()
img_path = Path('images')
IMG_A = cv2.imread(str(img_path / "a.jpg"))
IMG_B = cv2.imread(str(img_path / "b.jpg"))
IMG_C = cv2.imread(str(img_path / "c.jpg"))

# ##### Utility functions
# 

# In[334]:

cached_tree = None


def get_neighbours(data, point, r):
    global cached_tree
    cached_tree = spatial.cKDTree(data) if cached_tree is None else cached_tree
    return cached_tree.query_ball_point(point, r)


def get_neighbours_cdist(data, point, r):
    distances = spatial.distance.cdist(np.array([point]), data)[0]
    return data[np.where(distances < r)]


# In[335]:

def find_peak(data, point, r, t=0.01):
    def calc_new_shift(data, point, r):
        return data[get_neighbours(data, point, r)].mean(axis=0)

    dist = t
    while dist >= t:
        peak = calc_new_shift(data, point, r)
        dist = spatial.distance.euclidean(peak, point)
        point = peak
    return peak


def meanshift(data, r):
    peaks, points, point_peaks = [], [], []
    for point in log_progress(data, 1, len(data)):
        peak = find_peak(data, point, r)
        # Match peak to possible neighbours. Use cdist because we have only few peaks
        neighbours = get_neighbours_cdist(np.array(peaks), peak, r / 2.) if len(peaks) > 0 else []
        if len(neighbours) > 1:
            peak = neighbours[0]
        else:
            peaks.append(peak)
        points.append(point)
        point_peaks.append(np.where(peaks == peak)[0][0])
    return np.array(peaks), np.array(points), np.array(point_peaks)


# In[336]:

def find_peak_opt(data, point, r, c=4.0, t=0.01):
    def calc_new_shift(data, point, r):
        return data[get_neighbours(data, point, r)].mean(axis=0)

    dist = t
    cpts = set()
    while dist >= t:
        peak = calc_new_shift(data, point, r)
        dist = spatial.distance.euclidean(peak, point)
        cpts = cpts.union(get_neighbours(data, point, r / c))
        point = peak
    return peak, list(cpts)


def meanshift_opt(data, r):
    peaks, point_peaks = [], np.zeros(data.shape[0], dtype='int16') - 1
    bar = Bar("Progress:", max=len(data))
    for i, point in enumerate(data):
        bar.next()
        if point_peaks[i] != -1:
            continue
        peak, cpts = find_peak_opt(data, point, r)
        # Match peak to possible neighbours. Use cdist because we have only few peaks
        peak_neighbours = get_neighbours_cdist(np.array(peaks), peak, r / 2.) if len(peaks) > 0 else []
        if len(peak_neighbours) > 1:
            peak = data[neighbours][0]
        else:
            peaks.append(peak)
        # Basin of Attraction
        peak_id = np.where(peaks == peak)[0]
        neighbours = get_neighbours(data, peak, r)
        point_peaks[neighbours] = peak_id
        point_peaks[cpts] = peak_id
    bar.finish()
    return np.array(peaks), point_peaks


# In[337]:

def image_segment(image, r, scale=0.5):
    # preprocess the image
    image = cv2.resize(image, None, fx=scale, fy=scale)
    orig_image = np.array(image)
    image = cv2.GaussianBlur(image, (5, 5), 5.0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    print("Image has {} points".format(image.shape))
    peaks, point_peaks = meanshift_opt(image, r)
    print("Found {} peaks !".format(len(peaks)))
    # convert back to show format
    converted_peaks = cv2.cvtColor(np.array([peaks[:, 0:3]], dtype=np.uint8), cv2.COLOR_LAB2BGR)[0]
    image = converted_peaks[point_peaks]
    image = image.reshape(*orig_image.shape)
    plt.imshow(image)
    plt.imshow(orig_image)


# ##### Execute the meanshift function
# Visualize the results

# In[ ]:

def visualize(image, r, func):
    peaks, _ = func(image, r)
    print("Found {} peaks in {} points !".format(len(peaks), image.shape))
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*peaks.transpose(), c='black', s=100)
    ax.scatter(*image.transpose(), c='blue', s=1)


# In[ ]:

# visualize(SAMPLE_DATA, r=5, func=meanshift_opt)
image_segment(IMG_A, r=6)
