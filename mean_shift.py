#!/usr/bin/python3

import json
import logging
import sys
from pathlib import Path
import time
import cv2
import numpy as np
from scipy import spatial
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
sys.setrecursionlimit(10000)
cached_tree = None


def get_neighbours(data, point, r):
    global cached_tree
    cached_tree = spatial.cKDTree(data) if cached_tree is None else cached_tree
    return cached_tree.query_ball_point(point, r)


def get_neighbours_cdist(data, point, r):
    distances = spatial.distance.cdist(np.array([point]), data)[0]
    return data[np.where(distances < r)]


def find_peak(data, point, r, t=0.01):
    def calc_new_shift(data, point, r):
        return data[get_neighbours(data, point, r)].mean(axis=0)

    peak, dist = None, t
    while dist >= t:
        peak = calc_new_shift(data, point, r)
        dist = spatial.distance.euclidean(peak, point)
        point = peak
    return peak


def meanshift(data, r):
    peaks, points, point_peaks = [], [], []
    for point in tqdm(data):
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


def find_peak_opt(data, point, r, c, t=0.01):
    def calc_new_shift(data, point, r):
        return data[get_neighbours(data, point, r)].mean(axis=0)

    dist, cpts, peak = t, [], None
    while dist >= t:
        peak = calc_new_shift(data, point, r)
        dist = spatial.distance.euclidean(peak, point)
        cpts.extend(get_neighbours(data, point, r / c))
        point = peak
    return peak, list(set(cpts))


def meanshift_opt(data, r, c=4.0):
    global cached_tree
    cached_tree = None
    peaks, point_peaks = [], np.zeros(data.shape[0], dtype='int16') - 1
    for i, point in enumerate(tqdm(data)):
        if point_peaks[i] != -1:
            continue
        peak, cpts = find_peak_opt(data, point, r, c)
        # Match peak to possible neighbours. Use cdist because we have only few peaks
        peak_neighbours = get_neighbours_cdist(np.array(peaks), peak, r / 2.) if len(peaks) > 0 else []
        if len(peak_neighbours) > 1:
            peak = peak_neighbours[0]
        else:
            peaks.append(peak)
        # Basin of Attraction
        peak_id = np.where(peaks == peak)[0]
        neighbours = get_neighbours(data, peak, r)
        point_peaks[neighbours] = peak_id
        point_peaks[cpts] = peak_id
    return np.array(peaks), point_peaks


def image_segment(image, r, out="out", scale=1, c=4.0):
    # preprocess the image
    image = cv2.imread(image)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    orig_image = np.array(image)
    image = cv2.GaussianBlur(image, (5, 5), 5.0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    # meanshift
    peaks, point_peaks = meanshift_opt(image, r, c)
    print("Found {} peaks !".format(len(peaks)))
    # convert back to show format
    converted_peaks = cv2.cvtColor(np.array([peaks[:, 0:3]], dtype=np.uint8), cv2.COLOR_LAB2BGR)[0]
    image = converted_peaks[point_peaks]
    image = image.reshape(*orig_image.shape)
    cv2.imwrite(out + '.jpg', image)
    return len(peaks)


def visualize(image, r, func):
    peaks, _ = func(image, r)
    print("Found {} peaks in {} points !".format(len(peaks), image.shape))
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*peaks.transpose(), c='black', s=100)
    ax.scatter(*image.transpose(), c='blue', s=1)

if __name__ == '__main__':

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

    import argparse

    parser = argparse.ArgumentParser(description="Executes meanshift.")
    parser.add_argument("image", help="The RGB image to process.")
    parser.add_argument("scale", default=1, type=float, help="Rescaling factor.")
    parser.add_argument("r", default=10, type=float, help="The neighbourhood range to consider.")
    args = vars(parser.parse_args())
    import time
    global_start = time.time()
    peaks = {}
    for c in range(1, 11, 1):
        for r in range (5, 21, 5):
            args['c'] = c
            args['r'] = r
            args['out'] = '{image}_c{c}_r{r}_s{scale}'.format(**args)
            logging.info("Calculating meanshift for {}".format(args))
            start = time.time()
            peaks[args['out']]=image_segment(**args)
            end = time.time()
            logging.info("Finished in {} s".format(end-start))
    logging.info("Processing all images took {} s".format(time.time()-global_start))
    print(peaks)
