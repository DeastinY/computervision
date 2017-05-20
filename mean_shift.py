#!/usr/bin/python3

import json
import logging
import sys
import time
from pathlib import Path

import cv2  # Check README.md for installation instructions
import numpy as np
from scipy import spatial
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
sys.setrecursionlimit(10000)  # Needed for large KDTrees
cached_tree = None


def get_neighbours(data, point, r):
    """
    Calculates the distance between all points in data and point and returns those < r.
    Uses a c implementation of KDTrees and is fast when searching in high amounts of points.
    Uses caching and has to be reset manually to generate a new KDTree.
    :param data: The lba or lbaxy data of the image.
    :param point: The point to calculate the distances to.
    :param r: The distance to use as a threshold.
    :return: The indices of all points in data that are less then r from point. 
    """
    global cached_tree
    cached_tree = spatial.cKDTree(data) if cached_tree is None else cached_tree
    return cached_tree.query_ball_point(point, r)


def get_neighbours_cdist(data, point, r):
    """
    Calculates the spatial distance between all points in data and point and returs those < r.
    This method is used when only a small set of points are to consider.
    :param data: The lba or lbaxy data of the image.
    :param point: The point to calculate the distances to.
    :param r: The distance to use as a threshold.
    :return: All points in data that are less then r from point.
    """
    distances = spatial.distance.cdist(np.array([point]), data)[0]
    return data[np.where(distances < r)]


def find_peak(data, point, r, t=0.01):
    """
    Executes the non-optimized find_peak method.
    :param data: The lba or lbaxy data of the image.
    :param point: The point to find the peak for.
    :param r: The distance to consider for 'close' points. 
    :param t: Convergence criterium. If the shift was smaller than t the peak is returned.
    :return: The found peak.
    """
    peak, dist = None, t
    while dist >= t:
        peak = data[get_neighbours(data, point, r)].mean(axis=0)
        dist = spatial.distance.euclidean(peak, point)
        point = peak
    return peak


def meanshift(data, r):
    """
    Executes the non-optimized meanshift.
    :param data: The lba or lbaxy data of the image.
    :param r: The distance to consider for 'close' points.
    :return: the peaks and the corresponding index for each point.
    """
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
    return np.array(peaks), point_peaks


def find_peak_opt(data, point, r, c, t=0.01):
    """
    Executes the optimized find_peak method. The optimization adds points that are at a distance < r/c on the way
    to the peak to the peak as well.
    :param data: The lba or lbaxy data of the image.
    :param point: The point to find the peak for.
    :param r: The distance to consider for 'close' points.
    :param c: r/c points on the 'way to the peak' are collected and added to the same peak. 
    :param t: Convergence criterium. If the shift was smaller than t the peak is returned.
    :return: The peak and a list of the indices of the points that were on the < r/c path.
    """
    dist, cpts, peak = t, [], None
    while dist >= t:
        peak = data[get_neighbours(data, point, r)].mean(axis=0)
        dist = spatial.distance.euclidean(peak, point)
        cpts.extend(get_neighbours(data, point, r / c))
        point = peak
    return peak, list(set(cpts))


def meanshift_opt(data, r, c=4.0):
    """
    Executes the optimized meanshift algorithm on the passed image data.
    :param data: The lba or lbaxy data of the image.
    :param r: The distance to consider for 'close' points.
    :param c: r/c points on the 'way to the peak' are collected and added to the same peak.
    :return: the peaks and the corresponding index for each point.
    """
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


def image_segment(image, r, use_5d=False, scale=1.0, c=4.0):
    """
    Preprocesses the image and executes meanshift.
    :param image: The path to the image to work on.
    :param r: The distance to consider for 'close' points.
    :param use_5d: Add spatial features (x,y) to the LBA color features. 
    :param scale: Scale the input image to adjust runtime.
    :param c: r/c points on the 'way to the peak' are collected and added to the same peak.
    :return: How many peaks are found and an image with colors changed to the peak colors.
    """
    # Load image
    image = cv2.imread(image)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    orig_image = np.array(image)
    # Preprocess applying GaussioaBlur, converting the color to LAB and adding x,y if use_5d is true.
    image = cv2.GaussianBlur(image, (5, 5), 5.0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    if use_5d:
        [x, y] = np.meshgrid(range(1, orig_image.shape[1] + 1), range(1, orig_image.shape[0] + 1))
        x = np.array(x.T.reshape(x.shape[0] * x.shape[1]), dtype=float)
        y = np.array(y.T.reshape(y.shape[0] * y.shape[1]), dtype=float)
        l = np.array([y / np.max(y), x / np.max(x)]).transpose()
        image = np.concatenate((image, l), axis=1)
    # Execute meanshift
    peaks, point_peaks = meanshift_opt(image, r, c)
    # Convert the image back into its original colorspace and colors the pixels based on their peaks.
    converted_peaks = cv2.cvtColor(np.array([peaks[:, 0:3]], dtype=np.uint8), cv2.COLOR_LAB2BGR)[0]
    image = converted_peaks[point_peaks]
    image = image.reshape(*orig_image.shape)
    return len(peaks), image


def generate_report_images(image):
    """Generates images seen in the report. Uses c=1...10 r=5,10,15,20 and use_5d=True/False."""
    img_path = Path('images/report')
    img_path.mkdir(exist_ok=True)
    stats = {}
    for c in range(1, 11, 1):
        for r in range(5, 21, 5):
            for use_5d in [True, False]:
                scale = 0.5
                out = "{}_r{}_c{}_5d{}_scale{}.jpg".format(image.split('/')[-1][:-4], r, c, use_5d, scale)
                logging.info(out)
                start = time.time()
                peaks, out_image = image_segment(image, r, use_5d=use_5d, scale=scale, c=c)
                stats[out] = (peaks, time.time() - start)
                cv2.imwrite(str(img_path / out), out_image)
    (img_path / "stats.json").write_text(json.dumps(stats))
    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Executes meanshift.")
    parser.add_argument("image", help="The RGB image to process.")
    parser.add_argument("-r", default=10, type=float, help="The neighbourhood range to consider.")
    parser.add_argument("-c", default=10, type=float, help="Optimization parameter for path collecting cpts.")
    parser.add_argument("-scale", default=1, type=float, help="Rescaling factor.")
    parser.add_argument("--use_5d", action='store_true', help="Use x and y position for meanshift.")
    parser.add_argument("--report", action='store_true', help="Generates the images seen in the report, may take a very long time !")
    args = vars(parser.parse_args())


    def exec_meanshift(args):
        logging.info("Calculating meanshift for {}".format(args))
        start = time.time()
        peaks, image = image_segment(**args)
        end = time.time()
        logging.info("Finished in {} s, found {} peaks".format(end - start, peaks))
        cv2.imshow(str(args), image)
        cv2.waitKey()


    if args['report']:
        generate_report_images(args['image'])
    else:
        exec_meanshift(args)
