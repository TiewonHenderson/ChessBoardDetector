import sys
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from math import sin,cos
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import HoughTransform as ht
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import cv_filter_groups as cvfg


def closest_corner(corners, line):
    """
    Formula to see how close a corner is in respect to the line
    x * np.cos(theta) + y * np.sin(theta) - rho ~ 0 (closer to 0, closer to line)
    due to
    rho = x * np.cos(theta) + y * np.sin(theta)

    Revised old code by AI
    :param corners: A 2D list of points representing cartesian points
    :param line: A single [rho, theta] line
    :return:
    """
    if len(corners) < 1 or line is None:
        return None

    rho, theta = line
    get_dist = lambda x,y: abs(x * cos(theta) + y * sin(theta) - rho)

    corner_distances = []
    for corner in corners:
        x, y = corner
        dist = get_dist(x, y)
        corner_distances.append([dist, corner])

    # Sort by distance and take top 8 corners
    corner_distances.sort(key=lambda x: x[0])
    top_closest = corner_distances[:8]
    return top_closest


def intersect_verification(sect_list, corner, threshold=10):
    """
    Verifies intersection if they're close enough to a corner point
    :param sect_list:
    :param corner:
    :param threshold: 10 pixel distance is the default offset of points to be considered the same
    :return:
    """
    verified = set()
    max_dist = threshold ** 2
    for pt1 in sect_list:
        if pt1 is None:
            continue
        x_1, y_1 = pt1
        # Use of dist^2 = (x_2 - x_1)^2 + (y_2 - y_1)^2 to avoid sqrt operations
        min_dist_pt = (sys.maxsize, None)
        for pt2 in corner:
            if pt2 is None:
                continue
            x_2, y_2 = pt2
            dist_sqre = (x_2 - x_1)**2 + (y_2 - y_1)**2
            if min_dist_pt[0] > dist_sqre:
                min_dist_pt = (dist_sqre, [x_2, y_2])
        if min_dist_pt[0] < max_dist:
            verified.add(tuple(min_dist_pt[1]))
    return verified


def hough_line_intersect(line, point):
    """
    :param line: [rho, theta]
    :param point: [x, y]
    :param tolerance: Represented by total pixels off to be considered intersecting a corner
    :return: distance to consider intersect
    """
    rho, theta = line
    x, y = point
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    distance = abs(x * cos_t + y * sin_t - rho)
    return distance


def line_interpolate_by_corner(group, corners, direction, threshold=10):
    """
    Interpolate lines by checking if they come close to intersecting harris corner points
    Makes sure gap consistency is similar to those found in group 1 and 2
    :param group:
    :param corners:
    :param direction:
    :param threshold:
    :return:
    """
    if 3 < len(group) < 9:
        """
        Applies near same logic within Remove_outliers
        """
        if direction == 0 or direction == 180:
            lines_c = [(rho, (theta + np.pi / 4) % np.pi) for rho, theta in group]

    return None


