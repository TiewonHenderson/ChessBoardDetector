import sys
import cv2
import numpy as np
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import HoughTransform as ht
from math import sin,cos, log10
from sklearn.cluster import DBSCAN


def cv_sort_rho(lines, defined_gap, image_shape):
    """
    Sort and groups lines by their rho (distance to the original, top left of images)
    Uses coefficient variance in order to get gap consistency

    This also uses vanishing point concept.
    The group of lines must intersect at one point which is the vanishing point.

    Minimum gap expectation:
    Chessboard >= 1/4 * (image area)
    Horizontal tilt >= 25 degree (90 degree == same angle as chessboard, cannot see grid)

    min(W, H) / 2 / 8 * sin(25°)
    final formula = 0.0264 × min(W, H)
    :param lines: unpacked 2D array of rho,theta values
    :param image_shape: [height, width] of the image
    :return: A 2D list that are grouped based off gap between them,
            with a 1D array representing gap average of each group of lines
    """

def mid_point(line, image_shape):
    """
    Gets a line that generally perpendicular to the inputted line and the point that intersection
    that is around the midpoint of the inputted line in respect of the image shape bounds

    Using ρ=xcosθ+ysinθ formula
    as points goes down, y actually increases
    :param line:
    :return:
    """
    h, w = image_shape
    x_bound = [0, w]
    y_bound = [0, h]



def most_common_gap(gaps_list):
    """
    Bucket sort to get most common gap
    :param gaps_list: A 2D list of gaps, each list representing gaps found from each line
    :return:
    """
    buckets = {}
    for i, scanned in enumerate(gaps_list):
        for j, gap in enumerate(scanned):
            if gap in buckets:
                buckets[gap][0] += 1
                buckets[gap][1].add(i)
            else:
                buckets[gap] = [1, {i, i + j + 1}]
    common_gap = max(buckets, key=lambda k: buckets[k][0])
    return common_gap, buckets[common_gap][1]


def get_starting_lines(lines, image_shape):
    """
    This function runs through the lines inputed (assuming its a theta cluster) to find
    consistent gaps with brute force.
    :param lines:
    :param image_shape:
    :return:
    """
    # Starting lines must be output a len > 8, and < 10.
    if len(lines) < 8 and is_starting:
        return None
    copy_line = [fg.normalize_line(line) for line in lines]
    copy_line.sort(key=lambda x: x[0])
    h, w = image_shape
    min_gap = min(h, w) * 0.0264

    curr_line = 0
    scan_line = 1
    all_gaps = []
    while curr_line < len(copy_line):
        line0 = copy_line[curr_line]
        scan_line = curr_line + 1
        curr_gaps = []
        while scan_line < len(copy_line):
            line1 = copy_line[scan_line]
            gap = abs(line0[0] - line1[0])

            # Merge line if gap is way too small
            if gap < min_gap / 4:
                avg_rho = (line0[0] + line1[0]) / 2
                avg_theta = (line0[1] + line1[1]) / 2
                current_line = [avg_rho, avg_theta]
                # Replace the two lines with the mean line
                # Preferably this only happens 1
                copy_line.pop(scan_line)
                copy_line.pop(curr_line)
                copy_line.insert(curr_line, current_line)
                curr_gaps.clear()
                line0 = current_line
                continue
            # Rounds to the third digit of the given number
            curr_gaps.append(round(gap, -(int(log10(abs(gap))))))
            scan_line += 1
        all_gaps.append(curr_gaps)
        curr_line += 1
    print("gaps:")
    for gaps in all_gaps:
        print(gaps)
    best_gap, lines_with_gap = most_common_gap(all_gaps)
    return_lines = []
    for i in lines_with_gap:
        # Ideally, common gaps shouldnt return an index out of bounds
        if i >= len(copy_line):
            continue
        return_lines.append(copy_line[i])
    return return_lines


def got_enough_lines(labels):
    """
    Simple check to see if a cluster has 9 or more lines to be a chessboard grid
    :param labels: Labels from a clustering algorithm
    :return: boolean to indicate if enough lines were found for a label
    """
    label_dict = {}
    for x in labels:
        if x in label_dict:
            label_dict[x] += 1
        elif x != -1:
            label_dict[x] = 1
    for key in label_dict:
        if label_dict[key] >= 9:
            return True
    return False


def orientation_abs(theta):
    """
    Returns angle's absolute orientation in [0, π/2] (like abs for orientation)
    """
    theta = theta % np.pi  # bring to [0, π)
    return min(theta, np.pi - theta)


def dbscan_cluster_lines(lines, image_shape):
    """
        Clusters using DBSCAN, then labels each cluster by it's average theta

        DBSCAN is not good with a lot of points, if using small threshold settings,
        Make sure to use small epsilon value

            radians       degree
            0.4       =   22.9183
            0.3       =   17.1887
            0.2       =   11.4592
            0.1       =   5.72958
            0.05      =   2.864789
            0.02      =   1.145916

        :param lines: unpacked 2D array of rho,theta values
        :param image_shape: [height, width] of the image
    """
    if len(lines) == 0:
        return {}
    h, w = image_shape
    line_array = np.array([
        [sin(orientation_abs(theta)), cos(orientation_abs(theta))]
        for rho, theta in lines  # Use lines directly
    ])
    """
    DBSCAN expects a 2d np array representing points
    """
    eps = 0.001         # radians = 0.05729578 degrees
    max_eps = 0.1       # radians = 5.72958 degrees
    got_9_lines = False
    labels = []
    while not got_9_lines and eps < max_eps:
        clustering = DBSCAN(eps=eps, min_samples=2, algorithm='ball_tree').fit(line_array)  # tune eps
        labels = clustering.labels_
        got_9_lines = got_enough_lines(labels)
        if not got_9_lines:
            eps *= 2
    print(labels)
    return label_to_cluster(lines, labels)


def label_to_cluster(lines, labels):
    """
    :param lines: unpacked lines from houghline transformation
    :param labels: Labels from a clustering algorithm to assign each line
    :return: A list representing clusters of lines with similar thetas (3d list)
            Another 2d list representing outliers
    """
    clusters = []
    left_overs = []
    # lines is 3d of 2d representing [[rho,theta]...]
    for x in range(len(lines)):
        label = labels[x]
        # Outliers wont be considered
        if label == -1:
            left_overs.append(lines[x])
            continue
        # Add more empty groups
        while len(clusters) <= label:
            clusters.append([])
        clusters[label].append(lines[x])

    return clusters, left_overs
