import sys
import cv2
import numpy as np
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import HoughTransform as ht
from math import sin,cos, log10
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


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
    final formula = 0.0264 * min(W, H)
    :param lines: unpacked 2D array of rho,theta values
    :param image_shape: [height, width] of the image
    :return: A 2D list that are grouped based off gap between them,
            with a 1D array representing gap average of each group of lines
    """
    # Not enough lines to be considered a grid
    if len(lines) <= 4:
        return [lines], []
    rhos = [fg.normalize_line(line)[0] for line in lines]
    rhos.sort()
    min_gap = 0.0264 * min(W, H)

    # Gives rho difference as:
    # [x1-x0, x2-x1, ..., x(n-1) - x(n-2)]
    gaps = np.diff(rho)
    # [gap1/gap0, gap2/gap1, ..., gap(n-1)/gap(n-2)]
    ratios = gaps[1:] / gaps[:-1]


def check_imbalance(elements, new_element=None):
    """
    Usees a combination of Coefficient of Variation (CV)
    and
    Median Absolute Deviation (MAD)
    To calculate if the new_gap is an outlier and shouldnt be kept

    CV is powerful in order to spike in value when variance happens, i.e outlier(s)
    However falls short if the outlier is still close within the gap values

    MAD will be able to confirm CV passing the new_gap as low variance
    MAD reference:
    https://www.statology.org/modified-z-score/

    Summary (Good explaination from GPT):
    CV gives you group-level insight — "Are the gaps consistent overall?"
    MAD gives you point-level robustness — "Is this new gap too far from the typical ones?"

    mad can only be 0 when:
    [45, 55, 55, 55, 999]
    [10, 0, 0, 0, 944]
    :param gaps:
    :param new_gap:
    :return: if new_element is a new element:
             0 = Not part of the gap group
             1 = Is a near multiple of the gap, could be useful for line interpolation
             2 = Is part of the line group
             if new_element is none:
             0 = has outliers
             1 = consistent throughout
    """
    if len(elements) < 3:
        return 0
    elem_copy = np.array(elements)
    if new_element is not None:
        elem_copy = np.append(elem_copy, new_element)
    # 2 or less lines cannot form > 1 gap
    # Find MAD, we can treat it like a general standard deviation
    median = np.median(elem_copy)
    dist_med = np.abs(elem_copy - median)
    mad = np.median(dist_med)

    if mad == 0:
        # mad being 0 means more then half of the gaps are the same
        mad = np.median(np.unique(dist_med))
        if mad == 0:
            # mad still being 0 means all gaps are equal
            # Need to calculate CV to confirm "new_gap"
            mad = None

    if mad is not None and mad != 0 and new_element is not None:
        mod_z_score = 0.6745 * (new_element - median) / mad
        # Apparantly 3.5 std dev is a good indication of outlier
        if mod_z_score > 3.5:
            return 0

    # MAD passes, check CV
    std_dev = np.std(elem_copy)
    mean = np.mean(elem_copy)

    if mean < 0.0001:
        # CV cannot be calculate, but MAD passed, keep new gap
        return 2

    cv = std_dev / mean
    if cv < 0.1:
        return 2
    elif new_element is not None:
        # check if gap is a multiple of mean
        ratio = new_gap/mean
        scale = round(ratio)
        # Only allow 1-2 lines to be interpolated
        if 2 <= scale <= 4 and abs(scale - ratio) < 0.2:
            return 1
    return 0


def got_enough_lines(labels):
    """
    Simple check to see if a cluster has 9 or more lines to be a chessboard grid
    :param labels: Labels from a clustering algorithm
    :return: amount of groups with len >= 9 lines as int
    """
    label_dict = {}
    amount_of_groups = 0
    for x in labels:
        if x in label_dict:
            label_dict[x] += 1
        elif x != -1:
            label_dict[x] = 1
    for key in label_dict:
        if label_dict[key] >= 9:
            amount_of_groups += 1
    return amount_of_groups


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
        [theta % np.pi]
        for rho, theta in lines  # Use lines directly
    ])
    """
    DBSCAN expects a 2d np array representing points
    """
    eps = 0.001         # radians = 0.05729578 degrees
    max_eps = 0.174533  # radians = 10 degrees
    got_9_lines = 0
    labels = []
    while got_9_lines < 2 and eps < max_eps:
        clustering = DBSCAN(eps=eps, min_samples=2, algorithm='ball_tree').fit(line_array)  # tune eps
        labels = clustering.labels_
        got_9_lines = got_enough_lines(labels)
        eps += np.deg2rad(1)

    clusters, left_overs, theta_labels = label_to_cluster(lines, labels, eps)
    return clusters, left_overs, theta_labels


def kmeans_cluster_lines(lines, image_shape):
    """
    Uses 1D KMeans to cluster theta (normalized)
    :param lines: unpacked 2D array of rho,theta values
    :return:
    """
    line_array = lines.copy()
    line_array = np.array([
        [sin(2 * theta), cos(2 * theta)]
        for _, theta in line_array
    ])
    labels = KMeans(n_clusters=2, n_init='auto').fit_predict(line_array)
    print(labels)
    max_eps = 0.174533
    clusters, _ = label_to_cluster(lines, labels, max_eps)
    return clusters
    # clusters, left_overs, theta_labels = label_to_cluster(lines, labels, max_eps)
    # return clusters, left_overs, theta_labels


def label_to_cluster(lines, labels, eps):
    """
    :param lines: unpacked lines from houghline transformation
    :param labels: Labels from a clustering algorithm to assign each line
    :return: A list representing clusters of lines with similar thetas (3d list)
            Another 2d list representing outliers
    """
    clusters = []
    left_overs = []
    # amt_sum = []
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
        #     amt_sum.append([0,0])
        # amt_sum[label][0] += 1
        # amt_sum[label][1] += lines[x][1]
        clusters[label].append(lines[x])

    return clusters, left_overs
    # print(amt_sum)
    # c_index = 0
    # means = []
    # while c_index < len(clusters):
    #     amt, sum = amt_sum[c_index]
    #     average = sum / amt
    #     # Should not consider less then 3 lines for a 8x8 grid
    #     if len(clusters[c_index]) < 3:
    #         clusters.pop(c_index)
    #         amt_sum.pop(c_index)
    #         continue
    #     # Combine like theta clusters
    #     skip = False
    #     for i, m in enumerate(means):
    #         if abs(average - m) < eps:
    #             clusters[i].extend(clusters[c_index])
    #             clusters.pop(c_index)
    #             amt_sum.pop(c_index)
    #             skip = True
    #             break
    #     if skip:
    #         continue
    #     means.append(average)
    #     c_index += 1
    #
    # return clusters, left_overs, means
