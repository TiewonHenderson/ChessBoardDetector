import sys
import cv2
import math
import numpy as np
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import HoughTransform as ht
from sklearn.cluster import DBSCAN


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
    Bucket sort to get most common gap.
    Lines that created the gaps are saved by index
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
                # Dict represents the lines used to create the gap
                buckets[gap] = [1, {i, i + j + 1}]
    common_gap = max(buckets, key=lambda k: buckets[k][0])
    return common_gap, buckets[common_gap][1]


def most_common_point(points, tolerance=15):
    """
    GPT suggested using cluster to find most common point
    :param points:
    :param tolerance:
    :return:
    """
    points_np = np.array(points.copy())  # Shape (N, 2)

    # Cluster with fixed radius (tolerance)
    db = DBSCAN(eps=tolerance, min_samples=1).fit(points_np)
    labels, counts = np.unique(db.labels_, return_counts=True)

    # Find the label with the most points
    most_common_label = labels[np.argmax(counts)]

    # Extract all points in that cluster
    common_cluster = points_np[db.labels_ == most_common_label]
    indexes = []
    for i, label in enumerate(db.labels_):
        if label == most_common_label:
            indexes.append(points[i])
    return set(indexes), len(common_cluster)


def has_vanishing_point(lines, image_shape):
    """
    Checks vanishing points between lines, lines with NONE is usually a good sign
    If VP is relatively close to the image, its a high chance its not part of the grid

    If amount of None >= 4, replace all lines with those lines instead
    If amount of None < 4, but points > 3x(h,w), count those as good VPs
    :param lines:
    :param image_shape:
    :return:
    """
    copy_lines = lines.copy()
    vp_list = []
    h, w = image_shape
    center_w = [w * (1/4), w * (3/4)]
    center_h = [h * (1/4), h * (3/4)]
    i = 0
    # Loops appends all intersection points (considered Vanishing points)
    while i < len(copy_lines):
        rho, theta = copy_lines[i]
        vp_list.append([])
        j = 0
        while j < len(copy_lines):
            if i != j:
                #intersection
                sect = fg.intersection_polar_lines(copy_lines[i],
                                                   copy_lines[j])
                if sect is not None:
                    # If vp is within the center, that makes no sense, don't append that
                    if sect[0] > center_w[0] and sect[0] < center_w[1]:
                        if sect[1] > center_h[0] and sect[1] < center_h[1]:
                            vp_list[i].append([])
                            j += 1
                            continue
                vp_list[i].append(sect)
            else:
                vp_list[i].append([])
            j += 1
        i += 1

    # for x in vp_list:
    #     print(x)

    # If there are 4 or more VPs that are none or extremely outside the image
    # We can consider those lines good
    min_needed = 4
    good_lines = []
    shared_vp_lines = []
    for i, line_vp in enumerate(vp_list):
        good_vps_index = []
        escape = False
        for j, vp in enumerate(line_vp):
            if i == j:
                good_vps_index.append(i)
                continue
            # How an image works is, even negative is considered outside of image
            # Checks if the vanish point is outside 3 times the dimensions of the image
            if (vp is None or
               (len(vp) == 2 and ((vp[0] > (3 * w) and vp[1] > (3 * h)) or
                                  (vp[0] < (-3 * w) and vp[1] < (-3 * h))))
                ):
                good_vps_index.append(j)
        # near parallel lines were found
        if len(good_vps_index) >= min_needed:
            escape = True
            for index in good_vps_index:
                good_lines.append(copy_lines[index])
        if escape:
            break
    # no parallel lines were found
    # Multi-sharing VP lines needs to run
    if len(good_lines) == 0:
        candidates, amt = most_common_point(vp_list)
        for index in candidates:
            good_lines.append(copy_lines[index])

    return good_lines
