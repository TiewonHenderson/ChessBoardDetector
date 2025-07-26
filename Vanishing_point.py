import sys
import cv2
import math
import numpy as np
from collections import Counter
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import HoughTransform as ht
from ChessBoardDetector import cv_filter_groups as cvfg
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression


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
    :param points: Expects 3D
    :param tolerance: Area of pixel toleration (how much offset area to be considered a shared VP)
    :return: Gets a list of indices for most common intersection points (vanishing points) and the angle direction
             is represented as a unit circle (0 degree is left, 90 degree is vertical top)
    """
    if points is None or len(points) == 0:
        return None, None

    mask = []
    valid_points = []
    # Checks points is a list of (sect, direction)
    if isinstance(points[0], tuple):
        for i, result in enumerate(points):
            sect, direction = result
            if sect is None or len(sect) == 0:
                continue
            valid_points.append(sect)
            mask.append(i)
    else:
        # If points is a 3d list
        for i in range(len(points)):
            for j in range(len(points[i])):
                if points[i][j]:
                    valid_points.append(points[i][j])
                    mask.append(j)
    if len(valid_points) == 0:
        return None, None
    points_np = np.array(valid_points)  # Shape (N, 2)

    # Cluster with fixed radius (tolerance)
    db = DBSCAN(eps=tolerance, min_samples=1).fit(points_np)
    labels, counts = np.unique(db.labels_, return_counts=True)

    """
    The commented code below gets top 5 VP groups instead of MAX
    """
    # Sort by count (descending) and get top 5
    # sorted_indices = np.argsort(counts)[::-1][:5]

    # Extract all points within each cluster
    # results = []
    # for idx in sorted_indices:
    #     label = labels[idx]
    #     count = counts[idx]
    #
    #     # Get indexes for this cluster
    #     indexes = [mask[i] for i, l in enumerate(db.labels_) if l == label]
    #     results.append((set(indexes), count))

    max_index = np.argmax(counts)

    # Extract all points within each cluster
    indexes = [mask[i] for i, l in enumerate(db.labels_) if l == max_index]

    # Extract most dominant direction of VP
    all_points = [points[i] for i in indexes]
    directions = [direction for _, direction in all_points]
    direction_counts = Counter(directions)
    most_common_direction, count = direction_counts.most_common(1)[0]

    return indexes, most_common_direction


def get_all_intersections(lines, image_shape):
    """
    Gets intersections represented as a 2 element tuple:
    (point, angle direction)
    If point is None, that represents parallel lines between the two compared lines
    (or too large of an intersection point)

    angle direction represents where the VP is to relative to the center of the image
    degree
    0           Right
    45          Top right
    90          Up
    135         Top left
    180         Left
    225         Bottom left
    270         Down
    315         Bottom right
    :param lines:
    :param image_shape:
    :return:
    """
    vp_list = [[] for i in range(len(lines))]
    h, w = image_shape
    center_w = [w * (1/4), w * (3/4)]
    center_h = [h * (1/4), h * (3/4)]
    i = 0
    # Loops appends all intersection points (considered Vanishing points)
    while i < len(lines):
        rho, theta = lines[i]
        j = i
        while j < len(lines) - 1:
            if i != j:
                #intersection
                sect, direction = fg.intersection_polar_lines(lines[i],
                                                              lines[j],
                                                              need_dir=True)
                if sect is not None:
                    # If vp is within the center, that makes no sense, don't append that
                    if center_w[0] < sect[0] < center_w[1]:
                        if center_h[0] < sect[1] < center_h[1]:
                            vp_list[i].append(([], None))
                            vp_list[j].append(([], None))
                            j += 1
                            continue
                    # Goes far outside the image, consider as inf VP
                    if abs(sect[0]) > (3 * w) or abs(sect[1]) > (3 * h):
                        vp_list[i].append((None, direction))
                        vp_list[j].append((None, direction))
                        j += 1
                        continue
                vp_list[i].append((sect, direction))
                vp_list[j].append((sect, direction))
            else:
                vp_list[i].append(([], None))
                vp_list[j].append(([], None))
            j += 1
        i += 1
    return vp_list


def finalize_groups(group_dict, lines):
    """
    Used after has_vanishing_point is run, as it outputs a lot of duplicate and similar groups
    This function will combine/remove duplicates
    :param group_dict: A dictionary of key: line indices, value = (lines, direction)
    :param lines: The original list of lines
    :return: A list of tuples (lines, direction), maybe add average gap but not sure
    """

    def check_similarity(g1, g2, dir1, dir2):
        """
        Checks two groups for intersecting values (intersection of lists)
        :param g1: Group 1 of lines (1d collection of elements, prefer int)
        :param g2: Group 2 of lines
        :param dir1: Direction of VP for g1 (int)
        :param dir2: Direction of VP for g2 (int)
        :return: True, False to combine the groups
        """
        if dir1 != dir2:
            return False
        intersection = np.intersect1d(np.array(g1), np.array(g2))
        # True if (one group is completely in another) or
        # The groups overlap over 1/2 of elements
        return len(intersection) >= min(len(g1), len(g2)) or len(intersection) >= max(len(g1), len(g2))/2

    # Removes absolute duplicates by set collections
    unique_values = list(set((tuple(list), dir) for list, dir in group_dict.values()))

    # Need of combining groups with majority shared lines
    i = 0
    while i < len(unique_values):
        g1, dir1 = unique_values[i]
        j = i + 1
        while j < len(unique_values):
            g2, dir2 = unique_values[j]
            # Combines and removes the i+1 index element without incrementing i
            if check_similarity(g1, g2, dir1, dir2):
                combined = g1 + g2
                unique_values[i] = (tuple(set(combined)), dir1)
                # Update the old group
                g1, dir1 = unique_values[i]
                unique_values.pop(j)
                continue
            j += 1
        unique_values[i] = (list(set(g1)), dir1)
        i += 1

    return unique_values


def has_vanishing_point(lines, image_shape):
    """
    Checks vanishing points between lines, lines with NONE is usually a good sign
    If VP is relatively close to the image, its a high chance its not part of the grid

    If amount of None >= 4, replace all lines with those lines instead
    If amount of None < 4, but points > 3x(h,w), count those as good VPs

    Average operations: N^3
    :param lines:
    :param image_shape:
    :return:
    """
    # List of list of points
    vp_list = get_all_intersections(lines, image_shape)
    h, w = image_shape
    center_w = [w * (1/4), w * (3/4)]
    center_h = [h * (1/4), h * (3/4)]

    # for x in vp_list:
    #     print(x)

    # If there are 4 or more VPs that are none or extremely outside the image
    # We can consider those lines good
    min_needed = 4
    used = set()
    good_lines = {}
    shared_vp_lines = []
    for i, line_vp in enumerate(vp_list):
        # Line is already used, don't search again
        if i in used:
            continue
        good_vps_index = {}
        escape = False
        """
        This section is spilt into two sections:
        1) Common vanishing point thats nears infinite (3 * dimensions of image)
        2) Common intersection AREA
        """
        for j, vp in enumerate(line_vp):
            sect, direction = vp
            if i == j:
                continue
            # How an image works is, even negative is considered outside of image
            # Checks if the vanish point is outside 3 times the dimensions of the image
            if (sect is None or
               (len(sect) == 2 and abs(sect[0]) > (3 * w) and abs(sect[1]) > (3 * h))):
                # Lines need to have same direction VPs to be accepted
                # Lines can only form two VPs
                good_vps_index.setdefault(direction, {j}).add(j)
        # Add own line to all groups
        for key in good_vps_index:
            good_vps_index[key].add(i)
        # near parallel lines were found
        # key is direction, value if indexes of lines that also also vps infinitely in that direction
        for key, value in good_vps_index.items():
            if len(value) >= min_needed:
                escape = True
                line_group = []
                for index in value:
                    line_group.append(index)
                good_lines[i] = (line_group, key)
            if escape:
                break

        # no parallel lines were found for current line
        # Multi-sharing VP lines needs to run
        if i not in good_lines:
            results, direction = most_common_point(vp_list[i], min(h,w)/10)
            if results is None:
                continue
            if len(results) < 3 or len(results) > 10:
                continue
            used.add(index for index in results)
            for index in results:
                good_lines[i] = (results, direction)

    return finalize_groups(good_lines, lines)


def enough_similar_theta(group):
    """
    Static check for easy parallel check, accept in same theta
    Mainly for VPs that not directly away from the camera.
    :param group: 2D list of rho, theta lines
    :return: Edits group by reference, returns same object being passed through
    """
    buckets = cvfg.bucket_sort([l[1] for l in group])
    most_theta = max(buckets, key=lambda k: buckets[k])
    if buckets[most_theta] >= len(group):
        # If there is a dominant theta, remove all other thetas
        i = 0
        while i < len(group):
            if abs(group[i][1] - most_theta) > 0.04:
                group.pop(i)
                continue
            i += 1
    return group


def find_consistent_theta_trend(thetas, min_samples=4, residual_threshold=0.05):
    """
    GPT generated in order to get the best rate of theta changes in the cluster
    Uses RANSAC to find a subset of thetas that follow a consistent linear trend.
    :param thetas: List or 1D array of theta values (float)
    :param min_samples: Minimum points needed to assert a model is valid
    :param residual_threshold: Max allowed error for a point to be an inlier
    :return: slope of trend, intercept, inlier thetas, indices
    """
    if len(thetas) < min_samples:
        return None, None, [], []

    x = np.arange(len(thetas)).reshape(-1, 1)
    y = np.array(thetas)

    model = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=1000
    )
    try:
        model.fit(x, y)
    except ValueError as e:
        print("RANSAC model likely found nothing")
        return None, None, [], []

    inlier_mask = model.inlier_mask_

    inlier_indices = np.where(inlier_mask)[0]
    inlier_thetas = y[inlier_mask]

    slope = model.estimator_.coef_[0]
    intercept = model.estimator_.intercept_

    return slope, intercept, inlier_thetas, inlier_indices


