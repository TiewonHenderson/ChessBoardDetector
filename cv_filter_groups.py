import sys
import cv2
import numpy as np
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import HoughTransform as ht
from ChessBoardDetector import Chessboard_detection as cd
from ChessBoardDetector import Remove_outliers as ro
from math import sin,cos
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


def bucket_sort(elements, threshold=0.4):
    """
    Bucket sort to get max quantity of item
    :param elements: A 1d list of elements
    :param threshold: How close the bucket needs to be to be accepted in the same bucket
    :return: A dict of elements counted with 0.04 difference threshold
    """
    # gets priority theta by bucket sort
    bucket = {}
    for x in elements:
        if not bucket:
            bucket[round(float(x), 5)] = 1
        # GPT generated, going through all keys to get the closest one to current theta
        closest_key = min(bucket.keys(), key=lambda k: abs(k - x))
        # Around 2 degrees
        if abs(closest_key - x) < threshold:
            bucket[closest_key] += 1
        else:
            bucket[round(float(x), 5)] = 1
    return bucket


def get_intervals(gap_list, gap_stats, max_gap, min_gap):
    """
    Function meant to detect extreme outlier gaps with mad and cv, mad
    :param gap_list:
    :param gap_stats:
    :param max_gap:
    :param min_gap:
    :return:
    """
    med, mean, mad, dist_med, std_dev = gap_stats
    mad_scores = check_MAD(gap_list, get_score=True)
    intervals = [[]]
    i = 0
    combine_last = False
    # Shows the previous gap was more then min_gap
    prev_gap = True

    def can_add(i, intervals=intervals):
        if len(intervals[-1]) == 0 or i not in intervals[-1][-1]:
            return True
        return False

    while i < len(gap_list):
        # In theory we can use gap_list again, but it requires a lot of operations to keep track
        # Since gap_list is so dynamic its not worth using a static one
        gap = round(gap_list[i], 4)
        if gap > max_gap or (mad_scores[i] > 2.5 and gap > med):
            # Likely different interval if lines with same vp
            z_score = abs((gap - mean) / std_dev)
            if can_add(i):
                intervals[-1].append([i])
            if z_score > 2:
                intervals.append([])
        elif gap < min_gap:
            # Overlapping line belong to the same SET
            # Same interval line belong to the same LIST
            if prev_gap or len(intervals[-1]) == 0:
                intervals[-1].append([])
            if can_add(i):
                intervals[-1][-1].append(i)
            intervals[-1][-1].append(i + 1)
            if i == len(gap_list) - 1:
                combine_last = True
            prev_gap = False
            i += 1
            continue
        else:
            if can_add(i):
                intervals[-1].append([i])
            combine_last = False
            prev_gap = True
        prev_gap = True
        i += 1
    if not combine_last:
        intervals[-1].append([len(gap_list)])
    return intervals


def cv_clean_lines(lines, direction, image_shape, image=None):
    """
    gap and lines structure:
    lines = [0 , 1 , 2 , 3 , 4..]
    gaps = [   0,  1,  2,  3..]

    Minimum gap expectation:
    Chessboard >= 1/4 * (image area)
    8 Grids = 1/8 gap
    Horizontal tilt >= 25 degree (0 degree == same angle as chessboard, cannot see grid)

    1/4 * 1/8 * sin(25degree) = 0.0132
    The min gap expected is: 0.0132 * min(W, H)
    The max gap would definitely be 0.15 * max(W, H)

    Uses a perpendicular line in respect to given direction in order to get
    better consistent gap calculation

    :param lines: unpacked 2D array of rho,theta values
    :param direction: The direction the vp is in respect to the middle of the image
    :param image_shape: [height, width] of the image
    :return: A 2D list that are grouped based off gap between them,
             with a 1D array representing gap average of each group of lines

             !can return None, None if group of lines were determined as bad!
    """

    def get_gaps(sect_list):
        """Gets gaps between the intersection points in its order"""
        sect_array = np.array(sect_list)  # convert list of points to numpy array
        return np.linalg.norm(sect_array[1:] - sect_array[:-1], axis=1)

    # Not enough lines to be considered a grid
    if len(lines) < 4:
        return None
    h, w = image_shape
    min_gap = 0.0132 * min(h, w)
    max_gap = 0.15 * max(h, w)
    cx = w / 2
    cy = h / 2
    perd_theta = (np.deg2rad(direction) + np.pi / 2) % np.pi
    # Formula given by GPT to get perpendicular line at center of image
    perd_line = (cx * np.cos(perd_theta) + cy * np.sin(perd_theta), perd_theta)

    # Gets intersections of each line to the perd_line, i to keep lines index intact
    # Sorts by x then y
    sect_list = [(i, fg.intersection_polar_lines(perd_line, l)) for i, l in enumerate(lines)]
    sect_list = sorted(sect_list, key=lambda p: (p[1][0], p[1][1]))

    # Aligns lines to the intersections saved in sect_list
    lines = [lines[item[0]] for item in sect_list]
    sect_list = [item[1] for item in sect_list]
    gap_list = get_gaps(sect_list)

    # MAD and CV only on gap_list
    med = np.median(gap_list)
    if med < 0.1:
        # If majority of the gap is < 0.1, this groups is definitely bad
        return None, None
    mad, dist_med = check_MAD(gap_list, get_mad=True)
    std_dev = np.std(np.array(gap_list))
    mean = np.mean(np.array(gap_list))
    gap_stats = (med, mean, mad, dist_med, std_dev)

    copy_line = lines.copy()
    copy_line.append(perd_line)
    print("sect_list", sect_list)
    print("gap_list", gap_list)
    print("lines", lines)
    print("stats", gap_stats)
    print("min, max", min_gap, max_gap)
    cd.find_exact_line(image, copy_line, index=-1, green=True)

    # Only checks biggest interval
    intervals = max(get_intervals(gap_list, gap_stats, max_gap, min_gap), key=len, default=[])
    print("intervals", intervals)
    if len(intervals) >= 2:

        got_lines = None
        if direction == 0 or direction == 180:
            got_lines = ro.remove_outlier_away(intervals, lines)
        elif direction == 90:
            got_lines = ro.remove_outlier_parallel(intervals, lines)
        else:
            got_lines = ro.remove_outlier_norm(intervals, lines, direction)

        print("got lines", got_lines)
        cd.find_exact_line(image, got_lines, index=-1, green=False)
        # ht.curve_fit_lines(intervals, lines, direction)

        return got_lines

    print("Filtered out all important lines, NO group found")
    return None


def cv_check_grid(group1, group2):
    """
    Scoring:
    1) Gap average in group1 and group2 is relatively the same
    2) Evenly spaced gap between each intersection
    3) Amount of lines


    :param group1:
    :param group2:
    :return:
    """
    # Intersection matrix
    sect_matrix = []
    # group 2 would be column, group 1 would be rows
    for j, line_j in enumerate(group2):
        sect_matrix.append([])
        for line_i in group1:
            sect, _ = intersection_polar_lines(line_i, line_j)
            sect_matrix[j].append(sect)


def check_MAD(elements, std_dev=3.5, get_mad=False, get_score=False):
    """
        Median Absolute Deviation (MAD)
        To calculate if the new_gap is an outlier and shouldnt be kept
        MAD reference:
        https://www.statology.org/modified-z-score/
    :param elements: a list of elements
    :param std_dev: threshold of deviation of mad to be considered an outlier
    :param get_mad: Just make this function return mad and deviation elements instead of outliers
    :param get_score: Gets all the mod_z_score instead of an evaluation
    :return:
    """
    if len(elements) < 3:
        return []
    elem_copy = elements
    if not isinstance(elements, np.ndarray):
        elem_copy = np.array(elements)
    # 2 or less lines cannot form > 1 gap
    # Find MAD, we can treat it like a general standard deviation
    median = np.median(elem_copy)
    dist_med = np.abs(elem_copy - median)
    mad = np.median(dist_med)

    if get_mad:
        return mad, dist_med

    if mad == 0:
        # mad being 0 means more then half of the gaps are the same
        mad = np.median(np.unique(dist_med))
        if mad == 0:
            return [0 for i in range(len(elements))]
    outliers = []
    for i, x in enumerate(elements):
        mod_z_score = 0.6745 * (x - median) / mad
        # Return all z_score instead
        if get_score:
            outliers.append(mod_z_score)
            continue
        # Apparantly 3.5 std dev is a good indication of outlier
        elif mod_z_score > std_dev:
            outliers.append(i)

    return outliers


def check_imbalance(elements, new_element=None, index=None, get_score=False):
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
    :param elements: A 1D array of numerical value
    :param new_element: A new value representing the same values in elements
    :param index: use index of inside elements and check if that is an outlier
    :param get_score: Get the CV and MAD scores of the new_element within elements instead
                      (only works with index param not being None)
    :return: if new_element is a new element:
             0 = Not part of the gap group
             1 = Is a near multiple of the gap, could be useful for line interpolation
             2 = Is part of the line group
             if new_element is none:
             0 = has outliers
             1 = consistent throughout
    """
    if not elements or len(elements) < 3:
        if get_score:
            return None, None
        return 0
    elem_copy = np.array(elements)
    test_element = None
    outliers = None
    if new_element is not None:
        elem_copy = np.append(elements, new_element)
        test_element = new_element
    # Run mad function for score or outliers themselves
    if get_score:
        outliers = check_MAD(elem_copy, get_score=True)[index]
    else:
        outliers = check_MAD(elem_copy)
        if len(outliers):
            return 0

    # MAD passes, check CV
    std_dev = np.std(elem_copy)
    mean = np.mean(elem_copy)

    if mean < 0.0001 and not get_score:
        # CV cannot be calculate, but MAD passed, keep new gap
        return 2

    cv = std_dev / mean
    if get_score:
        # early return for scores only
        return abs(outliers), abs(cv)

    if cv < 0.1:
        return 2
    elif new_element is not None:
        # check if gap is a multiple of mean
        ratio = new_element/mean
        scale = round(ratio)
        # Only allow 1-2 lines to be interpolated
        if 2 <= scale <= 4 and abs(scale - ratio) < 0.2:
            return 1
    return 0


def param_check_imbalance(elements, median, mad, dist_med, std_dev, mean, index=-1):
    """
    A non-calculated version of check_imbalance
    since mad, distance to median, standard deviation, and mean doesn't need to be recalculated
    :param elements: A 1D array of numerical value, could also be listed 1 value
    :param median: Median of the original element list
    :param mad: Median Absolute Deviation value of a given list of elements
    :param dist_med: Distance to median throughout the entire list
    :param std_dev: standard deviation of the original list
    :param mean: mean of the original list
    :param index: Index of element to check if its an outlier (add to end if elements if want to check new element)
    :return: False means it not an outlier, True means it is. None means the element index couldnt be accessed
    """
    if elements is None or len(elements) <= 0 or len(elements) <= index:
        return None

    if mad == 0:
        mad = 0.01

    mod_z_score = 0.6745 * (elements[index] - median) / mad
    # Apparantly 2.5-3.5 std dev is a good indication of outlier
    if mod_z_score > 3.0:
        return True

    if abs(mean) < 0.0001:
        # CV cannot be calculate, but MAD passed, keep new gap
        mean = 0.1

    ratio = elements[index] / mean
    scale = round(ratio)
    # Only allow 1 lines to be interpolated
    if scale == 2 and abs(scale - ratio) < 0.1:
        return False

    # Z score based std_dev formula
    z = (elements[index] - mean) / std_dev
    if z <= 2.5:
        return False

    return True


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
