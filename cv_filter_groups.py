import sys
import cv2
import numpy as np
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import HoughTransform as ht
from ChessBoardDetector import Chessboard_detection as cd
from math import sin,cos, log10
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


def bucket_sort(elements):
    """
    Bucket sort to get max quantity of item
    :param elements: A 1d list of elements
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
        if abs(closest_key - x) < 0.04:
            bucket[closest_key] += 1
        else:
            bucket[round(float(x), 5)] = 1
    return bucket


def get_outlier_lines(lines, sect_list, mask, gap_list, min_gap):
    """
    gap and lines structure:
    lines = [0 , 1 , 2 , 3 , 4..]
    gaps = [   0,  1,  2,  3..]

    to find gap -> lines
    if gap at i and i + 1 is inconsistent:
        remove line at i + 1
    if only i gap is inconsistent:
        Should only occur on the
        - border lines
            remove border line
        - Overlapping lines
            handled properly in vp.enough_similar_theta
            Choose line with evenly spaced theta.
    :param lines: Original list of lines represented as [rho, theta]
    :param sect_list: The list of intersection formed from a perpendicular line and lines
                      Assumed sorted and not aligned with lines, which is the role of mask
    :param mask: Masks the indices from sect_list to lines, in effect, masking gap_list to lines as well
    :param gap_list: The gaps between the sorted points in sect_list
    :param min_gap: The minimum distance acceptable as a grid square between two lines
    """

    """
    Due to gap_list needing to line up with the mask variable
    lines that are soon to be not included will form None gap values.
    The gaps of the other lines will represent the gaps skipping the bad line

    lines = [0 , 1 , 2 , 3 , 4..]
    gaps = [   0,  1,  2,  3..]
    """
    # MAD and CV to find outliers
    mid = np.median(gap_list)
    mad, dist_med = check_MAD(gap_list, get_mad=True)
    std_dev = np.std(np.array(gap_list))
    mean = np.mean(np.array(gap_list))

    bad_lines = []
    def update(index, bad_lines=bad_lines, gap_list=gap_list):
        """repeated operation to update bad_lines"""
        if index > 0:
            if gap_list[index - 1] is not None:
                gap_list[index - 1] += gap_list[index]
        gap_list[index] = None
        bad_lines.append(index)

    i = 0
    intervals = [[]]
    while i < len(gap_list):
        gap = gap_list[i]
        """
        Overlapping laps causes mingap, those check don't need to evaluate intervals of gaps
        """
        if gap is not None:
            if gap < min_gap:
                # Checks the relative growth/shrunk of theta to see if the line fits
                l1_var_test = []
                l2_var_test = []
                # Cannot check surrounding lines on border, use consecutive lines as reference
                if i < len(gap_list) - 1:
                    l1_var_test = [lines[mask[j]][1] for j in range(len(lines)) if j != i + 1]
                    l2_var_test = [lines[mask[j]][1] for j in range(len(lines)) if j != i]
                elif i == len(gap_list) - 1:
                    # reverse range from len(lines) - 3 to 0
                    l1_var_test = [lines[mask[j]][1] for j in range(len(lines) - 3, -1, -1)].append(lines[mask[i]])
                    l2_var_test = [lines[mask[j]][1] for j in range(len(lines) - 3, -1, -1)].append(lines[mask[i - 1]])
                l1_mad, l1_cv = check_imbalance(l1_var_test, index=-1, get_score=True)
                l2_mad, l2_cv = check_imbalance(l2_var_test, index=-1, get_score=True)
                # CV matter more in this context
                if l1_cv < l2_cv:
                    update(i + 1)
                elif l1_cv > l2_cv:
                    update(i)
                else:
                    if l1_mad < l2_mad:
                        update(i + 1)
                    else:
                        update(i)
            elif param_check_imbalance(gap_list, mid, mad, dist_med, std_dev, mean, i):
                # Border line cases, they can only form 1 bad group
                # remember, we're return index of LINES, not gaps
                other_index = None
                if i == 0:
                    other_index = [i + 1]
                elif i == len(gap_list) - 1:
                    other_index = [i - 1]
                else:
                    # All other cases, check both sides
                    other_index = [i - 1, i + 1]
                for index in other_index:
                    # If gap is None, it was likely removed, this neighboring gap is also likely bad
                    if gap_list[index] is None:
                        update(i)
                    # 0-1 gap is bad, but so is 1-2, so line 1 is bad
                    # n-1 - n gap is bad, but so is n-2 - n-1, so line n-1 is bad
                    z_score = abs((gap_list[index] - mean) / std_dev)
                    if gap_list[index] < min_gap or z_score > 2:
                        if gap_scale > 2.2:
                            intervals.append([])
                        update(index)
                # 0-1 gap is bad, but 1-2 isnt, so line 0 is bad
                # n-1 - n gap is bad, but n-2 - n-1 isnt, so line n is bad
                update(i)
            # Expand on the previous group
            else:
                intervals[-1].append(i)
        i += 1

    return bad_lines


def cv_clean_lines(lines, direction, image_shape, image=None):
    """
    Minimum gap expectation:
    Chessboard >= 1/4 * (image area)
    Horizontal tilt >= 25 degree (0 degree == same angle as chessboard, cannot see grid)

    min(W, H) / 2 / 8 * sin(25°)
    final formula for minimum gap expectation = 0.0264 * min(W, H)

    Uses a perpendicular line in respect to given direction in order to get
    better consistent gap calculation

    :param lines: unpacked 2D array of rho,theta values
    :param direction: The direction the vp is in respect to the middle of the image
    :param image_shape: [height, width] of the image
    :return: A 2D list that are grouped based off gap between them,
             with a 1D array representing gap average of each group of lines

             !can return None, None if group of lines were determined as bad!
    """
    # Not enough lines to be considered a grid
    if len(lines) < 4:
        return None, None
    h, w = image_shape
    min_gap = 0.0264 * min(h, w)
    # Gets the middle of the lines group and perpendicular theta
    cx = w / 2
    cy = h / 2
    perd_theta = (np.deg2rad(direction) + np.pi / 2) % np.pi
    # Formula given by GPT to get perpendicular line at center of image
    perd_line = (cx * np.cos(perd_theta) + cy * np.sin(perd_theta), perd_theta)

    # Gets intersections of each line to the perd_line, i to keep lines index intact
    sect_list = [(i, fg.intersection_polar_lines(perd_line, l)) for i, l in enumerate(lines)]
    # Sorts by x then y
    sect_list = sorted(sect_list, key=lambda p: (p[1][0], p[1][1]))

    # Extract indices and points separately
    mask = [item[0] for item in sect_list]
    sect_list = [item[1] for item in sect_list]

    gap_list = []
    for i in range(len(sect_list) - 1):
        p1 = np.array(sect_list[i])
        p2 = np.array(sect_list[i + 1])
        gap_list.append(round(float(np.linalg.norm(p1 - p2)), 4))

    print(gap_list)
    cd.find_exact_line(image, lines, 0, green=False)

    mid = np.median(gap_list)
    if mid < 0.1:
        # If majority of the gap is < 0.1, this groups is definitely bad
        return None, None

    # gets priority theta by bucket sort
    buckets = bucket_sort([l[1] for l in lines])
    most_theta = max(buckets, key=lambda k: buckets[k])
    if buckets[most_theta] < 2:
        # No point in priority theta with all different theta
        most_theta = None

    # Removes the labelled bad lines
    bad_lines = get_outlier_lines(lines, sect_list, mask, gap_list, min_gap)

    ret_lines = []
    new_gaps = []
    i = 0
    while i < len(lines):
        if i in bad_lines:
            i += 1
            continue
        ret_lines.append(lines[mask[i]])
        i += 1
    return ret_lines, [gap for gap in gap_list if gap is not None]


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
            return []
    outliers = []
    for i, x in enumerate(elements):
        mod_z_score = 0.6745 * (x - median) / mad
        # Return all z_score instead
        if get_score:
            outliers.append(mod_z_score)
            continue
        # Apparantly 3.5 std dev is a good indication of outlier
        if mod_z_score > std_dev:
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

    if mean < 0.0001:
        # CV cannot be calculate, but MAD passed, keep new gap
        return 2

    cv = std_dev / mean
    if get_score:
        # early return for scores only
        return outliers, cv

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
    :param elements: A 1D array of numerical value
    :param median: Median of the original element list
    :param mad: Median Absolute Deviation value of a given list of elements
    :param dist_med: Distance to median throughout the entire list
    :param std_dev: standard deviation of the original list
    :param mean: mean of the original list
    :param index: Index of element to check if its an outlier (add to end if elements if want to check new element)
    :return: False means it not an outlier, True means it is. None means the element index couldnt be accessed
    """
    if elements is None or len(elements) == 0 or len(elements) <= index:
        return None

    if mad == 0:
        mad = np.median(np.unique(dist_med))
        if mad < 0.0001:
            # All values are the same to cause 0, not an outlier
            return False

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
