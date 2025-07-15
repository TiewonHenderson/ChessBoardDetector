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
    points_2d = []
    # Mask to allow indexing the points back to which line they intersected
    mask = []
    for group in points:
        for i, point in enumerate(group):
            if point is not None and len(point) == 2:
                mask.append(i)
                points_2d.append(point)
    points_np = np.array(points_2d)  # Shape (N, 2)

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
            indexes.append(mask[i])
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


def get_starting_lines(lines, image_shape):
    """
    This function runs through the lines inputed (assuming its a theta cluster) to find
    consistent gaps with brute force.
    :param lines: Expects one cluster of lines (2D array)
    :param image_shape:
    :return:
    """
    # Starting lines must be output a len > 8, and < 10.
    if len(lines) < 8:
        return []
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
            gap = ht.orthogonal_gap(line0, line1)

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

    # for x in all_gaps:
    #     print(x)

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
