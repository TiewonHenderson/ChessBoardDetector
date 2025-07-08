import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from skimage.color import rgb2lab
from scipy import ndimage
from scipy.spatial import KDTree


def intersection_polar_lines(line1, line2, eps=1e-6):
    """
    Packed line = [[rho, theta]]
    :param line1: A packed line from hough line transformation
    :param line2: Different packed line from hough line transformation
    :param eps: epsilon threhold to determine if there is a valid intersection
    :return:
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    # linear form of polar coordinate lines: ax+by=ρ
    # where a, b = cos(theta), sin(theta) respectively
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)

    # Coefficient matrix
    A = np.array([[a1, b1],
                  [a2, b2]])
    B = np.array([rho1, rho2])

    # Check if determinant is close to zero → lines are parallel
    det = np.linalg.det(A)
    if abs(det) < eps:
        return None  # No intersection (parallel lines)

    # solves system of equations for both lines' ax+by=ρ equation
    x, y = np.linalg.solve(A, B)
    return (x, y)


def normalize_theta(theta):
    """
    This is for clustering lines by orientation
    :param theta: Theta corresponding to a detected hough line
    :return: normalized theta would represent the orientation better
    """
    return theta - np.pi if theta > np.pi/2 else theta


def scaled_scoring(cv, max_cv, max_points, power=1):
    """
    :param cv: The found coefficient of variation
    :param max_cv: The threshold value of CV to give 0 points
    :param max_points: The most amount of points the scoring can get
    :param power: The exponential power to scale the score given (higher power == only closer cv to 0 gets maj score)
    :return:
    """
    if cv >= max_cv:
        return 0
    score = max_points * (1 - (cv / max_cv) ** power)
    return round(score, 5)


def cluster_lines(lines, eps=0.2):
    """
        Clusters using DBSCAN, then labels each cluster by it's average theta

        Remember! the detection mode matters with distance,
        If detection mode is strict, distance should be a lot more

        DBSCAN is not good with a lot of points, if using small threshold settings,
        Make sure to use small epsilon value

            radians       degree
            0.5       =   28.6479
            0.4       =   22.9183
            0.3       =   17.1887
            0.2       =   11.4592
        :param eps: The radian epsilon to cluster lines within
    """
    line_array = lines.copy()
    line_array = np.array([
        [normalize_theta(theta)]
        for rho, theta in line_array
    ])

    """
    DBSCAN expects a 2d np array representing points
    """
    clustering = DBSCAN(eps=eps, min_samples=2, algorithm='ball_tree').fit(line_array)  # tune eps
    labels = clustering.labels_

    clusters = []
    sums = []
    # lines is 3d of 2d representing [[rho,theta]]
    for x in range(len(lines)):
        label = labels[x]
        # Outliers wont be considered
        if label == -1:
            continue
        # Add more empty groups
        while len(clusters) <= label:
            clusters.append([])
            sums.append([0,0])

        sums[label][0] += 1
        sums[label][1] += lines[x][1]
        clusters[label].append(lines[x])

    cluster_dict = {}
    clusters_index = 0
    for amt, sum in sums:
        average = sum / amt
        # Should not consider less then 4 lines for a 8x8 grid
        if len(clusters[clusters_index]) < 4:
            continue
        if average in cluster_dict:
            cluster_dict[average].extend(clusters[clusters_index])
        else:
            cluster_dict[average] = clusters[clusters_index]
        clusters_index += 1

    return cluster_dict


def filter_similar_lines(lines, eps=11):
    """
    Filters similar lines by similar between rhos and theta is below a threshold
    A general filtering function, not to be confused with
    """
    filtered = []
    filter_lines = []
    for rho, theta in lines:
        too_close = False
        for r, t in filtered:
            # checks if both rho and theta values are close, indicating extremely similar lines
            if abs(rho - r) < eps and abs(theta - t) < np.deg2rad(int(eps/4)):
                too_close = True
                break
        if not too_close:
            filtered.append((rho, theta))
            filter_lines.append([rho, theta])

    return filter_lines


def filter_similar_rho(clusters, eps):
    """
    This function also expects some filter process to happen beforehand
    Otherwise meaning two lines consecutively could cause major offsetting
    :param clusters: A 3D list of lines representing clusters of [rho, theta] lines
    :param eps: Threhold to consider two rhos to be similar enough
    :return: A filtered 3D list of clustered lines
    """
    line_clusters = clusters.copy()
    for lines in line_clusters:
        i = 0
        while i < len(lines) - 1:
            r1, t1= lines[i]
            r2, t2 = lines[i + 1]
            if abs(r2 - r1) < eps:
                # replace the two lines as the new mean line
                lines[i] = [(r1 + r2) / 2, (t1 + t2) / 2]
                lines.pop(i + 1)
            else:
                i += 1
    return line_clusters


def check_grid_like(cluster1, cluster2, image_shape=None, corners=[]):
    """
    Weight cases:
    Evenly spaced intersections = 40/100
    High ratio of intersections being corners = 30/100
    Evenly spaced rho difference = 20/100
    Amount of lines = 10/100

    :param cluster1: Cluster of lines represented as an unpacked 2D list
    :param cluster2: Cluster of lines represented as an unpacked 2D list
    :return: [0,1] interval score representing confidence
    """
    if len(cluster1) == 0 or len(cluster1) == 0:
        return
    cluster1.sort(key=lambda x: x[0])
    cluster2.sort(key=lambda x: x[0])

    c_tree = None
    h = w = None
    if len(corners) > 0 and image_shape is not None:
        c_tree = KDTree(corners)
        h, w, _ = image_shape

    # The more closer the amount of lines is to 2 * ([8,10]), the better
    total_lines = len(cluster1) + len(cluster2)
    total_intersect = 0
    corner_intersect = 0
    intersection_list = []
    # Used to calculate variance between intersections
    all_intersects = []
    # Used to calculate mean by: [sum, amount], used to calculate spacing based variance
    all_intersects_mean = [0, 0]
    # Used to calculte variance between lines
    rho_dist_list1 = []
    rho_dist_list2 = []

    # Section to find variance intersection between lines
    for i, lines1 in enumerate(cluster1):
        row = []
        prev_intersect_point = None
        for j, lines2 in enumerate(cluster2):
            intersect_point = intersection_polar_lines(lines1, lines2)
            if intersect_point is None:
                # None is appended to keep grid align
                row.append(None)
                continue
            if c_tree is not None and image_shape is not None:
                eps = max(h, w) / 100
                """
                KD-tree allows searching for same/similar points (intersections) without
                brute forcing all points
                Time complexity: 
                N intersections, M corners
                O(M log M + N log M)
                
                indices correspond to the original corners list that considered intersection
                """
                indices = c_tree.query_ball_point(intersect_point, eps)
                if len(indices) > 0:
                    corner_intersect += 1
            total_intersect += 1
            row.append(intersect_point)
            # Find distance between each line intersection
            if i > 0 and len(intersection_list[i - 1]) > j:
                above_point = intersection_list[i - 1][j]
                if above_point is not None:
                    dist_diff_up = np.linalg.norm(
                        np.array(above_point) - np.array(intersect_point)
                    )
                    all_intersects.append(dist_diff_up)
                    all_intersects_mean[0] += dist_diff_up
                    all_intersects_mean[1] += 1
            if prev_intersect_point is not None:
                dist_diff_left = np.linalg.norm(
                    np.array(prev_intersect_point) - np.array(intersect_point)
                )
                all_intersects.append(dist_diff_left)
                all_intersects_mean[0] += dist_diff_left
                all_intersects_mean[1] += 1
            prev_intersect_point = intersect_point
        intersection_list.append(row)

    # Section to find variance between line rhos
    for i, line1 in enumerate(cluster1[1:], start=1):
        dist = abs(cluster1[i - 1][0] - line1[0])
        rho_dist_list1.append(dist)
    for i, line2 in enumerate(cluster2[1:], start=1):
        dist = abs(cluster2[i - 1][0] - line2[0])
        rho_dist_list2.append(dist)

    # Final confidencce score to classify grid by lines
    score = 0
    max_cv = 0.20
    """
    Variance scoring
    
    CV              Interpretation
    [0, 0.05]       Perfect CV
    [0.05, 0.1]     Great CV
    [0.1, 0.2]      Acceptable CV
    [0.2, inf)      No points
    """
    # Score for: Evenly spaced intersections = 40/100
    intersect_mean = 1
    if all_intersects_mean[1] > 0 and all_intersects_mean[0] > 0:
        intersect_mean = all_intersects_mean[0] / all_intersects_mean[1]
    intersect_cv = np.std(np.array(all_intersects)) / intersect_mean
    score += scaled_scoring(intersect_cv, max_cv, 40, 2)

    # High ratio of intersections (with high intersections) being corners = 30/100
    corner_intersect_ratio = 1
    if total_intersect > 0:
        corner_intersect_ratio = abs(1 - (corner_intersect/total_intersect))
    score += scaled_scoring(corner_intersect_ratio, 1, 25, 1)

    # Score for: Evenly spaced rho difference = 20/100
    c1_mean, c2_mean = sum(rho_dist_list1), sum(rho_dist_list2)
    if c1_mean != 0 and len(rho_dist_list1) != 0:
        c1_mean /= len(rho_dist_list1)
    else:
        c1_mean = 1

    if c2_mean != 0 and len(rho_dist_list2) != 0:
        c2_mean /= len(rho_dist_list2)
    else:
        c2_mean = 1

    # cluster coefficient of variation
    c1_cv = np.std(np.array(rho_dist_list1)) / c1_mean
    c2_cv = np.std(np.array(rho_dist_list2)) / c2_mean
    aver_cluster_cv = (c1_cv + c2_cv) / 2

    score += scaled_scoring(aver_cluster_cv, max_cv, 20, 2)

    # Score for: Amount of lines = 10/100
    # Preferred amount of lines [8,27], done through parabolic scaling
    if total_lines > 7 and total_lines < 28:
        norm_amt = (total_lines - 18) / (18 - 8)
        score += 10 * max(0, 1 - norm_amt ** 2)

    # print("Grid scores")
    # print("Intersection variance: ", intersect_cv)
    # print("Line cluster 1 variance: ", c1_cv)
    # print("Line cluster 2 variance: ", c2_cv)
    # print("Amount of intersections: ", total_intersect)
    # print("Amount of good intersection: ", corner_intersect)
    # print("Amount of lines: ", total_lines)
    # print("Final score: ", score, "/ 100")
    return score

