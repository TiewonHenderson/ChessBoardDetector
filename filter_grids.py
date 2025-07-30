import numpy as np
import cv2
from math import sin,cos
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.spatial import KDTree


def snap_to_cardinal_diagonal(angle, cardinal_threshold=15):
    """
    Snap an angle to the nearest cardinal or diagonal direction.
    Claude generated
    """
    # Target angles: cardinal and diagonal directions
    targets = [0, 45, 90, 135, 180, 225, 270, 315]
    orientation_dict = {}

    # Find the target with minimum distance
    min_distance = float('inf')
    closest_target = 0

    for i, target in enumerate(targets):
        # Calculate distance considering circular nature of angles
        distance = min(abs(angle - target), 360 - abs(angle - target))

        if distance < min_distance:
            # cardinal degrees are snapped on when they're really CLOSE
            if i % 2 == 0 and distance > cardinal_threshold:
                continue
            min_distance = distance
            closest_target = target

    return closest_target


def vp_direction(angle, cardinal_threshold=15):
    """
    Improved version of snap_to_cardinal_diagonal
    Snap an angle to directions that matter chessboard orientation wise
    """
    # Target angles: cardinal and diagonal directions
    targets = [0, 45, 90, 270, 315]

    # Find the target with minimum distance
    min_distance = float('inf')
    closest_target = 0
    got_angle = (angle + 180) % 360 if angle > 90 else angle

    for i, target in enumerate(targets):
        # Calculate distance considering circular nature of angles
        distance = min(abs(got_angle - target), 360 - abs(got_angle - target))

        if distance < min_distance:
            # cardinal degrees are snapped on when they're really CLOSE
            if i % 2 == 0 and distance > cardinal_threshold:
                continue
            min_distance = distance
            closest_target = target

    return closest_target


def intersection_polar_lines(line1, line2, need_dir=False, eps=1e-6):
    """
    Packed line = [[rho, theta]]
    :param line1: A packed line from hough line transformation
    :param line2: Different packed line from hough line transformation
    :param need_dir: In case of a parallel line, this function can give the direction of parallel lines
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

    # Claude generate in order to get direction of infinity as intersection of the two lines

    # Assume Lines are parallel - calculate direction of infinity
    # Direction vector of the line is perpendicular to normal vector [a, b]
    # So direction is [b, a] (rotated 90 degrees)
    direction = np.array([a1, b1])  # Using first line's coefficients

    # Convert direction to angle in image coordinates (origin at top-left)
    # In image coordinates, positive y goes down, so we need to flip y
    angle_rad = np.arctan2(direction[1], direction[0])

    # Convert to degrees for easier interpretation
    angle_deg = np.degrees(angle_rad)

    # Normalize angle to [0, 360) range
    if angle_deg < 0:
        angle_deg += 360

    if abs(det) < eps:
        if need_dir:
            return None, snap_to_cardinal_diagonal(angle_deg)
        else:
            return None

    # solves system of equations for both lines' ax+by=ρ equation
    x, y = np.linalg.solve(A, B)
    if need_dir:
        return [round(x), round(y)], snap_to_cardinal_diagonal(angle_deg)
    return [round(x), round(y)]


def normalize_theta(theta):
    """
    This is for clustering lines by orientation
    This should not replace the theta value from houghline outputs!

    :param theta: Theta corresponding to a detected hough line
    :return: normalized theta would represent the orientation better
    """
    return theta % np.pi


def normalize_line(line):
    """
    One of the issues of polar coordinates is it revolves around the origin
    This creates the line as a positive rho,
    but offsets theta a lot due to the nature of polar coordinates
    :param line: A [rho, theta] value from hough line transform
    :return:
    """
    rho, theta = line

    # If rho is negative, the line itself is on the opposite side of the origin
    # we would need to rotate the line 180 degree which changes theta
    if rho < 0:
        rho = -rho
        theta = (theta + np.pi) % (np.pi * 2)

    return [rho, theta]


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


def label_to_cluster(lines, labels, indices):
    """

    :param lines: unpacked lines from houghline transformation
    :param labels: Labels from a clustering algorithm to assign each line
    :param indices: Return indices of lines that belongs to cluster
    :return: A dict representing key = average theta in cluster, value = line clusters (list of [rho, theta])
    """
    clusters = []
    # sums is used to assign average theta as keys to each cluster
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
        if indices:
            clusters[label].append(x)
        else:
            clusters[label].append(lines[x])

    cluster_dict = {}
    clusters_index = 0
    for amt, sum in sums:
        average = sum / amt
        # Should not consider less then 2 lines for a 8x8 grid
        if len(clusters[clusters_index]) < 2:
            continue
        for key in cluster_dict:
            if abs(average - key) < 0.02:
                cluster_dict[key].extend(clusters[clusters_index])
        else:
            cluster_dict[round(average, 5)] = clusters[clusters_index]
        clusters_index += 1

    return cluster_dict


def dbscan_cluster_lines(lines, indices=False, eps=0.02):
    """
        Clusters using DBSCAN, then labels each cluster by it's average theta

        Remember! the detection mode matters with distance,
        If detection mode is strict, distance should be a lot more

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
        :param indices: Return indices of lines that belongs to cluster
        :param eps: The radian epsilon to cluster lines within
    """
    if len(lines) == 0:
        return {}
    line_array = np.array([
        [normalize_theta(theta)]
        for _, theta in lines  # Use lines directly
    ])
    """
    DBSCAN expects a 2d np array representing points
    """
    clustering = DBSCAN(eps=eps, min_samples=2, algorithm='ball_tree').fit(line_array)
    labels = clustering.labels_

    return label_to_cluster(lines, labels, indices)


def kmeans_cluster_lines(lines):
    """
    Uses 1D KMeans to cluster theta (normalized)
    :param lines: unpacked 2D array of rho,theta values
    :return:
    """
    line_array = lines.copy()
    line_array = np.array([
        [theta % np.pi]
        for _, theta in line_array
    ])
    labels = KMeans(n_clusters=2, n_init='auto').fit_predict(line_array)
    return label_to_cluster(lines, labels)


def filter_similar_lines(lines, image_shape):
    """
    Filters similar lines by checking if rho and theta differences are below thresholds
    Compares orientation

    rho_eps = 0.01 * H
    Which is the minimum gap expected

    theta_eps = 0.03 radian = ~2 degree

    :param lines: unpacked 2D array of rho,theta values
    :param image_shape: [height, width] of the image
    """
    if len(lines) == 0:
        return lines

    filtered = []
    # Aspect ratio is 1:1
    h, w = image_shape
    rho_eps = 0.01 * h
    theta_eps = 0.03

    for rho, theta in lines:
        temp_rho = abs(rho)
        too_close = False

        for filtered_rho, filtered_theta in filtered:
            temp_f_rho = abs(filtered_rho)
            # Check if both rho and theta values are close
            rho_close = abs(temp_rho - temp_f_rho) < rho_eps
            theta_close = abs(theta - filtered_theta) < theta_eps

            if rho_close and theta_close:
                too_close = True
                break

        if not too_close:
            filtered.append((rho, theta))

    return [[rho, theta] for rho, theta in filtered]


def check_grid_like(group1, group2, image_shape=None, corners=[]):
    """
    Weight cases:
    Evenly spaced intersections = 30/100
    High ratio of intersections being corners = 30/100
    Evenly spaced rho difference = 20/100
    Amount of lines = 20/100


    :param group1: Cluster of lines represented as an unpacked 2D list
    :param group2: Cluster of lines represented as an unpacked 2D list
    :return: A score on how grid like the clu
    """
    if len(group1) == 0 or len(group2) == 0:
        return
    group1.sort(key=lambda x: x[0])
    group2.sort(key=lambda x: x[0])

    c_tree = None
    h = w = None
    if len(corners) > 0 and image_shape is not None:
        c_tree = KDTree(corners)
        h, w, _ = image_shape

    # The more closer the amount of lines is to 2 * ([8,10]), the better
    total_lines = len(group1) + len(group2)
    if total_lines <= 4:
        return 0, []
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
    for i, lines1 in enumerate(group1):
        row = []
        prev_intersect_point = None
        for j, lines2 in enumerate(group2):
            intersect_point = intersection_polar_lines(lines1, lines2)
            if intersect_point is None:
                # None is appended to keep grid align
                row.append(None)
                continue
            if c_tree is not None and image_shape is not None:
                eps = h / 100
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
    for i, line1 in enumerate(group1[1:], start=1):
        dist = abs(group1[i - 1][0] - line1[0])
        rho_dist_list1.append(dist)
    for i, line2 in enumerate(group2[1:], start=1):
        dist = abs(group2[i - 1][0] - line2[0])
        rho_dist_list2.append(dist)

    # Final confidencce score to classify grid by lines
    score = 0
    max_cv = 0.20
    intersect_points = 30
    line_points = 20

    """
    Variance scoring
    CV              Interpretation
    [0, 0.05]       Perfect CV
    [0.05, 0.1]     Great CV
    [0.1, 0.2]      Acceptable CV
    [0.2, inf)      No points
    """
    # Score for: Evenly spaced intersections = 30/100
    intersect_mean = 1
    if all_intersects_mean[1] > 0 and all_intersects_mean[0] > 0:
        intersect_mean = all_intersects_mean[0] / all_intersects_mean[1]
    intersect_cv = np.std(np.array(all_intersects)) / intersect_mean
    score += scaled_scoring(intersect_cv, max_cv, intersect_points, 2)

    # High ratio of intersections (with high intersections) being corners = 30/100
    corner_intersect_ratio = 1
    if total_intersect > 0:
        corner_intersect_ratio = abs(1 - (corner_intersect/total_intersect))
    score += scaled_scoring(corner_intersect_ratio, 1, intersect_points, 1)

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

    """
    Relative Difference of the gap means should be small (idea from GPT)
    The gaps should be consistent throughout
    If gap_mean relative difference is too much, no points can be given from
    Evenly spaced intersections, Evenly spaced rho difference
    """
    # if abs(c1_mean - c2_mean) / max(c1_mean, c2_mean) < 0.35:
    #     score += scaled_scoring(aver_cluster_cv, max_cv, 20, 2)
    # else:
    #     score = 0

    # Score for: Amount of lines = 20/100
    # amount of lines [8,27] gets half by default, additional through parabolic scaling
    if total_lines > 7 and total_lines < 28:
        norm_amt = (total_lines - 18) / (18 - 8)
        score += (line_points/2) * max(0, 1 - norm_amt ** 2) + (line_points/2)


    # print("Grid scores")
    # print("Intersection variance: ", intersect_cv)
    # print("Line cluster 1 variance: ", c1_cv)
    # print("Line cluster 2 variance: ", c2_cv)
    # print("Amount of intersections: ", total_intersect)
    # print("Amount of good intersection: ", corner_intersect)
    # print("Amount of lines: ", total_lines)
    print("Final score: ", score, "/ 100")
    return score, intersection_list


def sim_inserted_lines():
    """
    Given a sufficient score
    :return:
    """