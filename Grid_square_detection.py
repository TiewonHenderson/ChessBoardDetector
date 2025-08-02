import numpy as np
import cv2
from math import sin,cos
from scipy.spatial import KDTree


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


def relative_grid_check(group1, group2, image_shape=None, corners=[]):
    """
    Weight cases:
    High ratio of intersections being corners = 50/100
    Evenly spaced rho difference = 30/100
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
        corner_intersect_ratio = abs(1 - (corner_intersect / total_intersect))
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
        score += (line_points / 2) * max(0, 1 - norm_amt ** 2) + (line_points / 2)

    # print("Grid scores")
    # print("Intersection variance: ", intersect_cv)
    # print("Line cluster 1 variance: ", c1_cv)
    # print("Line cluster 2 variance: ", c2_cv)
    # print("Amount of intersections: ", total_intersect)
    # print("Amount of good intersection: ", corner_intersect)
    # print("Amount of lines: ", total_lines)
    print("Final score: ", score, "/ 100")
    return score, intersection_list




