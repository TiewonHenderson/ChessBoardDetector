import numpy as np
from sklearn.cluster import DBSCAN


def intersection_polar_lines(line1, line2, eps=1e-6):
    """
    Packed line = [[rho, theta]]
    :param line1: A packed line from hough line transformation
    :param line2: Different packed line from hough line transformation
    :param eps: epsilon threhold to determine if there is a valid intersection
    :return:
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
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


def cluster_lines(lines):
    """
        Clusters using DBSCAN, then labels each cluster by it's average theta

        To do:
        Finds similar spacing lines by sorting, then linearly find equal distances

        Remember! the detection mode matters with distance,
        If detection mode is strict, distance is a lot more
    """
    line_array = lines.copy()
    # unpack from nested array
    if line_array is None:
        line_array = []
    else:
        line_array = [l[0] for l in line_array]

    line_array = np.array([
        [normalize_theta(theta)]
        for rho, theta in line_array
    ])

    """
    radians       degree
    0.6       =   34.3775
    0.5       =   28.6479
    0.4       =   22.9183
    0.3       =   17.1887
    """
    eps = 0.4
    clustering = DBSCAN(eps=eps, min_samples=2).fit(line_array)  # tune eps
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
        sums[label][1] += lines[x][0][1]
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


def filter_similar_lines(lines, threshold=10):
    """
    Filters similar lines by similar between rhos and theta is below a threshold
    """
    filtered = []
    filter_lines = []
    for line in lines:
        rho, theta = line[0]
        too_close = False
        for r, t in filtered:
            # checks if both rho and theta values are close, indicating extremely similar lines
            if abs(rho - r) < threshold and abs(theta - t) < np.deg2rad(5):
                too_close = True
                break
        if not too_close:
            filtered.append((rho, theta))
            filter_lines.append([[rho, theta]])

    return filter_lines


def check_grid_like(cluster1, cluster2):
    """
    Weight cases:
    Evenly spaced intersections = 40/100
    Evenly spaced rho difference = 20/100
    Amount of intersections = 15/100
    Amount of lines = 15/100
    Intersections are close relative to image resolution = 10/100
    ^ depends on detection_mode

    :param cluster1: Cluster of lines represented as an unpacked 2D list
    :param cluster2: Cluster of lines represented as an unpacked 2D list
    :return: [0,1] interval score representing confidence
    """
    cluster1.sort(key=lambda x: x[0][0])
    cluster2.sort(key=lambda x: x[0][0])

    # The more closer the amount of lines is to 2 * ([8,10]), the better
    total_lines = len(cluster1) + len(cluster2)
    total_intersection = 0
    intersection_list = []
    # Used to calculate variance between intersections
    intersect_up_dist_list = []
    intersect_left_dist_list = []
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
            total_intersection += 1
            row.append(intersect_point)
            # Find distance between each line intersection
            if i > 0 and len(intersection_list[i - 1]) > j:
                above_point = intersection_list[i - 1][j]
                if above_point is not None:
                    dist_diff_up = np.linalg.norm(
                        np.array(above_point) - np.array(intersect_point)
                    )
                    intersect_up_dist_list.append(dist_diff_up)
            if prev_intersect_point is not None:
                dist_diff_left = np.linalg.norm(
                    np.array(prev_intersect_point) - np.array(intersect_point)
                )
                intersect_left_dist_list.append(dist_diff_left)
            prev_intersect_point = intersect_point
        intersection_list.append(row)

    # Section to find variance between line rhos
    for i, line1 in enumerate(cluster1[1:], start=1):
        dist = abs(cluster1[i - 1][0][0] - line1[0][0])
        rho_dist_list1.append(dist)
    for i, line2 in enumerate(cluster2[1:], start=1):
        dist = abs(cluster2[i - 1][0][0] - line2[0][0])
        rho_dist_list2.append(dist)

    up_intersect_variance = np.var(np.array(intersect_up_dist_list))
    left_intersect_variance = np.var(np.array(intersect_left_dist_list))
    intersect_variance = np.var(np.array([up_intersect_variance, left_intersect_variance]))
    cluster1_variance = np.var(np.array(rho_dist_list1))
    cluster2_variance = np.var(np.array(rho_dist_list2))
    print("Grid scores")
    print("Intersection variance: ", intersect_variance)
    print("Line cluster 1 variance: ", cluster1_variance)
    print("Line cluster 2 variance: ", cluster2_variance)
    print("Amount of intersections: ", total_intersection)
    print("Amount of lines: ", total_lines)
    # print("Average distance of each intersection: ", total_dist/count)

