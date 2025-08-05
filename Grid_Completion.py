import sys
import numpy as np
import cv2
from math import sin,cos
from sklearn.linear_model import RANSACRegressor
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import Chessboard_detection as cd
from ChessBoardDetector import HoughTransform as ht
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import cv_filter_groups as cvfg


def closest_corner(corners, line):
    """
    Formula to see how close a corner is in respect to the line
    x * np.cos(theta) + y * np.sin(theta) - rho ~ 0 (closer to 0, closer to line)
    due to
    rho = x * np.cos(theta) + y * np.sin(theta)

    Revised old code by AI
    :param corners: A 2D list of points representing cartesian points
    :param line: A single [rho, theta] line
    :return:
    """
    if len(corners) < 1 or line is None:
        return None

    rho, theta = line
    get_dist = lambda x,y: abs(x * cos(theta) + y * sin(theta) - rho)

    corner_distances = []
    for corner in corners:
        x, y = corner
        dist = get_dist(x, y)
        corner_distances.append([dist, corner])

    # Sort by distance and take top 8 corners
    corner_distances.sort(key=lambda x: x[0])
    top_closest = corner_distances[:8]
    return top_closest


def intersect_verification(sect_list, corner, threshold=7):
    """
    Verifies intersection if they're close enough to a corner point
    :param sect_list:
    :param corner:
    :param threshold: 7 pixel distance is the default offset of points to be considered the same
    :return:
    """
    verified = set()
    verified_indices = set()
    max_dist = threshold ** 2
    for row in sect_list:
        for element in row:
            if element is None:
                continue
            indices, pt1 = element
            x_1, y_1 = pt1
            # Use of dist^2 = (x_2 - x_1)^2 + (y_2 - y_1)^2 to avoid sqrt operations
            min_dist_pt = (sys.maxsize, None)
            for pt2 in corner:
                if pt2 is None:
                    continue
                x_2, y_2 = pt2
                dist_sqre = (x_2 - x_1)**2 + (y_2 - y_1)**2
                if min_dist_pt[0] > dist_sqre:
                    min_dist_pt = (dist_sqre, [x_2, y_2])
            if min_dist_pt[0] < max_dist:
                # This stores the group number with it's corresponding index
                for i, x in enumerate(indices):
                    verified_indices.add((i, x))
                verified.add(tuple(min_dist_pt[1]))
    return verified, verified_indices


def hough_line_intersect(line, point):
    """
    :param line: [rho, theta]
    :param point: [x, y]
    :param tolerance: Represented by total pixels off to be considered intersecting a corner
    :return: distance to consider intersect
    """
    rho, theta = line
    x, y = point
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    distance = abs(x * cos_t + y * sin_t - rho)
    return distance


def get_most_common_dist(lines, min_count_threshold=1):
    """
    Get most common distance between lines to be used as reference of how big the grid square is
    :param lines:
    :return:
    """
    if len(lines) <= 1:
        return None

    lines_c = np.array([rho for rho, _ in lines])

    # GPT suggestion to use histogram
    gaps = np.diff(np.sort(lines_c))

    hist, bin_edges = np.histogram(gaps, bins='auto')
    max_count = np.max(hist)

    if max_count < min_count_threshold:
        # No dominant gap → fallback to median
        return np.median(gaps)
    else:
        # Use bin center of most common gap
        idx = np.argmax(hist)
        most_common = (bin_edges[idx] + bin_edges[idx + 1]) / 2
        return most_common


def line_interpolate(group1, group2, sect_list, line_pts, corners, threshold=10, image=None):
    """
    Interpolate lines by:
    1) Verifying lines with valid intersection by corners
    2) Present the corners points intersecting the lines verifed by the previous step.
    3) Extend out until number of lines for both group reaches 9
    :param group1:
    :param group2:
    :param sect_list: the intersection between the two groups represented as points
    :param line_pts: the intersection between lines and corner points
    :param corners: all corner points found by a corner detection function
    :param threshold:
    :return:
    """
    if group1 is None or group2 is None or len(group1[0]) < 2 or len(group2[0]) < 2:
        return None, None


    all_sects = []
    for row in sect_list:
        for line_tuple in row:
            if line_tuple is None:
                continue
            line_indices, sect = line_tuple
            all_sects.append(sect)
    verified, lines_by_i = intersect_verification(sect_list, corners)
    g1 = []
    g2 = []
    for i, line_index in lines_by_i:
        if i == 0:
            g1.append(group1[line_index])
        else:
            g2.append(group2[line_index])

    if image is not None:
        g1_copy = g1.copy()
        g1_copy.extend(g2)
        cd.find_exact_line(image, g1_copy, -1, green=False)

    line_sects = []
    show_sects = []
    for g1_line in g1:
        key = tuple(g1_line)
        if key in line_pts:
            line_sects.append(list(line_pts[key]))
            show_sects.extend(line_pts[key])
    for g2_line in g2:
        key = tuple(g2_line)
        if key in line_pts:
            line_sects.append(list(line_pts[key]))
            show_sects.extend(line_pts[key])

    # show_points(verified)
    # show_points([], all_sects)
    # show_points([], [], show_sects)
    # show_points([], [], [], corners)

    # Absolute values groups of lines for relatives location of lines
    best_gap_g1 = get_most_common_dist([(abs(rho), theta) for rho, theta in g1])
    best_gap_g2 = get_most_common_dist([(abs(rho), theta) for rho, theta in g2])

    # Second round, find expansion lines outwards if not enough corners inbetween


def show_points(points, points_2=[], points_3=[], points_4=[], height=1000, width=1000, image=None):
    # Create a blank grayscale image if none is provided
    if image is None:
        use_image = np.zeros((height, width, 3), dtype=np.uint8)  # 3 channels for color
    else:
        use_image = image.copy()

    # Draw each point as a small white circle (colors from GPT)
    for x, y in points:
        cv2.circle(use_image, (int(x), int(y)), radius=2, color=(255, 255, 255), thickness=-1)
    # Green for first `points_2`
    for x, y in points_2:
        cv2.circle(use_image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=1)
    # Yellow for second `points_2`
    for x, y in points_3:
        cv2.circle(use_image, (int(x), int(y)), radius=8, color=(0, 255, 255), thickness=1)
    for x, y in points_4:
        cv2.circle(use_image, (int(x), int(y)), radius=12, color=(100, 100, 100), thickness=1)

    # Show the image
    cv2.imshow("Verified Points", use_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
