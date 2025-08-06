import sys
import numpy as np
import cv2
from math import sin,cos,ceil,sqrt
from scipy.optimize import curve_fit
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
    We also want the best representative line
    :param sect_list:
    :param corner:
    :param threshold: 7 pixel distance is the default offset of points to be considered the same
    :return:
    """

    """
    verified stores all intersections between lines verified by corner points
    verified_lines stores lines by their verified points
    """
    verified = set()
    verified_lines = {}
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
                found_corner = tuple(min_dist_pt[1])
                for i, x in enumerate(indices):
                    verified_lines.setdefault((i, x), []).append(found_corner)
                verified.add(found_corner)
    return verified, verified_lines


def get_bounding_square(points):
    """
    gpt generated
    :param points:
    :return:
    """
    # Convert to required format (e.g., float32 for OpenCV)
    points = np.array(points, dtype=np.float32)

    rect = cv2.minAreaRect(points)  # returns ((center_x, center_y), (width, height), angle)
    box = cv2.boxPoints(rect)  # gets 4 corner points of the rotated rect
    box = np.int0(box)  # convert to int if needed
    return box


def sort_points_by_range(points):
    """
        Claude generated to quickly sort by x/y first by bigger range
        then sort by the other factor
    """
    points = np.array(points)
    if points.shape[1] != 2:
        raise ValueError("Points must be 2D coordinates (N, 2)")
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    if x_range > y_range:
        sorted_indices = np.lexsort((y_coords, x_coords))
    else:
        sorted_indices = np.lexsort((x_coords, y_coords))
    sorted_points = points[sorted_indices]
    return sorted_points


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


def get_points(sect_list, line_pts, get_set=False):
    """
    Help get the points from both list
    :param sect_list:
    :param line_pts:
    :return:
    """
    all_sects = []
    all_line_pts = []
    if get_set:
        all_sects = set()
        all_line_pts = set()
    for row in sect_list:
        for pt_info in row:
            if pt_info is not None:
                if get_set:
                    all_sects.add(tuple(pt_info[1]))
                else:
                    all_sects.append(pt_info[1])
    for key in line_pts:
        if get_set:
            for line in line_pts[key]:
                all_line_pts.add(tuple(line))
        else:
            all_line_pts.extend(line_pts[key])
    return all_sects, all_line_pts


def get_best_dist_rep(points):
    """
    Gets a median of steps between x and y to be used for point interpolation
    :param points:
    :return:
    """
    points = np.array(points)
    if len(points) < 2:
        return None
    diffs = np.diff(points, axis=0)
    step_x = np.median(diffs[:, 0])
    step_y = np.median(diffs[:, 1])

    return np.array([step_x, step_y])


def score_points(verified, all_sects, line_corners, corners):
    """
    Scores points base on how many lists they are in
    verified overrules all as being the highest scoring points, so they aren't considered
    scoring:
    corner      line_corner     line_sect
    2           +4              *4
    :param verified:
    :param all_sects:
    :param line_corners:
    :param corners:
    :return:
    """
    point_system = {}
    for c in corners:
        c_pt = tuple(c)
        score = 2
        if c_pt in line_corners:
            score += 4
        if c_pt in all_sects:
            score *= 4
        point_system[c_pt] = score
    return point_system


def mask_9x9(verified, g1_steps, g2_steps):
    """
    Insert artifical points into a copy of verified to get a full 9x9 to find the nearest neighbor of corners
    :param verified:
    :param g1_steps:
    :param g2_steps:
    :return:
    """


def point_interpolate(group1, group2, sect_list, line_pts, corners, lines, threshold=10, image=None):
    """
    Interpolate lines by:
    :param group1:
    :param group2:
    :param sect_list: the intersection between the two groups represented as points
    :param line_pts: the intersection between lines and corner points
    :param corners: all corner points found by a corner detection function
    :param lines:
    :param threshold:
    :param image: OPTIONAL for display
    :return:
    """
    if group1 is None or group2 is None or len(group1) < 2 or len(group2) < 2:
        return None, None

    verified, verified_lines = intersect_verification(sect_list, corners)
    verified_list = list(verified)
    g1 = []
    g2 = []
    max_g1 = []
    max_g2 = []
    for key in verified_lines:
        i, line_index = key
        if i == 0:
            if len(verified_lines[key]) > len(max_g1):
                max_g1 = verified_lines[key]
            g1.append((line_index, group1[line_index]))
        else:
            if len(verified_lines[key]) > len(max_g2):
                max_g2 = verified_lines[key]
            g2.append((line_index, group2[line_index]))

    g1_steps = get_best_dist_rep(max_g1)
    g2_steps = get_best_dist_rep(max_g2)

    # Expand bounding box until a 9x9 grid is found
    box = get_bounding_square(verified_list)
    dimension = ceil(sqrt(len(verified_list)))
    missing_amt = abs(len(verified_list) - dimension**2)

    # Find high scoring points within the box first
    # while missing_amt > 0:

    # if image is not None:
    #     g1_copy = g1.copy()
    #     g1_copy.extend(g2)
    #     cd.find_exact_line(image, g1_copy, -1, green=False)

    all_sects, line_corners = get_points(sect_list, line_pts, True)
    show_points(verified, box=box)
    # show_points([], all_sects)
    # show_points([], [], line_corners)
    # show_points([], [], [], corners)
    print(len(verified))
    show_points(verified, all_sects, line_corners, corners, lines=lines, box=box)


def show_points(points, points_2=[], points_3=[], points_4=[],
                height=1000, width=1000, image=None, lines=None, box=None):
    """
    for displaying, not useful for true corners
    :param points:
    :param points_2:
    :param points_3:
    :param points_4:
    :param height:
    :param width:
    :param image:
    :return:
    """
    # Create a blank grayscale image if none is provided
    if image is None:
        use_image = np.zeros((height, width, 3), dtype=np.uint8)  # 3 channels for color
    else:
        use_image = image.copy()

    # Draw lines between each corner
    if box is not None:
        for i in range(4):
            pt1 = tuple(box[i])
            pt2 = tuple(box[(i + 1) % 4])
            cv2.line(use_image, pt1, pt2, (0, 255, 0), 2)
    if lines is not None:
        ht.put_lines(lines, use_image, (0, 0, 255), 1)

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
