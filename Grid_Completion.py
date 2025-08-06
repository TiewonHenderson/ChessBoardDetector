import sys
import numpy as np
import cv2
from math import sin,cos
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


def outlier_by_diff(points):
    """
    Claude generated to use second order difference between points
    :param points:
    :return:
    """
    points = sort_points_by_range(points)
    if len(points) < 4:  # Need at least 4 points for meaningful second diff
        return points
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    second_diff = np.diff(distances)
    # Find the biggest absolute change in second difference
    max_change_idx = np.argmax(np.abs(second_diff))
    point_to_remove_idx = max_change_idx + 1
    # Ensure we don't remove the last point or go out of bounds
    if point_to_remove_idx >= len(points) - 1:
        # Find alternative - look for second largest change that's not at the end
        second_diff_abs = np.abs(second_diff)
        second_diff_abs[max_change_idx] = -1  # Mask the largest
        # Find valid alternatives (not pointing to last point)
        valid_indices = []
        for i in range(len(second_diff_abs)):
            if second_diff_abs[i] > 0 and (i + 1) < len(points) - 1:
                valid_indices.append(i)
        if valid_indices:
            # Choose the one with highest remaining change
            best_alt_idx = max(valid_indices, key=lambda x: second_diff_abs[x])
            point_to_remove_idx = best_alt_idx + 1
        else:
            # No suitable point to remove
            return points
    # Actually remove the point
    updated_points = np.delete(points, point_to_remove_idx, axis=0)
    updated_distances = np.linalg.norm(np.diff(updated_points, axis=0), axis=1)
    return updated_points


def find_closest_set(ref_points, candidates):
    """
    GPT generated to see which points are closest to the reference points overall
    :param ref_points:
    :param candidates:
    :return:
    """
    ref_points = np.array(ref_points)  # shape: (9, 2)
    candidates = np.array(candidates)  # shape: (n, 9, 2)

    # Compute Euclidean distance between corresponding points
    diffs = candidates - ref_points[np.newaxis, :, :]  # shape: (n, 9, 2)
    dists = np.sum(diffs**2, axis=1)

    # Sum distances for each candidate set
    total_dists = np.sum(dists, axis=1)  # shape: (n,)

    # Find index of minimum total distance
    best_index = np.argmin(total_dists)
    best_set = candidates[best_index]

    return best_index, best_set


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


def get_points(sect_list, line_pts):
    all_sects = []
    all_line_pts = []
    for row in sect_list:
        for pt_info in row:
            if pt_info is not None:
                all_sects.append(pt_info[1])
    for key in line_pts:
        all_line_pts.extend(line_pts[key])
    return all_sects, all_line_pts


def line_interpolate(group1, group2, sect_list, line_pts, corners, lines, threshold=10, image=None):
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
    g1 = []
    g2 = []
    for key in verified_lines:
        i, line_index = key
        if i == 0:
            g1.append((line_index, group1[line_index]))
        else:
            g2.append((line_index, group2[line_index]))

    # Loop through the larger group to finish
    selected_group = g1
    group_num = 0
    if len(g1) < len(g2):
        group_num = 1
        selected_group = g2

    for index, _ in selected_group:
        verified_corners = np.array(verified_lines[(group_num, index)])
        reference_pts = None
        if len(verified_corners) > 9:
            # If too many points, remove by outlier diff with second order difference of distance
            while len(verified_corners) != 9:
                verified_corners = outlier_by_diff(verified_corners)
        elif len(verified_corners) < 9:
            # If too little points
            # Expand out with hcd masking and see which mask is best
            points_copy = np.array(verified_corners)
            step_diffs = np.diff(points_copy, axis=0)  # N-1
            for i, diff in enumerate(step_diffs):
                # Stage to get the interpolated points
                x_d, y_d = diff
                masking_pts = hcd.create_point_mask(points_copy[i], points_copy[i+1], x_d, y_d)
                for i in range(len(masking_pts)):

            if reference_pts is None:


    # if image is not None:
    #     g1_copy = g1.copy()
    #     g1_copy.extend(g2)
    #     cd.find_exact_line(image, g1_copy, -1, green=False)

    all_sects, line_corners = get_points(sect_list, line_pts)
    show_points(verified)
    # show_points([], all_sects)
    # show_points([], [], line_corners)
    # show_points([], [], [], corners)
    print(len(verified))
    show_points(verified, all_sects, line_corners, corners, lines=lines)


def show_points(points, points_2=[], points_3=[], points_4=[], height=1000, width=1000, image=None, lines=None):
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
