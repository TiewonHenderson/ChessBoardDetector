import sys
import numpy as np
import cv2
from math import sin,cos,ceil,sqrt
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
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
    verified = []
    verified_lines = {}
    max_dist = threshold ** 2
    for row in sect_list:
        verified.append([])
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
                verified[-1].append(found_corner)
        if len(verified[-1]) == 0:
            verified.pop(-1)
    return verified, verified_lines


def get_bounding_pts(points):
    """
    gpt generated
    :param points:
    :return:
    """
    # Convert to required format (e.g., float32 for OpenCV)
    points = np.array(points, dtype=np.float32)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    points_list = hull_points.astype(int).tolist()
    return points_list


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
    return sorted_points.tolist()


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
            # Filters out None intersections
            if pt_info is not None:
                if get_set:
                    all_sects.add(tuple(pt_info[1]))
                else:
                    all_sects.append(tuple(pt_info[1]))
    for key in line_pts:
        if get_set:
            for line in line_pts[key]:
                all_line_pts.add(tuple(line))
        else:
            all_line_pts.extend(tuple(line_pts[key]))
    return all_sects, all_line_pts


def get_best_dist_rep(points):
    """
    Gets a median of steps between x and y to be used for point interpolation
    :param points:
    :return: min, med, and max distance between points values
    """
    points = np.array(points)
    if len(points) < 2:
        return None
    diffs = np.diff(points, axis=0)
    sqre_dist = np.sum(diffs**2, axis=1)  # squared distances

    # Indices for min, max, and median
    min_idx = np.argmin(sqre_dist)
    max_idx = np.argmax(sqre_dist)
    median_idx = np.argsort(sqre_dist)[len(sqre_dist)//2]
    return (diffs[min_idx], diffs[median_idx], diffs[max_idx])


def inverse_quadratic_score(dist, eps=5):
    """
    Uses Inverse Quadratic Decay formula in order see if the distance is good
    However if distance is more then half towards the next point, its prob bad or grid is heavily skewed
    0.1     =   ~15 pixels off
    0.05    =   ~22 pixels off
    :param dist:
    :param eps:
    :return:
    """
    return 1 / (1 + (dist / eps)**2)


def out_of_bounds(pt, bounds=(1000,1000)):
    """
    Helper function to see if the points is out of bounds
    No reason for the chessboard to be out of image
    :param pt:
    :param bounds: default dimension
    :return:
    """
    x, y = pt
    if x >= bounds[0] or x < 0 or\
       y >= bounds[1] or y < 0:
        return True
    return False


def top_k_neighbors(point, points_tree, k=5):
    """
    GPT generated
    Finds the k nearest neighbors to a single point using a prebuilt KDTree.

    :param point: A single 2D point (tuple or list or np.array of shape (2,))
    :param corner_tree: A KDTree built from corner points
    :param k: Number of neighbors to find
    :return: two 1D arrays that represent indices within corner_tree and
             euclidean distance to those indices from point
    """
    point = np.array(point).reshape(1, -1)  # Ensure shape (1, 2)
    dists, indices = points_tree.query(point, k=k)
    neighbor_dists = dists[0]
    neighbor_indices = indices[0]
    return neighbor_indices, neighbor_dists


def score_points(flat_verified, all_sects, line_corners, corners, threshold=4):
    """
    sets doesn't work since it needs exact points, use KDTree (GPT given syntax)

    Scores points base on how many lists they are in
    verified overrules all as being the highest scoring points, so they aren't considered
    scoring:
    corner      line_corner     line_sect
    2           +4              +8
    :param verified:
    :param all_sects:
    :param line_corners:
    :param corners:
    :param threshold: threshold here is stricter then intersect_verification because its based
                      corners pts not intersection, intersection by houghline are more inconsistent and
                      intersections are a lot more important then general corner pts.
    :return:
    """
    def build_tree(points):
        return KDTree(np.array(points)) if points else None

    tree_verified = build_tree(flat_verified)
    tree_all_sects = build_tree(all_sects)
    tree_line_corners = build_tree(line_corners)

    point_system = {}
    for c in corners:
        c_pt = tuple(c)
        if tree_verified and tree_verified.query(c_pt)[0] <= threshold:
            point_system[c_pt] = 14
            continue
        score = 2
        if tree_line_corners and tree_line_corners.query(c_pt)[0] <= threshold:
            score += 4
        if tree_all_sects and tree_all_sects.query(c_pt)[0] <= threshold:
            score += 8

        point_system[c_pt] = score

    all_pts = corners.copy()

    for sect in all_sects:
        sect_pt = tuple(sect)
        if sect_pt in point_system:
            continue
        all_pts.append(sect)
        # Either verified or re-verified
        if tree_verified and tree_verified.query(c_pt)[0] <= threshold or\
           tree_line_corners and tree_line_corners.query(c_pt)[0] <= threshold:
            point_system[c_pt] = 14
            continue
        score = 10   # from 2 + 8
        point_system[sect_pt] = score

    return point_system, build_tree(all_pts), all_pts


def find_abnormal_jumps_dynamic(points, window_size=3, base_factor=1.5):
    """
    GPT generated
    Detects abnormal jumps in consecutive points using a dynamic threshold
    based on local median spacing in a sliding window.

    :param points: list or Nx2 array of points (already sorted in order along row/column)
    :param window_size: number of neighboring gaps on each side to compute local median spacing
    :param base_factor: multiplier over local median spacing to flag a jump
    :return: list of ((indices), amount of lines needed, how off threshold is) where jump is abnormal
    """
    pts = np.array(points)
    diffs = np.diff(pts, axis=0)
    dists = diffs[:, 0] ** 2 + diffs[:, 1] ** 2  # squared distances for efficiency

    abnormal_jumps = []

    for i, dist_sq in enumerate(dists):
        # Define window boundaries
        start = max(0, i - window_size)
        end = min(len(dists), i + window_size + 1)
        local_median = np.median(dists[start:end])
        local_threshold = local_median * (base_factor ** 2)  # squared factor

        if dist_sq > local_threshold:
            needed_lines = round(dist_sq/local_median)
            abnormal_jumps.append(((i, i+1), needed_lines, local_threshold-dist_sq))

    return abnormal_jumps, dists


# Convert polar lines to image bounded points (2 points to represent 1 line)
def get_outer_points(g1, g2, sect_list):
    """
    Gets the border points created by the intersection of the given lines and the border
    :param g1:
    :param g2:
    :param sect_list: The entire matrix of intersection
    :return: Two groups (perserved indices) of lines represented
             as two points (x1,y1),(x2,y2) instead of rho,theta
    """
    g1_pts, g2_pts = [], []
    for row in sect_list:
        pt1, pt2 = None, None
        for i in range(len(row)):
            if row[i][1] is not None:
                pt1 = row[i][1]
                break
        for i in range(len(row)-1, -1, -1):
            if row[i][1] is not None:
                pt2 = row[i][1]
                break
        if pt1 is not None and pt2 is not None:
            g1_pts.append((pt1, pt2))
    for i in range(len(sect_list[0])):
        pt1, pt2 = (), ()
        for j in range(len(sect_list)):
            if sect_list[j][i][1] is not None:
                pt1 = sect_list[j][i][1]
                break
        for j in range(len(sect_list) - 1, -1, -1):
            if sect_list[j][i][1] is not None:
                pt2 = sect_list[j][i][1]
                break
        if pt1 is not None and pt2 is not None:
            g1_pts.append((pt1, pt2))
    return g1_pts, g2_pts


# Insert lines inbetween if needed
def insert_lines(g_pts):
    pt1_row = [tuple(end_pts[0]) for end_pts in g_pts]
    pt2_row = [tuple(end_pts[1]) for end_pts in g_pts]
    total_need = 9 - len(pt1_row)
    if total_need <= 0:
        return pt1_row, pt2_row
    insert = sorted(find_abnormal_jumps_dynamic(pt1_row), key=lambda x: x[2], reverse=True)
    for needed_lines in insert:
        indices, amt = needed_lines
        # insert is sorted from max to min in terms of how off the gap between outer points was
        # So insertion of lines are done on more important gaps if any
        ref_pt_1, ref_pt_2 = np.array(pt1_row[indices[0]]), np.array(pt1_row[indices[1]])
        oth_pt_1, oth_pt_2 = np.array(pt2_row[indices[0]]), np.array(pt2_row[indices[1]])
        inserted_pts = []
        needed_index = indices[0]
        for i in range(1, amt + 1):
            alpha = i / (amt + 1)
            # An addition way to get points divided by amt
            new_pt = (1 - alpha) * ref_pt_1 + alpha * ref_pt_2
            oth_new_pt = (1 - alpha) * oth_pt_1 + alpha * oth_pt_2
            pt1_row.insert(needed_index, new_pt)
            pt2_row.insert(needed_index, oth_new_pt)
            needed_index += 1
            total_need -= 1
            if total_need <= 0:
                return pt1_row, pt2_row

    return pt1_row, pt2_row


# Mask over the line, at most 7 different possibilities
def mask_lines(pt1_row, pt2_row):
    print(1)

# Intersection of each lines

