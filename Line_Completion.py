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

    return abnormal_jumps


def neighbor_corners_by_line(line_pt1, line_pt2, all_pts, threshold=5.0):
    """
    GPT generated in order to get the closest corners to the given line
    :param line_pt1: One end of the line
    :param line_pt2: Other end of the line (must not be the same as pt1)
    :param all_pts: list of all relevant points
    :param threshold: How much distance off to consider the corner to be part of the line
    :return:
    """
    p1 = np.array(line_pt1, dtype=float)
    p2 = np.array(line_pt2, dtype=float)
    v = p2 - p1  # direction vector of the line
    all_pts = np.array(all_pts)

    # Compute perpendicular distances for all corners at once
    num = np.abs((v[1]) * (all_pts[:, 0] - p1[0]) - (v[0]) * (all_pts[:, 1] - p1[1]))
    denom = np.hypot(v[0], v[1])
    distances = num / denom

    # Filter corners within threshold
    mask = distances <= threshold
    filtered_corners = all_pts[mask]
    filtered_distances = distances[mask]

    # Return list of (point, distance) tuples
    return list(zip(map(tuple, filtered_corners), filtered_distances))


def steps_and_accel(pt_row, needs):
    """
    Uses steps and acceleration by getting the first and second order difference between points
    Assumes pt_row is sorted
    :param pt_row: List of single points
    :param needs:
    :return: two list representing mask expansion beyond left and right of index 0, -1 respectively
    """
    pts = np.array(pt_row)
    pts_diff = np.diff(pts, axis=0)
    pts_accel = np.diff(pts_diff, axis=0)

    # Left/Right doesn't mean in respect to image, but by index
    left = []
    right = []

    # Uses far end acceleration values since thats the closest to the outer points
    left_diff = pts_diff[0] + pts_accel[0]
    right_diff = pts_diff[-1] + pts_accel[-1]
    gotten = 0
    left_ref_pt = pts[0]
    right_ref_pt = pts[-1]
    while gotten < needs:
        left.append(list(left_ref_pt - left_diff))
        right.append(list(right_ref_pt + right_diff))
        # Updates with same acceleration
        left_diff += pts_accel[0]
        right_diff += pts_accel[-1]
        # Use updated points as reference
        left_ref_pt = left[-1]
        right_ref_pt = right[-1]

        gotten += 1
    return left, right


# Convert polar lines to image bounded points (2 points to represent 1 line)
def get_outer_points(sect_list):
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
            if row[i] is not None:
                pt1 = row[i][1]
                break
        for i in range(len(row)-1, -1, -1):
            if row[i] is not None:
                pt2 = row[i][1]
                break
        if pt1 is not None and pt2 is not None:
            g1_pts.append((pt1, pt2))
    for i in range(len(sect_list[0])):
        pt1, pt2 = (), ()
        for j in range(len(sect_list)):
            if sect_list[j][i] is not None:
                pt1 = sect_list[j][i][1]
                break
        for j in range(len(sect_list) - 1, -1, -1):
            if sect_list[j][i] is not None:
                pt2 = sect_list[j][i][1]
                break
        if pt1 is not None and pt2 is not None:
            g2_pts.append((pt1, pt2))
    return g1_pts, g2_pts


# Insert lines inbetween if needed
def insert_lines(g_pts):
    """
    Similar to fill in missing points for grid completion, this function inserts lines between gaps
    that offset the local threshold the most, since those take priority
    :param g_pts: A list of point pairs that represents the end points of lines
    :return:
    """
    pt1_row = [end_pts[0] for end_pts in g_pts]
    pt2_row = [end_pts[1] for end_pts in g_pts]
    total_need = 9 - len(g_pts)
    if total_need <= 0:
        return pt1_row, pt2_row
    insert = sorted(find_abnormal_jumps_dynamic(pt1_row), key=lambda x: x[2], reverse=True)
    for needed_lines in insert:
        indices, amt, _ = needed_lines
        # insert is sorted from max to min in terms of how off the gap between outer points was
        # So insertion of lines are done on more important gaps if any
        ref_pt_1, ref_pt_2 = np.array(pt1_row[indices[0]]), np.array(pt1_row[indices[1]])
        oth_pt_1, oth_pt_2 = np.array(pt2_row[indices[0]]), np.array(pt2_row[indices[1]])
        inserted_pts = []
        needed_index = indices[1]
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
    """
    At this point, pt1 and pt2 should be sorted throughout the entire process
    Uses
    :param pt1_row: List of single point representing the row of endpoints for a row of lines
    :param pt2_row: List of single point representing the row of endpoints corresponding to pt1_row
    :return: 2 tuples representing expansion on both ends of the given row pts,
             The data structure is:
            (left_1, right_1)
            left_1 = List of single point in expansion toward the direction in name to ptx_row
            So in this case it expanded left from index 0 of pt1_row
    """
    masks = []
    total_need = 9 - len(pt1_row)
    while total_need <= 0:
        if total_need == 0:
            return pt1_row, pt2_row, 0
        # It almost guaranteed theres outlier lines, which can somewhat detected by find_abnormal_jumps_dynamic
        # but tailored to any kind of outlier (or even worst scoring ones)
        pt1_row.pop(len(pt1_row)//2)
        pt2_row.pop(len(pt2_row)//2)
        total_need += 1

    # Expand lines outwards
    left_1, right_1 = steps_and_accel(pt1_row, total_need)
    left_2, right_2 = steps_and_accel(pt2_row, total_need)
    return (left_1, left_2), (right_1, right_2), total_need


# Save top x lines
def get_best_k_lines(left_1, left_2, right_1, right_2, all_pts, score_system, k):
    """
    Use results from mask_lines, this still only applies to one cluster of lines
    :param left_1: List of single point in expansion toward left from index 0 (row1)
    :param left_2: List of single point in expansion toward left from index 0 (row2)
    :param right_1: List of single point in expansion toward right from index -1 (row1)
    :param right_2: List of single point in expansion toward right from index -1 (row2)
    :param all_pts: all relative points
    :param score_system: the score/importance of all points
    :param k: how many lines needed for a full 9x9 grid
    :return:
    """
    def search_line(pt1, pt2, all_pts=all_pts, score_system=score_system):
        """
        Helper function to check
        :param pt1:
        :param pt2:
        :param all_pts:
        :param score_system:
        :return:
        """
        score = []
        if len(pt1) == len(pt2):
            # Iterate each pair of points and check if near corners suffic
            for i in range(len(pt1)):
                nearest_pts = neighbor_corners_by_line(pt1[i], pt2[i], all_pts)
                curr_score = 0
                for point in nearest_pts:
                    point = tuple(point)
                    if point in score_system:
                        curr_score += score_system[point]
                score.append((curr_score, pt1[i], pt2[i]))
        return score

    score = search_line(left_1, left_2)
    score.extend(search_line(right_1, right_2))
    score.sort(key=lambda x: x[0], reverse=True)
    top_k_lines = []
    for i in range(k):
        _, pt1, pt2 = score[i]
        top_k_lines.append((tuple(pt1), tuple(pt2)))
    return top_k_lines


def show_lines(group_pts, image):
    img = image.copy()
    for i in range(len(group_pts)):
        cv2.line(img, group_pts[i][0], group_pts[i][1], color=(255, 0, 0), thickness=1)
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Intersection of each lines

