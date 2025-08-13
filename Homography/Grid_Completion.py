import sys
import numpy as np
import cv2
from math import sin,cos,ceil,sqrt
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
from sklearn.linear_model import RANSACRegressor
from ChessBoardDetector.Line_Point_detection import HarrisCornerDetection as hcd
from ChessBoardDetector.Line_Point_detection import HoughTransform as ht
from ChessBoardDetector import Chessboard_detection as cd
from ChessBoardDetector.Clustering import filter_grids as fg
from ChessBoardDetector.Remove_outliers import cv_filter_groups as cvfg

"""
Uses the concept of point interpolation by first:
verifiying the intersections of the found grid lines with the corners found.
Intersections are kept, but we only expand on the verified corners.

Ideally we would want to expand outwards in all directions, all at once.
OR
Expand row wise, but applied to all rows at once, whichever direction (left/right for explaination sake)
scores better, keep that entire column.

Once we have Nx9 points, we replicate the outer rows and expand outwards by row (up/down for explaination sake)
In order to get 9x9 points

In extremely rare cases where MxN and either M or N > 9, we would end up removing rows/cols
that scores the least based of the corners they represent.

We run cv homography RANSAC in order to get the final homography mask over the points, and shift
the four outer most corners around 1 "square" in order to see which includes the most
significant corners within the square boundaries.

Corners are scored:
corner = 2 points
corner thats part of a line = 6 points
intersection = 8 points
corner + intersection = 14 points
"""


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


def fill_in_missing(row, points_tree, all_points, score_system, dist_eps=0.1):
    """
    :param row: 2d array of verified points
    :param points_tree: KDtree of all_points in order to even access the points
    :param all_points: all_points that are relative, corners, intersections, etc.
    :return:
    """
    if row is None or len(row) <= 1:
        return (-1, row)
    if len(row) >= 9:
        # Take to face value
        return (-1, row)
    p1 = np.array(row[0], dtype=int)
    p2 = np.array(row[-1], dtype=int)
    mask_list = []
    step_list = []
    # Iterate WITH the current amount of points in mind
    for total_points in range(len(row) + 1, 10):  # can range from 3 to 9 inclusive
        step = (p2 - p1) / (total_points - 1)
        mask = [p1 + step * i for i in range(1, total_points - 1)]  # Skip the end points
        mask_list.append(mask)

    best_mask = (14 * len(row), row)
    for i, mask in enumerate(mask_list):
        current_mask = [tuple(p1)]
        # Default score of 28 since it includes the two verified end points
        mask_score = 28
        for pts in mask:
            neighbor_indices, neighbor_dists = top_k_neighbors(tuple(pts), points_tree)
            """
            distance scoring formula 
            """
            best_point = tuple(pts)
            max_score = 0
            for j in range(len(neighbor_dists)):
                pt_index = neighbor_indices[j]
                point = tuple(all_points[pt_index])
                dist_score = inverse_quadratic_score(neighbor_dists[j])
                if dist_score < dist_eps:
                    continue
                if point in score_system:
                    pt_score = score_system[point]
                else:
                    pt_score = 1
                final_pt_score = pt_score * dist_score
                if final_pt_score > max_score:
                    max_score = final_pt_score
                    best_point = tuple(map(int, point))
            mask_score += max_score
            current_mask.append(best_point)
        current_mask.append(tuple(p2))
        if mask_score > best_mask[0]:
            best_mask = (mask_score, current_mask)
    return best_mask


def choose_best_row_mask(row, true_index, score_system, points_tree, all_pts,
                         bounds=(1000,1000), dist_eps=0.1):
    """
    TO-DO
    Given a row of points, rank them by how related to the corners points they are.
    We do this because there is a possiblity more then 9 row was created.

    could use prev_gap, but would require sqrt operations
    :param row:
    :param true_index: The index of a verified point to start off
    :param score_system:
    :param points_tree: corners list converted to a KDtree
    :param all_pts: The list of points used to create points_tree
    :param dist_eps: scaling offset the distance between the mask point to the corner point to be accepted/too far
    :return:
    """

    def loop_check(indices, range_op):
        total_score = 0
        finalize_row = []
        # Loop from good index to 0, allows for error adapting
        # Expected inclusive indices = [0,8]
        for i in range(indices[0], indices[1], range_op):
            if out_of_bounds(row[i], bounds):
                return -1, []
            neighbor_indices, neighbor_dists = top_k_neighbors(tuple(row[i]), points_tree)
            best_points = (1, row[i])
            for i, dist in enumerate(neighbor_dists):
                point = all_pts[neighbor_indices[i]]
                dist_score = inverse_quadratic_score(dist)
                # mask point too far to actual point
                if dist_score < dist_eps:
                    continue
                if point in score_system:
                    # point should be in score_system
                    score = score_system[point] * inverse_quadratic_score(dist)
                else:
                    continue
                if score > best_points[0]:
                    best_points = (score, point)
            total_score += best_points[0]
            finalize_row.append(best_points[1])
        finalize_row = finalize_row[::-1]
        return total_score, finalize_row

    total_score = 0
    finalize_row = []
    s, f = loop_check((true_index, -1), -1)
    total_score += s
    finalize_row = f
    s, f = loop_check((true_index + 1, len(row)), 1)
    total_score += s
    finalize_row.extend(f)
    return total_score, finalize_row


def scoring_window(mask, points_tree, all_pts, score_system, start_index=7):
    """
    Assumes mask is at least 9 points long (should be at least 10 in theory)
    :param mask:
    :param points_tree: all points with a list converted to a KDtree
    :param all_pts: All points that represents corners and intersections
    :param score_system:
    :param start_index: where the starting point should be, and expand out both ways if needed
    :return:
    """
    window = [0, 9]
    max_score = (0, None)
    while window[1] < len(mask):
        row_score, new_row = choose_best_row_mask(mask[window[0]:window[1]], start_index,
                                                  score_system=score_system,
                                                  points_tree=points_tree,
                                                  all_pts=all_pts)
        if row_score > max_score[0]:
            max_score = (row_score, new_row, window.copy())
        window = [window[0] + 1, window[1] + 1]
    return max_score


def get_row_mask(row, points_tree, all_pts, score_system, dim=1000, steps=None):
    """
    Depending on how many points are within row, the masking process will be different:
    If len(row) <= 4, we would need to reconstruct the entire row.
    Create new mask from every two points

    If len(row) > 4, despite the possibility of missing points inbetween, this is
    still the majority of points present, use it as reference and only expand outwards
    :param row:
    :param points_tree: all points with a list converted to a KDtree
    :param all_pts: All points that represents corners and intersections
    :param score_system:
    :param dim: The dimension of the image, it should be 1:1 and default to 1000x1000
    :param steps: In respect to the previous interpolate, the following rows will do the same
    :return:
    """

    def filter_in_bounds(points, bounds=(dim,dim)):
        """
        Helper function generated by gpt to remove points out of bounds
        :param points:
        :param bounds:
        :return:
        """
        bounds = np.array(bounds)
        return points[
            (points[:, 0] >= 0) & (points[:, 0] < bounds[0]) &
            (points[:, 1] >= 0) & (points[:, 1] < bounds[1])
            ]

    row_len = len(row)
    # Cannot use same points to represent multiple (cause of duplication)
    if row == None or row_len <= 1 or row_len == 9:
        # Row too short
        return row
    if row_len < 9:
        """
        Two approaches:
        1) Generate masking over pairs of points until suffices
        
        Seems filling in gaps might be worse
        2) Take the outer border of the current row, divide by 2-8 segments
           Fill in gaps if detected, then mask out
        """
        mask_list = []
        for i in range(row_len - 1):
            diff = np.array(row[i]) - np.array(row[i + 1])
            x_d, y_d = diff[0], diff[1]
            gen_mask = hcd.create_point_mask(row[i], row[i + 1],
                                            x_d, y_d,
                                            True)
            if len(gen_mask) >= 9:
                mask_list.append(gen_mask)
        max_score = (0, row)
        for mask in mask_list:
            score_tuple = scoring_window(mask, points_tree, all_pts, score_system)
            if score_tuple[0] > max_score[0]:
                max_score = score_tuple
        return max_score[1]
    else:
        """
        Any bigger row is just another version of sliding window, except a few of them
        are misrepresented as verified points
        
        to-do ideas:
            1) Outlier gaps between points, remove until 9 points is reached
            2) DP the longest chain and expand by outer most points
        
        but for now, its unlikely removing an inner point will affect the homography too much
        """
        while len(row) > 9:
            row.pop(len(row)//2)
        return row


def full_row_step_diff(row_1, row_2, dir="right"):
    """
    Adds the difference between the previous row in order to interpolate a next row (simple method)
    Then snap the point to a known corner in score_system
    :param row_1: must be before row 2 in some order given (2d list of points)
    :param row_2: (2d list of points)
    :param dir: Direction of cloning the row towards by index
    :return:
    """
    row_a = np.array(row_1)
    row_b = np.array(row_2)
    if dir == "right":
        diff = row_b - row_a
        new_row = row_b + diff
    else:
        diff = row_a - row_b
        new_row = row_a + diff
    return new_row.tolist()


def mask_9x9(verified, points_tree, all_pts, score_system, dim=1000):
    """
    Insert artifical points into a copy of verified to get a full 9x9 to find the nearest neighbor of corners

    To-Do:
    Needs improvement in order to not include duplicate points

    :param verified: a matrix version of corner points, where points are in row
    :param points_tree: all points with a list converted to a KDtree
    :param all_pts: All points that represents corners and intersections
    :param score_system:
    :param dim: The dimension of the image, it should be 1:1 and default to 1000x1000
    :return:
    """

    def rank_row(row, score_system=score_system):
        """
        Given a row of points, rank them by how related to the corners points they are.
        We do this because there is a possiblity more then 9 row was created.
        :param row:
        :param point_system:
        :return:
        """
        total_score = 0
        for pts in row:
            total_score += score_system[tuple(pts)]
        return total_score

    mask = []
    # If theres two many rows then needed, remove some rows that score relatively low
    if len(verified) > 9:
        amt_row_removed = len(verified) - 9
        worst_rows = []
        for i in range(len(verified)):
            worst_rows.append((i, rank_row(verified[i])))
        # Sort and sublist to get top 9 rows in terms of scoring
        worst_rows = sorted(worst_rows, key=lambda x: x[1])
        worst_rows = worst_rows[amt_row_removed:]
        mask = [verified[i] for i, _ in worst_rows]
    else:
        mask = verified.copy()
    row_result = []
    for row in mask:
        row = sort_points_by_range(row)
        row_result.append(sort_points_by_range(get_row_mask(row, points_tree, all_pts, score_system)))

    if len(row_result) < 3:
        # Too little rows to be worth interpolating
        return []


    # Instead of transpose, we realign and expand by row instead
    while len(row_result) < 9:
        left = full_row_step_diff(row_result[0], row_result[1], "left")
        right = full_row_step_diff(row_result[-2], row_result[-1], "right")
        left_score, left = choose_best_row_mask(left, len(left)//2, score_system, points_tree, all_pts)
        right_score, right = choose_best_row_mask(right, len(right) // 2, score_system, points_tree, all_pts)
        left_valid = len(left) == 9
        right_valid = len(right) == 9
        if left_valid and right_valid:
            # Check score after all len of row suffices
            if right_score > left_score:
                row_result.append(right)
            else:
                row_result.insert(0, left)
        elif left_valid and not right_valid:
            row_result.insert(0, left)
        elif right_valid and not left_valid:
            row_result.append(right)
        else:
            break
    return row_result


def point_interpolate(group1, group2,
                      sect_list, line_pts, corners,
                      image_shape, threshold=10, image=None, lines=None):
    """
    Interpolate lines by:
    :param group1:
    :param group2:
    :param sect_list: the intersection between the two groups represented as points (are sorted in terms of rows)
    :param line_pts: the intersection between lines and corner points
    :param corners: all corner points found by a corner detection function
    :param image_shape: Used to make sure points are WITHIN the image
    :param threshold:
    :param image: OPTIONAL for display
    :param lines: OPTIONAL for display
    :return: Return the four corners that fits best from RANSAC
    """
    if group1 is None or group2 is None or len(group1) < 2 or len(group2) < 2:
        return None
    height, width = image_shape
    corners = list(map(tuple, corners))
    verified, verified_lines = intersect_verification(sect_list, corners)
    all_sects, line_corners = get_points(sect_list, line_pts)
    flat_verified = [pt for row in verified for pt in row]
    # show_points(flat_verified, all_sects, line_corners, corners, height=height, width=width)
    if len(flat_verified) <= 9:
        return None
    g1 = []
    g2 = []
    """
    sect_list is stored by row/col, 
    """
    for key in verified_lines:
        i, line_index = key
        if i == 0:
            g1.append((line_index, group1[line_index]))
        else:
            g2.append((line_index, group2[line_index]))

    # Expand bounding box until a 9x9 grid is found
    box = get_bounding_pts(flat_verified)
    dimension = ceil(sqrt(len(flat_verified)))
    missing_amt = abs(len(flat_verified) - dimension**2)

    score_system, all_pt_tree, all_pts = score_points(flat_verified, all_sects, line_corners, corners)

    """
    Old grid interpolation by rows of points
    """
    final_grid = mask_9x9(verified, all_pt_tree, all_pts, score_system, height)
    flat_grid = [pt for row in final_grid for pt in row]
    if len(flat_grid) != 81:
        print("Failed points")
        return None

    h_grid, grid_by_row = ransac_grids(final_grid, height)
    f_grid = [sort_points_by_range(row) for row in grid_by_row]
    best_score = (0, [])
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            moved_grid = offset_grid_simple_robust(f_grid, i, j)
            four_corners = [moved_grid[0][0], moved_grid[0][-1], moved_grid[-1][-1], moved_grid[-1][0]]
            score = all_pts_in_corners(all_pts, score_system, four_corners)
            if score > best_score[0]:
                best_score = (score, four_corners)
    return best_score[1]


def ransac_grids(grid_mask, dim=1000, threshold=5):
    """
    GPT generated
    :param grid_mask:
    :param dim: The dimension of the image
    :param threshold:
    :return:
    """
    grid_space = dim/10
    dst_grid = np.array([
        [grid_space + (grid_space * 0.8) * x, grid_space + (grid_space * 0.8) * y]
        for y in range(9)
        for x in range(9)
    ], dtype=np.float32)  # Shape (81, 2)

    if len(grid_mask) == 9:
        src_points = [pt for row in grid_mask for pt in row]
    else:
        src_points = grid_mask
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_grid, dtype=np.float32)

    H, inliers = cv2.findHomography(
        dst_points,             # points from the distorted image (your detected/masked points)
        src_points,             # perfect reference grid points
        cv2.RANSAC,
        threshold
    )

    full_grid_projected = cv2.perspectiveTransform(dst_grid.reshape(-1, 1, 2), H)
    full_grid_projected = full_grid_projected.reshape(-1, 2)
    full_grid_3d = full_grid_projected.reshape(9, 9, 2)
    full_grid_3d = full_grid_3d.astype(int)
    full_grid = [list(map(int, pt)) for pt in full_grid_projected]
    return full_grid, full_grid_3d


def offset_grid_simple_robust(grid_points_3d, offset_x_squares=0, offset_y_squares=0):
    """
    Claude generated
    Uses offset of the extreme four corners to move the whole grid
    """
    grid_3d = np.array(grid_points_3d)
    all_points = grid_3d.reshape(-1, 2)

    # Find the span in each direction
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])

    # Calculate spacing (total span divided by 8 squares)
    x_spacing_per_square = (x_max - x_min) / 8
    y_spacing_per_square = (y_max - y_min) / 8

    offset_vector = np.array([
        offset_x_squares * x_spacing_per_square,
        offset_y_squares * y_spacing_per_square
    ])

    offset_grid_3d = grid_3d + offset_vector

    return offset_grid_3d.astype(int)


def all_pts_in_corners(all_pts, score_system, four_corner):
    """
    GPT generated
    Used to see all the points within the border, its then totals the score
    :param all_pts:
    :param score_system:
    :param four_corner:
    :return:
    """
    # points: list of [x, y]
    points = np.array(all_pts)  # shape (N, 2)

    # your 4 boundary points
    boundary = np.array(four_corner)  # shape (4, 2)

    min_x, max_x = boundary[:, 0].min(), boundary[:, 0].max()
    min_y, max_y = boundary[:, 1].min(), boundary[:, 1].max()

    inside_mask = (
            (points[:, 0] >= min_x) &
            (points[:, 0] <= max_x) &
            (points[:, 1] >= min_y) &
            (points[:, 1] <= max_y)
    )

    inside_points = points[inside_mask]
    total_score = 0
    for in_pts in inside_points:
        if tuple(in_pts) in score_system:
            total_score += score_system[tuple(in_pts)]
    return total_score


def show_points(points, points_2=[], points_3=[], points_4=[],
                height=1000, width=1000, image=None, lines=None, box=None, thickness=1):
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
        for i in range(len(box)):
            pt1 = tuple(box[i])
            pt2 = tuple(box[(i + 1) % len(box)])
            cv2.line(use_image, pt1, pt2, (0, 255, 0), 2)
    if lines is not None:
        ht.put_lines(lines, use_image, (0, 0, 255), 1)

    # Draw each point as a small white circle (colors from GPT)
    for x, y in points:
        cv2.circle(use_image, (int(x), int(y)), radius=2, color=(255, 255, 255), thickness=3)
    # Green for first `points_2`
    for x, y in points_2:
        cv2.circle(use_image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=thickness)
    # Yellow for second `points_2`
    for x, y in points_3:
        cv2.circle(use_image, (int(x), int(y)), radius=8, color=(0, 255, 255), thickness=thickness)
    for x, y in points_4:
        cv2.circle(use_image, (int(x), int(y)), radius=12, color=(100, 100, 100), thickness=thickness)


    # Show the image
    cv2.imshow("Verified Points", use_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
