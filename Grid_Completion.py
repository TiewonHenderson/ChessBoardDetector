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
    :return:
    """
    points = np.array(points)
    if len(points) < 2:
        return None
    diffs = np.diff(points, axis=0)
    step_x = np.median(diffs[:, 0])
    step_y = np.median(diffs[:, 1])

    return step_x, step_y


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
    # Uses O(1) search in key dict to see if sect is already in corners
    for sect in all_sects:
        if tuple(sect) in point_system:
            continue
        all_pts.append(sect)

    for sect in all_sects:
        sect_pt = tuple(sect)
        if sect_pt in point_system:
            continue
        # Either verified or re-verified
        if tree_verified and tree_verified.query(c_pt)[0] <= threshold or\
           tree_line_corners and tree_line_corners.query(c_pt)[0] <= threshold:
            point_system[c_pt] = 14
            continue
        score = 10   # from 2 + 8
        point_system[sect_pt] = score

    return point_system, build_tree(all_pts), all_pts


def inverse_quadratic_score(dist, step):
    """
    Uses Inverse Quadratic Decay formula in order see if the distance is good
    However if distance is more then half towards the next point, its prob bad or grid is heavily skewed
    :param dist:
    :param step:
    :return:
    """
    return 1 / (1 + (dist / step)**2)


def rank_row(row, true_index, score_system, points_tree, steps):
    """
    TO-DO
    Given a row of points, rank them by how related to the corners points they are.
    We do this because there is a possiblity more then 9 row was created.
    :param row:
    :param true_index: The index of a verified point to start off
    :param point_system:
    :param corner_tree: corners list converted to a KDtree
    :param steps: x and y offsets that were used to generate each point
    :return:
    """
    total_score = 0

    # Loop from good index to 0, allows for error adapting
    for i in range(true_index - 1, -1, -1):
        # Theses are nearest neighbor by distance squared
        steps_sq = steps[0]**2 + steps[1]**2
        steps_sq = max(math.isqrt(steps_sq), 1)
        neighbor_indices, neighbor_dists = top_k_neighbors(pts, points_tree)
        best_points = None
        for dist in neighbor_dists:
            if dist > (steps_sq + 1)//2:
                continue
        if best_points == None:
            continue
    return total_score


def fill_in_missing(row, points_tree, all_points, score_system, threshold=7):
    """
    :param row: 2d array of verified points
    :param points_tree: KDtree of all_points in order to even access the points
    :param all_points: all_points that are relative, corners, intersections, etc.
    :param score_system:
    :return:
    """
    if row is None or len(row) <= 1:
        return (-1, row)
    if len(row) >= 9:
        # Take to face value
        return (-1, row)
    p1 = np.array(row[0], dtype=float)
    p2 = np.array(row[-1], dtype=float)
    mask_list = []
    step_list = []
    for total_points in range(3, 10):  # From 3 to 9 inclusive
        step = (p2 - p1) / (total_points - 1)
        mask = [p1 + step * i for i in range(1, total_points - 1)]  # Skip the end points
        mask_list.append(mask)

    best_mask = (0, None)
    for i, mask in enumerate(mask_list):
        current_mask = [tuple(p1)]
        # Default score of 48 since it includes the two verified end points
        mask_score = 48
        for pts in mask:
            neighbor_indices, neighbor_dists = top_k_neighbors(pts, points_tree)
            """
            distance scoring formula 
            """
            best_point = None
            max_score = 0
            for j in range(len(neighbor_dists)):
                pt_index = neighbor_indices[j]
                pt_score = score_system[tuple(all_points[pt_index])]
                final_pt_score = pt_score * inverse_quadratic_score(neighbor_dists[j], threshold)
                if final_pt_score > max_score:
                    max_score = final_pt_score
                    best_point = all_points[pt_index]
            mask_score += max_score
            current_mask.append(best_point)
        current_mask.append(tuple(p2))
        show_points(current_mask, mask)
        if mask_score > best_mask[0]:
            best_mask = (mask_score, current_mask)
    return best_mask


def get_row_mask(row, corner_tree, score_system, steps=None):
    """
    Depending on how many points are within row, the masking process will be different:
    If len(row) <= 4, we would need to reconstruct the entire row.
    Create new mask from every two points

    If len(row) > 4, despite the possibility of missing points inbetween, this is
    still the majority of points present, use it as reference and only expand outwards
    :param row:
    :param corner_tree: corners list converted to a KDtree
    :param score_system:
    :param steps: A fail safe for row being just 1 point, the only place you could get steps
                  is from previous rows.
    :return:
    """
    def scoring_window(mask, start_index, step, score_system=score_system):
        """
        Assumes mask is at least 9 points long (should be at least 10 in theory)
        :param mask:
        :param start_index:
        :param steps: x and y offsets that were used to generate each point
        :param score_system:
        :return:
        """
        window = [0, 9]
        max_score = (0, None)
        while window[1] < len(mask):
            row_score = rank_row(mask[window[0]:window[1]], start_index,
                                 point_system=score_system,
                                 points_tree=points_tree)
            if row_score > max_score:
                max_score = (row_score, tuple(window))
        return max_score


    if row == None or len(row) == 0:
        return row
    if len(row) == 1:
        # Row is too short, need predetermined steps
        if steps is None:
            return row
        x_diff, y_diff = steps
        x_0, y_0 = row[0]
        extend_part = []
        mask = []
        # adds points in both directions
        for j in range(1, 9):
            pt_before = [x_0 - (j * x_diff), y_0 - (j * y_diff)]
            pt_after = [x_0 + (j * x_diff), y_0 + (j * y_diff)]
            mask.append(pt_before)
            extend_part.append(pt_after)
        # Before points are backwards, reverse then extend with afters points
        mask = mask[::-1]
        good_index = len(mask)
        mask.append(row[0])
        mask.extend(extend_part)
        max_score = scoring_window(mask)
        return max_score[1]

    x_d, y_d = get_best_dist_rep(row)
    if len(row) <= 4:
        """
        Two approaches:
        1) Generate masking over pairs of points until suffices
        2) Take the outer border of the current row, divide by 2-8 segments
           Fill in gaps if detected, then mask out
        """
        mask_list = [hcd.create_point_mask(row[i], row[i + 1], x_d, y_d)]



def mask_9x9(verified, steps):
    """
    Insert artifical points into a copy of verified to get a full 9x9 to find the nearest neighbor of corners
    :param verified: a matrix version of corner points, where points are in row
    :param steps:
    :return:
    """

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

    for row in mask:
        if len(row) == 9:
            continue
        elif len(row) > 9:
            continue


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
    :return:
    """
    if group1 is None or group2 is None or len(group1) < 2 or len(group2) < 2:
        return None, None

    corners = list(map(tuple, corners))
    verified, verified_lines = intersect_verification(sect_list, corners)
    flat_verified = [pt for row in verified for pt in row]
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

    all_sects, line_corners = get_points(sect_list, line_pts)
    score_system, all_pt_tree, all_pts = score_points(flat_verified, all_sects, line_corners, corners)

    # for row in verified:
    #     print(row)
    #     show_points(row)

    masked_grid = []
    for row in verified:
        masked_row = fill_in_missing(row, all_pt_tree, all_pts, score_system)
        masked_grid.extend(masked_row[1])

    # print(score_system)
    # print(all_pt_tree, "\nlen", all_pt_tree.n)
    # print(all_pts, "\nlen", len(all_pts))

    show_points(masked_grid, box=box)
    show_points(flat_verified, box=box)
    show_points(flat_verified, all_sects, line_corners, corners, lines=lines, box=box)


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
        for i in range(len(box)):
            pt1 = tuple(box[i])
            pt2 = tuple(box[(i + 1) % len(box)])
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
