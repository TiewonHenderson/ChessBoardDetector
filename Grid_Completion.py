import sys
import numpy as np
from math import sin,cos
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import filter_grids as fg


def cv_check_grid(group1, group2):
    """
    Scoring:
    1) Gap average in group1 and group2 is relatively the same
    2) Evenly spaced gap between each intersection
    3) Amount of lines


    :param group1:
    :param group2:
    :return:
    """
    # Intersection matrix
    sect_matrix = []
    # group 2 would be column, group 1 would be rows
    for j, line_j in enumerate(group2):
        sect_matrix.append([])
        for line_i in group1:
            sect, _ = intersection_polar_lines(line_i, line_j)
            sect_matrix[j].append(sect)

def insert_dummy_lines(groups):
    """
    Currently doesn't extend with the original lines
    :param groups: Expected 2 lists of lists containing houghline [rho, theta] values, with list sorted by rhos
    :return:
    """
    if len(groups) != 2 or len(groups[0]) == 0 or len(groups[1]) == 0:
        return None

    dummies = [[], []]
    complete_group = [[], []]
    for i in range(len(groups)):
        # How many lines needs to be artifically inserted
        needed_lines = 8 - len(groups[i])
        if needed_lines <= 0:
            continue
        temp_group = []
        left = -needed_lines
        right = -1
        min_line = groups[i][0]
        max_line = groups[i][-1]
        # gap_mean is used to insert dummy lines with multipled gaps to form a uniform group of lines
        # Means of theta is used to generate lines with this theta
        gap_mean = 0
        theta_mean = 0
        for j in range(len(groups[i])):
            rho, theta = groups[i][j]
            theta_mean += theta
            if j == len(groups[i]) - 1:
                break
            gap_mean += abs(groups[i][j + 1][0] - rho)
        theta_mean /= len(groups[i])
        gap_mean /= len(groups[i]) - 1
        for created in range(needed_lines + 1):
            # need interval from [-n, -1] to represent needed gaps
            # cannot include 0, and [-n, -1] represents subtracting gap by min rho
            # [1, n] represents adding gap to max rho
            left = -needed_lines + created
            right = created
            if left == 0:
                left = 1
            if right == 0:
                right = -1
            temp_group.append([left, right])
        # Adds dummy lines into each group
        for interval in temp_group:
            left, right = interval
            dummies[i].append([])
            complete_group[i].append([])
            # The reference rho to offset off of,
            # its min_line's rho for negative multiples
            # max_line's rho for positive multiples
            ref_rho = min_line[0]
            for multiple in range(left, right + 1):
                if multiple == 0:
                    continue
                # If interval has 1 included, put copy before adding any dummy lines
                # This doesn't work for ending -1, thats extended later
                if multiple == 1 or left == multiple == 1:
                    complete_group[i][-1].extend(groups[i].copy())
                    ref_rho = max_line[0]
                complete_group[i][-1].append([ref_rho + (multiple * gap_mean), theta_mean])
                dummies[i][-1].append([ref_rho + (multiple * gap_mean), theta_mean])
            # Extend for [-n, -1] interval
            complete_group[i][0].extend(groups[i].copy())

    # for clusters in dummies:
    #     print("cluster")
    #     for groups in clusters:
    #         print(len(groups))
    #         print(groups)

    return dummies, complete_group


def closest_line(line_list, line):
    """
    Given a list of houghlines, this function linearly search through to find the closest line
    in terms of normalizing rho and theta differences between the houghline and given line with
    the range of line_list

    Combining both normalized rho and theta to meet below the threshold to be accepted
    :param line_list: a list of houghlines represented as [rho, theta]
    :param line: The polar coordinate line to search through line_list for most similar line under threshold
    :return: A index within line_list representing the most similar line, None if it doesn't exist,
             difference represents the lowest difference found throughout the line,
             if threshold was greater then returned difference, a line WOULD be returned
    """
    if len(line_list) == 0 or line is None:
        return None, None

    min_distance = sys.maxsize
    min_index = -1

    # Find the ranges for normalization
    rhos = [rho for rho, _ in line_list]
    thetas = [fg.normalize_theta(theta) for _, theta in line_list]

    # AI suggest to find range to scale difference by the two elements
    rho_range = max(rhos) - min(rhos) if max(rhos) != min(rhos) else 1

    # For line orientation, A line can only wrap around pi to become itself again
    theta_range = np.pi

    for i in range(len(line_list)):
        rho, theta = rhos[i], thetas[i]
        diff_rho = abs(rho - line[0]) / rho_range
        # This would get orientation wise minimum difference between the line
        theta_diff = abs(theta - line[1])
        theta_diff = min(theta_diff, np.pi - theta_diff)

        diff_theta = theta_diff / theta_range

        distance = diff_rho + diff_theta

        if distance < min_distance:
            min_distance = distance
            min_index = i

    return min_index, min_distance


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


def check_dummy_lines(dummies, lines, corners):
    """
    It's almost guaranteed the dummy lines would form a grid, instead they'll be graded on:
    1) If there is an near enough line similar to the dummy lines (With more lines missing, epsilon is lowered)
    This feature would prove weak with lenient threshold on houhgline
    2) If they intersect the harris corner points that were detected

    :param dummies: A 3D list of 2 clusters filled with 2d arrays (groups) of rho, theta lines
    :param lines: A 2D list of rho, theta representing all lines found from houghline
    :param corners: A 2D list of points representing cartesian points
    :return:
    """
    if len(lines) < 20:
        None
    scores = [[] for i in range(len(dummies))]
    best_lines = []
    for i, cluster_i in enumerate(dummies):
        for j, groups in enumerate(cluster_i):
            # Dummy line outcome will store a list of
            # [ closest line threshold needed (to see if a houghline can verify the line exist),
            #   corners that were considered intersected, check the gap consistency afterwards
            # ]
            dummy_line_outcome = []
            usable_lines = lines.copy()
            for k, l in enumerate(groups):
                # The more lenient the threshold, the less score given
                # If no lines were found, it will scale heavily

                # HAS TO return a closest line, we need 8x8 lines to form a chessboard grid
                min_index, difference = closest_line(usable_lines, l)
                dummies[i][j][k] = usable_lines[min_index]
                usable_lines.pop(min_index)

                # Threshold for corner intersection: pixel_offset -> [1, 5] bounds inclusive
                intersect_corners = []
                top_corners = closest_corner(corners, l)
                for dist, c in top_corners:
                    if dist <= 5:
                        intersect_corners.append(c)

                # Two intersections will cause inflated gap variance score
                if len(intersect_corners) > 2:
                    dummy_line_outcome.append([difference, len(intersect_corners),
                                                            hcd.consistent_gaps(intersect_corners, True)])
                else:
                    dummy_line_outcome.append([difference, len(intersect_corners), None])

            # Averages the outcome from difference (threshold needed for the nearest line)
            # Scores Harris corners intersection using coefficient variance (less variance = better)
            # Less then 2 corners makes no sense to get the CV, use amount of corners intersect instead
            # < 0.2 is good, < 0.275 is acceptable
            line_score = 0
            corner_score = 0
            for outcome in dummy_line_outcome:
                difference, corner_num, corner_cv = outcome
                line_score += max(0, 0.5 - difference)
                if corner_num <= 2:
                    corner_score += 0.1 * corner_num
                else:
                    corner_score += corner_num * (0.3 * max(0, 1 - (corner_cv / 0.25) ** 4))
            line_score /= len(dummy_line_outcome)
            corner_score /= len(dummy_line_outcome)

            # Final score just the average of the two, might change in the future
            final_score = (line_score + corner_score)/ 2
            scores[i].append(final_score)
        # Gets the min (represents the closest to existing lines and least variance)
        best_lines.append(scores[i].index(max(scores[i])))

    return best_lines


def grid_interpolation(groups, lines, corners):
    """

    :param groups:
    :param lines:
    :param corners:
    :return:
    """
    g1, g2 = groups
    dummies, grid_groups = insert_dummy_lines(groups)
    labels = check_dummy_lines(grid_groups, lines, corners)
    print("len1:", len(grid_groups[0][labels[0]]))
    print(grid_groups[0][labels[0]])
    print("len2:", len(grid_groups[0][labels[0]]))
    print(grid_groups[1][labels[1]])
    return [grid_groups[0][labels[0]], grid_groups[1][labels[1]]]


def main():
    x = [[[650.0, 1.5707963705062866],
          [756.0, 1.5707963705062866],
          [865.0, 1.5882495641708374]],
         [[405.0, 0.2617993950843811],
          [483.0, 0.20943951606750488],
          [562.0, 0.15707963705062866],
          [641.0, 0.10471975803375244]]]
    dummies = insert_dummy_lines(x)


if __name__ == "__main__":
    main()
