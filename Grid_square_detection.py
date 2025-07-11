import sys
import numpy as np


def insert_dummy_lines(cluster):
    """

    :param cluster: Expected 2 lists of houghline [rho, theta] values, with list sorted by rhos
    :return:
    """
    if len(cluster) != 2 or len(cluster[0]) == 0 or len(cluster[1]) == 0:
        return None

    dummies = [[], []]
    for i in range(len(cluster)):
        # How many lines needs to be artifically inserted
        needed_lines = 8 - len(cluster[i])
        temp_group = []
        left = -needed_lines
        right = -1
        min_line = cluster[i][0]
        max_line = cluster[i][-1]
        # gap_mean is used to insert dummy lines with multipled gaps to form a uniform group of lines
        # Means of theta is used to generate lines with this theta
        gap_mean = 0
        theta_mean = 0
        for j in range(len(cluster[i])):
            rho, theta = cluster[i][j]
            theta_mean += theta
            if j == len(cluster[i]) - 1:
                break
            gap_mean += abs(cluster[i][j + 1][0] - rho)
        theta_mean /= len(cluster[i])
        gap_mean /= len(cluster[i]) - 1
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
        print(temp_group)
        # Adds dummy lines into each group
        for interval in temp_group:
            left, right = interval
            dummies[i].append([])
            # The reference rho to offset off of,
            # its min_line's rho for negative multiples
            # max_line's rho for positive multiples
            ref_rho = min_line[0]
            for multiple in range(left, right + 1):
                if multiple == 0:
                    continue
                # If interval has 1 included, put copy before adding any dummy lines
                # This doesn't work for ending -1, thats extended later
                # if multiple == 1 or left == multiple == 1:
                #     dummies[i][-1].extend(cluster[i].copy())
                #     ref_rho = max_line[0]
                dummies[i][-1].append([ref_rho + (multiple * gap_mean), theta_mean])
            # Extend for [-n, -1] interval
            # dummies[i][0].extend(cluster[i].copy())

    for clusters in dummies:
        print("cluster")
        for groups in clusters:
            print(len(groups))
            print(groups)

    return dummies


def closest_line(line_list, line, threshold):
    """
    :param line_list:
    :param line:
    :return:
    """
    if len(line_list) == 0 or line is None:
        return None, None

    min_distance = sys.maxsize
    difference = ()
    min_index = -1
    # Find the ranges for normalization
    rhos = [rho for rho, _ in line_list]
    thetas = [theta for _, theta in line_list]

    rho_range = max(rhos) - min(rhos) if max(rhos) != min(rhos) else 1
    theta_range = max(thetas) - min(thetas) if max(thetas) != min(thetas) else 1

    for i, rho, theta in enumerate(line_list):
        diff_rho = abs(rho - line[0]) / rho_range
        diff_theta = abs(theta - line[1]) / theta_range

        distance = diff_rho + diff_theta

        if distance < min_distance:
            min_distance = distance
            difference = (diff_rho, diff_theta)
            min_index = i

    if min_distance <= threshold:
        return min_index, difference
    return None, difference


def check_dummy_lines(dummies, lines, corners):
    """
    It's almost guaranteed the dummy lines would form a grid, instead they'll be graded on:
    1) If there is an near enough line similar to the dummy lines (With more lines missing, epsilon is lowered)
    2) If they intersect the harris corner points that were detected

    :param dummies: A 3D list of 2 clusters filled with 2d arrays (groups) of rho, theta lines
    :param lines: A 2D list of rho, theta representing all lines found from houghline
    :param corners: A 2D list of points representing cartesian points
    :return:
    """
    scores = [[] for i in range(len(dummies))]
    for cluster_i in range(len(dummies)):
        for groups in cluster_i:
            for l in groups:
                # Threshold for line similarity
                # The more lenient the threshold, the less score given
                # If no lines were found, it will scale heavily
                for i in np.arange(0.1, 0.6, 0.1):
                    min_index, difference = closest_line(lines, l, i)
                    if min_index is None:
                        continue
                    else:







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
