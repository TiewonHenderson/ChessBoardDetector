from ChessBoardDetector import filter_grids as fg


def get_intersections(group1, group2):
    intersect_matrix = [[] for x in range(len(group1))]
    for i, l1 in enumerate(group1):
        for l2 in group2:
            intersect_matrix[i].append(fg.intersection_polar_lines(l1, l2))
    return intersect_matrix


def get_center(p1, p2, p3, p4):
    center_x = sum(p[0] for p in points) / 4
    center_y = sum(p[1] for p in points) / 4
    return (center_x, center_y)


def get_squares(cluster):
    """

    :param cluster: Requires 2 lists of lists containing houghline [rho, theta] values, with list sorted by rhos
                    Requires the 2 groups to form a perfect 8x8 intersection matrix
    :param corners: A 2D list of points representing cartesian points filtered to be intersecting with lines
                    within cluster
    :return:
    """
    if len(cluster) != 2:
        return None
    matrix = get_intersections(cluster[0], cluster[1])
    squares = [[] for x in range(len(matrix) - 1)]
    for i in range(len(matrix) - 1):
        for j in range(len(matrix[i]) - 1):
            #tl = top left,     tr = top right
            #bl = bottom left,  br = bottom right
            tl, tr = matrix[i][j], matrix[i][j + 1]
            bl, br = matrix[i + 1][j], matrix[i + 1][j + 1]
            squares[i].append(get_center(tl, tr, bl, br))
    return squares



