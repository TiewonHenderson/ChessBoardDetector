import numpy as np
from ChessBoardDetector import HoughTransform as ht

# Convert polar lines to image bounded points (2 points to represent 1 line)
def get_outer_points(g1, g2, bounds=(1000,1000)):
    """
    Gets the border points created by the intersection of the given lines and the border
    :param g1:
    :param g2:
    :param bounds: h,w values. But the image should be 1000x1000
    :return: Two groups (perserved indices) of lines represented
             as two points (x1,y1),(x2,y2) instead of rho,theta
    """
    def get_bounds(line, bounds=bounds):
        """
        Uses the idea that the border intersection must fall within the image bounds
        :param line:
        :param bounds:
        :return:
        """
        r, theta = line
        pt1, pt2 = None, None
        # Checks for the vertical intersections
        y0 = ht.y_for_line(0, r, theta)
        yW = ht.y_for_line(bounds[1] - 1, r, theta)

        if y0 is not None and 0 <= y0 < bounds[0]:
            y0 = int(y0)
            pt1 = (0, y0)

        if yW is not None and 0 <= yW < bounds[0]:
            yW = int(yW)
            if pt1 is not None:
                pt2 = (bounds[1] - 1, yW)
                return pt1, pt2
            pt1 = (bounds[1] - 1, yW)

        # Checks for the horizontal intersections
        x0 = ht.x_for_line(0, r, theta)
        xH = ht.x_for_line(bounds[0] - 1, r, theta)

        if x0 is not None and 0 <= x0 < bounds[1]:
            x0 = int(x0)
            if pt1 is not None:
                pt2 = (x0, 0)
                return pt1, pt2
            pt1 = (x0, 0)

        if xH is not None and 0 <= xH < bounds[1]:
            xH = int(xH)
            if pt1 is not None:
                pt2 = (xH, bounds[0] - 1)
            else:
                pt1 = (xH, bounds[0] - 1)
        return pt1, pt2

    g1_pts = []
    g2_pts = []
    for l in g1:
        pt1,pt2 = get_bounds(l)
        if pt1 is None or pt2 is None:
            continue
        g1_pts.append((pt1,pt2))
    for l in g2:
        pt1,pt2 = get_bounds(l)
        if pt1 is None or pt2 is None:
            continue
        g2_pts.append((pt1,pt2))
    return g1_pts, g2_pts

# Mask over the line, at most 8 different possibilities
def mask_lines(g1_pts, g2_pts):
    print(1)

# Intersection of each lines