import sys
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from ChessBoardDetector import cv_filter_groups as cvfg


def show_corners(image, points):
    # AI generated to only see points
    for x, y in points:
        cv2.circle(image, (x, y), radius=3, color=(0, 255, 255), thickness=1)

    cv2.imshow("Harris Corners with Circles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cluster_duplicates(corner_points, eps):
    """
        Uses DBSCAN clustering methods which doesn't need explicit cluster amount

        eps = max distance between duplicates
        min_samples = minimum number of points to be a cluster
    """
    # GPT used to get DBSCAN syntax
    dbscan = DBSCAN(eps=eps, min_samples=1)
    labels = list(dbscan.fit_predict(corner_points))
    unique_corners = []
    for label in set(labels):
        # Get all points belonging to this cluster
        cluster_mask = np.array(labels) == label
        cluster_points = corner_points[cluster_mask]
        # Calculate mean and add to unique corners
        unique_corners.append(cluster_points.mean(axis=0).astype(int).tolist())
    return unique_corners


def shi_tomasi(image, ksize, min_gap, points=200):
    """
    shi_tomasi method of corner detection to return cartesian points
    :param image: Instance of image from cv2
    :param ksize: Blur intensity of the image
    :param min_gap: Minimum distance each corner has to be away from
    :param points:
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 1)
    corners = cv2.goodFeaturesToTrack(
                blurred,
                maxCorners=200,
                qualityLevel=0.1,
                minDistance=10,
                blockSize=5
            )
    corners = np.intp(corners)
    xy_points = [list(pt[0]) for pt in corners]
    return xy_points

# GPT generated
def harris(image, ksize, mask):
    """
    harris corner detection
    Inital function to return cartesian points representing corners
    :param image: Instance of image from cv2
    :param ksize: Blur intensity of the image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 1)

    """
    blockSize = How big the kernel is to compare in general
    ksize = How big the sobel gradient kernel is for (how the intensity is changing)
    k = sensitivity constant for separating corners form normal edges
    """
    dst = cv2.cornerHarris(blurred, blockSize=3, ksize=ksize + 2, k=0.03)

    # Dilate to find local maxima
    dst_dilated = cv2.dilate(dst, None)

    # Create mask of local maxima
    local_max_mask = (dst == dst_dilated)

    # Threshold the Harris response (gives all local maxima where the response value > 5%)
    threshold = 0.05 * dst.max()
    threshold_mask = (dst > threshold)

    # Combine masks: points that are local maxima AND above threshold
    corner_mask = np.logical_and(local_max_mask, threshold_mask)
    corner_mask = np.logical_and(corner_mask, mask)

    # Get coordinates of final corners
    corner_points = np.argwhere(corner_mask)
    corner_points = corner_points[:, [1, 0]]  # convert to (x, y)
    height, width, _ = image.shape
    eps = height / 100

    return cluster_duplicates(corner_points, eps)


def hough_line_intersect(line, point, tolerance=2):
    """
    :param line: [rho, theta]
    :param point: [x, y]
    :param tolerance: Represented by total pixels off to be considered intersecting a corner
    :return: boolean
    """
    rho, theta = line
    x, y = point
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # The point to likne distance formula derived from x*cos(θ) + y*sin(θ) = ρ
    distance = abs(x * cos_t + y * sin_t - rho)
    if distance < tolerance:
        return True
    return False


def create_point_mask(pt1, pt2, x_d, y_d, include_origin_pts=False):
    """
    Helper function to create inserted points with the same steps given
    GPT rewriten to use numpy array instead of python list
    :param pt1:
    :param pt2:
    :param x_d: The x difference between the two points
    :param y_d: The y difference between the two points
    :param include_origin_pts: whether the mask points should include the inputted 2 points
    :return:
    """
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    step = np.array([x_d, y_d])

    # checks whether pt1 or pt2 should be first
    pt1_c = pt1 + step
    if np.array_equal(pt1_c, pt2):
        first, second = pt1, pt2
    else:
        first, second = pt2, pt1

    j = np.arange(1, 8)[:, None]

    # generating points using diff offset
    before_pts = first - j * step
    after_pts = second + j * step

    before_pts = before_pts[::-1]
    if include_origin_pts:
        before_pts = np.vstack([before_pts, first, second])
    mask_pts = np.vstack([before_pts, after_pts])

    return mask_pts.tolist()


def point_masking(points, min_gap, tolerance=5, needed_score=3):
    """
    Use each gap and create a 1x7 mask with distance
    :param points: collection of (x,y)/[x,y], length must be >= 4
    :param min_gap: Minimum gap expected between points
    :param tolerance:
    :param needed_score:
    :return:
    """
    # Gets the x, y difference to be used as a 1x7 point masking
    points_copy = np.array(list(points))                  #N
    sorted_points = points_copy[np.lexsort((points_copy[:, 1], points_copy[:, 0]))]
    step_diffs = np.diff(sorted_points, axis=0)     # N-1
    best_points = None
    for i, diff in enumerate(step_diffs):
        # Stage to get the interpolated points
        x_d, y_d = diff
        masking_pts = create_point_mask(points_copy[i], points_copy[i + 1], x_d,  y_d)

        # Stage to see if any points
        passing_pts = [points_copy[i], points_copy[i + 1]]
        for mask_pt in masking_pts:
            for index, point in enumerate(sorted_points):
                if index == i or index == i + 1:
                    continue
                if np.linalg.norm(mask_pt - point) < tolerance:
                    passing_pts.append(point)

        # Remove duplicates if any
        passing_pts = np.unique(passing_pts, axis=0)
        if len(passing_pts) >= needed_score:
            return passing_pts.tolist()
    return None


def filter_hough_lines_by_corners(lines, corners, min_gap, tolerance=2, min_hits=3):
    """
    :param lines:
    :param corners:
    :param min_gap
    :param tolerance: Represented by total pixels off to be considered intersecting a corner
    :param min_hits: The amount of intersections a line needs to be added
    :return:
    """
    filtered_lines = []
    pt_by_line = {}
    for l in lines:
        hits = set()
        for point in corners:
            if hough_line_intersect(l, point, tolerance):
                hits.add(tuple(point))
        # Rest of intersected points are appended to 2nd list
        if len(hits) >= min_hits:
            if point_masking(hits, min_gap) is not None:
                filtered_lines.append(l)
                pt_by_line[tuple(l)] = hits
    return filtered_lines, pt_by_line


def main():
    """
        corner_points = your list/array of [x, y]

        run RANSAC line fitting directly on corner_points
        then later:
        compare RANSAC line angles to PCA chessboard angle

        No need to rotate or align points first
    """
    image_name = "Real_Photos/Far,angled,solid.jpg"
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 1)
    width, height, channels = img.shape
    corners = harris(blurred, img.shape)

    for x, y in corners:
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow("",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()