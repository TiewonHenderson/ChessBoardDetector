import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.spatial import KDTree


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
    for label in labels:
        # GPT generated
        cluster_points = corner_points[labels == label]
        unique_corners.append(cluster_points.mean(axis=0).astype(int).tolist())
    return unique_corners


# GPT generated
def harris(image, ksize):
    """
    Inital function to return cartesian points representing corners
    :param image: Instance of image from cv2
    :param ksize: Blur intensity of the image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 1)

    # OpenCV’s cornerHarris internally computes gradients and matrix M
    dst = cv2.cornerHarris(blurred, blockSize=2, ksize=3, k=0.04)

    # Dilate to find local maxima
    dst_dilated = cv2.dilate(dst, None)

    # Create mask of local maxima
    local_max_mask = (dst == dst_dilated)

    # Threshold the Harris response (gives all local maxima where the response value > 1%)
    threshold = 0.01 * dst.max()
    threshold_mask = (dst > threshold)

    # Combine masks: points that are local maxima AND above threshold
    corner_mask = np.logical_and(local_max_mask, threshold_mask)

    # Get coordinates of final corners
    corner_points = np.argwhere(corner_mask)
    corner_points = corner_points[:, [1, 0]]  # convert to (x, y)
    height, width, _ = image.shape
    eps = max(width, height) / 100

    return cluster_duplicates(corner_points, eps)


def hough_line_intersect(line, point, tolerance=1):
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


def consistent_gaps(points, get_variance=False):
    """
    CV Scoring:
    consistent spacing  	< 0.07
    somewhat structured	        0.10–0.15
    Noisy / outlier	        > 0.20

    Uses 0.15 during to 3d aspect we cannot control with images
    :param points: list of [x,y], length must be >= 4
    :param get_variance: Gets the variance of the points given instead of a boolean
    :return:
    """
    gaps = []
    eps = 0.4

    points_copy = points.copy()
    points_copy.sort(key=lambda x: x[0])
    for i in range(len(points_copy) - 1):
        a, b = np.array(points_copy[i]), np.array(points_copy[i + 1])
        dist = float(np.linalg.norm(a - b))
        if dist > 0.0001:
            gaps.append(dist)
    # get variance instead of consistency evaluation
    mean = np.mean(gaps)
    cv = sys.maxsize
    if mean > 0.0001:
        cv = np.std(gaps) / mean
    if get_variance:
        return cv
    # There arent enough points, or some points are too close to each other
    if len(gaps) < 3:
        return False
    return cv < eps


def filter_hough_lines_by_corners(lines, corners, min_hits=3):
    """

    :param lines:
    :param corners:
    :param tolerance: Represented by total pixels off to be considered intersecting a corner
    :param min_hits: The amount of intersections a line needs to be added
    :return:
    """
    filtered_lines = []
    for l in lines:
        hits = []
        for point in corners:
            if hough_line_intersect(l, point):
                hits.append(point)
        if len(hits) == 0:
            continue
        # Attempts to get lines with consistent gaps as priority
        # Rest of intersected points are appended to 2nd list
        if len(hits) >= min_hits:
            filtered_lines.append(l)
    return filtered_lines


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