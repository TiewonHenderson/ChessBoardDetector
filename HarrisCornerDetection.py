import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
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


def shi_tomasi(image, ksize, min_gap):
    """
    shi_tomasi method of corner detection to return cartesian points
    :param image: Instance of image from cv2
    :param ksize: Blur intensity of the image
    :param min_gap: Minimum distance each corner has to be away from
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 1)
    corners = cv2.goodFeaturesToTrack(blurred, maxCorners=200, qualityLevel=0.01, minDistance=min_gap)
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
    Uses MAD to find median gap, and if majority of the points
    :param points: collection of (x,y)/[x,y], length must be >= 4
    :param get_variance: Gets the variance of the points given instead of a boolean
    :return:
    """
    gaps = []
    points_copy = np.array([[x, y] for x, y in points])
    """
        Gets distance to the nearest point instead
        Slots n neighbors closest to each point, needs to be 2 or it'll use itself
        Distance is the dist to the neighboring points
        Indices represent the point index in the list
    """
    neighbors = NearestNeighbors(n_neighbors=2).fit(points_copy)
    distances, indices = neighbors.kneighbors(points_copy)
    # Take the distance to the nearest *other* point (exclude self at index 0)
    nearest_dists = distances[:, 1]

    # Finds mad and makes sure every distance is within it
    outliers = cvfg.check_MAD(nearest_dists, 1.5)
    if abs(len(points) - len(outliers)) < 4:
        return False

    return True


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
        hits = set()
        for point in corners:
            if hough_line_intersect(l, point):
                hits.add(tuple(point))
        # Rest of intersected points are appended to 2nd list
        if len(hits) >= min_hits:
            if consistent_gaps(hits):
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