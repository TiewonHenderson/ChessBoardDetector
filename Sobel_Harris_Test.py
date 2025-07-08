import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree


def plot_points(points):
    """
    Plots a set of 2D points.

    Args:
        points: List or array of [x, y] points, shape (N, 2).
        title: Optional title for the plot.
        show: Whether to call plt.show() immediately.
    """
    points = np.array(points)  # Convert to numpy array if not already

    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=30, marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


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


def remove_noise(corners, eps_factor=1.0):
    """
    :param corners: 2D python list representing cartesian corner points
    :param eps_factor: scaling multiplier (larger == more lenient) for eps, which would represent typical distance
    between corner points to be merged as a group
    :return:
    """
    if corners is None or len(corners) < 2:
        return corners

    tree = KDTree(corners)
    dists, _ = tree.query(corners, k=min(len(corners), 5))  # Use more neighbors for better estimation
    gap_est = np.percentile(dists[:, 1], 50)  # Median of nearest neighbor distances
    eps = max(eps_factor * gap_est, 1.0)
    dbs_input = np.array(corners)
    db = DBSCAN(eps=eps, min_samples=2, algorithm='ball_tree').fit(dbs_input)
    labels = db.labels_
    core_pts = list(dbs_input[labels != -1])

    return core_pts


# GPT generated
def harris(image, ksize, filter_eps=None):
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
    if filter_eps is not None:
        return remove_noise(cluster_duplicates(corner_points, eps), eps_factor=filter_eps)
    return cluster_duplicates(corner_points, eps)


def hough_line_intersect(line, point, tolerance=5):
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


def consistent_gaps(points):
    """
    :param points: list of [x,y], length must be >= 4
    :return:
    """
    gaps = []
    for i in range(len(points) - 1):
        a, b = np.array(points[i]), np.array(points[i + 1])
        dist = float(np.linalg.norm(a - b))
        if dist > 0.0001:
            gaps.append(dist)
    # There aren't 4 points, or some points are too close to each other
    if len(gaps) < 3:
        return False
    min_gap = min(gaps)
    eps = 0.1 * min_gap
    for g in gaps:
        ratio = g / min_gap
        m = round(ratio)
        if abs(ratio - m) > eps:
            return False
    return True


def filter_hough_lines_by_corners(lines, corners, tolerance=5, min_hits=4):
    """

    :param lines:
    :param corners:
    :param tolerance: Represented by total pixels off to be considered intersecting a corner
    :param min_hits: The amount of intersections a line needs to be added
    :return:
    """
    filtered_lines = [[], []]
    for l in lines:
        hits = []
        for point in corners:
            if hough_line_intersect(l, point, tolerance):
                hits.append(point)
        if len(hits) == 0:
            continue
        # Attempts to get lines with consistent gaps as priority
        # Rest of intersected points are appended to 2nd list
        if consistent_gaps(hits):
            filtered_lines[0].append(l)
        if len(hits) >= min_hits:
            filtered_lines[1].append(l)
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