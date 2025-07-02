import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression


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
    dbscan = DBSCAN(eps=eps, min_samples=1)
    labels = list(dbscan.fit_predict(corner_points))

    unique_corners = []
    for label in labels:
        cluster_points = corner_points[labels == label]
        unique_corners.append(cluster_points.mean(axis=0).astype(int).tolist())
    return unique_corners


"""
Inital function to return cartesian points representing corners 
"""
def Harris(img):
    # Load image in grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # OpenCV’s cornerHarris internally computes gradients and matrix M,
    dst = cv2.cornerHarris(blurred, blockSize=2, ksize=3, k=0.04)

    # Dilate to find local maxima
    dst_dilated = cv2.dilate(dst, None)

    # Create mask of local maxima
    local_max_mask = (dst == dst_dilated)

    # Threshold the Harris response
    threshold = 0.01 * dst.max()
    threshold_mask = (dst > threshold)

    # Combine masks: points that are local maxima AND above threshold
    corner_mask = np.logical_and(local_max_mask, threshold_mask)

    # Get coordinates of final corners
    corner_points = np.argwhere(corner_mask)
    corner_points = corner_points[:, [1, 0]]  # convert to (x, y)
    width, height, channels = img.shape
    eps = max(width, height) / 100
    return cluster_duplicates(corner_points, eps)


def filter_hough_lines_by_corners(lines, corners, tolerance=5, min_hits=4):
    filtered_lines = []
    for line in lines:
        rho, theta = line[0]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        hits = 0
        for x0, y0 in corners:
            distance = abs(x0 * cos_t + y0 * sin_t - rho)
            if distance < tolerance:
                hits += 1
        if hits >= min_hits:
            filtered_lines.append([[rho, theta]])
    return filtered_lines


def main():
    """
        corner_points = your list/array of [x, y]

        run RANSAC line fitting directly on corner_points
        then later:
        compare RANSAC line angles to PCA chessboard angle

        No need to rotate or align points first
    """
    image_name = "Chess-Bud-Talk-at-Cavan-Chess-Congres-2023.jpeg"
    img = cv2.imread(image_name)
    width, height, channels = img.shape
    corners = Harris(img, width, height)

    for x, y in corners:
        cv2.circle(img, (x, y), 3, (0, 0, 255), 1)

    cv2.imshow("",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()