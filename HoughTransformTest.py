import sys
import math
import cv2
import numpy as np
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import Sobel_Harris_Test as sh

def show_images(img):
    # Display the result
    cv2.imshow("",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_line(rho, theta):
    """
    Closest point of the found line formula:
    (x0,y0) = (ρ * cos(θ), ρ * sin(θ))

    Since cos = adj/hypt in respect of x axis, cos(theta) * rho (hypt) = x value
    Same applies to sin, sin(theta) * rho = y value
    """
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # To expand the line, totalling to 2000 pixel length
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return [(x1, y1), (x2, y2)]


def polar_theta(point):
    """
    Get the polar coordinate theta value from two lines
    point is a [[x1, y1, x2, y2]] type argument
    """
    x1, y1, x2, y2 = point[0]
    return np.arctan2(y2 - y1, x2 - x1) % np.pi


def polar_rho(point, theta):
    x1, y1, x2, y2 = point[0]
    return x1 * np.cos(theta) + y1 * np.sin(theta)


def show_lines(lines, image):
    image_copy = image.copy()
    if len(lines) != 0:
        if len(lines[0][0]) == 2:
            for line in lines:
                rho,theta = line[0]
                points = get_line(rho,theta)
                cv2.line(image_copy, points[0], points[1], (0, 0, 255), 1)
        else:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                cv2.line(image_copy, (x0,y0), (x1,y1), (0, 0, 255), 1)
    show_images(image_copy)


def main():
    """
    Load and grayscale image
    Gaussian blur
    Hough line transformation
    Harris corner detection
    Filter lines by harris corners and similarity
    Run grid detection on filtered lines
    """
    image_name = "GoChess-Fully-Robotic-Chess-Board-1-1024x683.jpg"
    # 0 = strict, 1 = default
    detection_mode = 0
    image = cv2.imread(image_name)
    # Get image dimensions
    width, height, channels = image.shape
    image_center = (width / 2, height / 2)

    # Scaled eps by resolution of image
    eps = max(width, height) / 200
    """
    index 0 params is when the chessboard nearly fills the whole photo
    index 1 params are default params when chessboard is around 1/5-1/4 the image
    index 2 might be too noisy (currently not considered)
    """
    detection_params = [[11, 150, 200, 150],
                        [7, 100, 150, 100],
                        [5, 80, 100, 80]]
    settings = detection_params[detection_mode]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    """
    Gaussian Blur
    
    ksize: is responsible for how blurred the image will be (must be odd)
    sigmaX: Controls how much smoothing is applied (Standard Deviation in the X)
    """
    ksize = settings[0]
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 1)

    """
    Canny edge detection
    
    parameters:
    image
    threshold1: The minimum edge threshold, any weak edge below this is discarded
    threshold2: The strong threshold, anything above this threshold is automatically accepted
    """
    edges = cv2.Canny(blurred, settings[1], settings[2])
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)  # Wait for any key press
    # cv2.destroyAllWindows()

    """
    We should expect the chess board to be at least 15% of the image,
    This calculates 15% of dimensions of the image as the minimum length of lines
    """

    """
    Hough Line Transform
    
    parameters:
    image = image instance from opencv
    rho = Tells amount of steps in pixels to check
    theta = The total rotation to check for line interactions
    threshold = minimum number of votes (intersections in the accumulator) 
    
    Returns a packed list of [[[rho,theta]],...] representing lines
    """
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=settings[3])
    lines = fg.filter_similar_lines(lines)
    corners = sh.Harris(image)

    lines = sh.filter_hough_lines_by_corners(lines, corners, tolerance=eps)
    clusters = fg.cluster_lines(lines)
    cluster_list = list(clusters.items())

    for i, theta in enumerate(cluster_list[1:], start=1):
        _, cluster1 = cluster_list[i - 1]
        _, cluster2 = cluster_list[i]
        fg.check_grid_like(cluster1, cluster2)

    for x, y in corners:
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    show_lines(lines,image)

    # groups = fg.group_lines(lines, True, width, height)
    # for grouped_lines in groups:
    #     show_lines(grouped_lines, image)

    """
    Probabilistic Hough Line Transform

    parameters:
    image
    rho = Tells amount of steps in pixels to check
    theta = The total rotation to check for line interactions
    threshold = minimum number of votes (intersections in the accumulator) 
    
    returns packed (3D array of [[[x0,y0,x1,y1]]])

    These must be determined by the dimensions of the photo
    minLineLength
    maxLineGap
    """
    # minLen = int(np.sqrt(width * height) * 0.15)
    # minCombine = int(0.02 * np.sqrt((np.square(width) + np.square(height)))) # image diagonal length with form factor
    #
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold = 20, minLineLength=minLen, maxLineGap=minCombine)
    # show_lines(lines, image)
    # groups = fg.group_lines(lines, is_polar=False, width=width, height=height)
    # for grouped_lines in groups:
    #     show_lines(grouped_lines, image)


if __name__ == "__main__":
    main()