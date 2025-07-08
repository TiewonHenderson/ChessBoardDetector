import sys
import math
import cv2
import numpy as np
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import Sobel_Harris_Test as sh
from ChessBoardDetector import HoughTransformTest as ht

"""
settings 
0 params is when the chessboard nearly fills the whole photo
1 params are default params when chessboard is around 1/5-1/4 the image
2 params for chessboards 1/4 of image

setting params are formatted as such:
0 = ksize (blur intensity, must be odd)
1 = edge_threshold1 (Min threshold for edge to be a candidate)
2 = edge_threshold2 (threshold for automatic edge detection)
3 = houghline_threshold (votes to accept a line)
4 = harris_eps (scaling multiplier (larger == more lenient) for harris corner to merge as a cluster)
purpose: remove noise corner points
5 = eps_scalar (scaling multiplier (larger == more lenient) for clustering and filtering lines by:
intersection with corners (how off they can be)
clustering lines by consistent gaps (how off the gaps can be to be merged))
6 = similar_houghline_eps (A constant value epsilon as a threshold for difference in
rho, theta to still be accepted as similar hough lines)
"""
detection_params = [[11, 150, 200, 150, None, 1, 1.75],
                    [7, 100, 150, 100, 0.5, 0.75, 1],
                    [5, 85, 100, 80, 1.0, 0.15, 0.5]]


def image_load(image_name):
    """
    :param image_name: The path/name of the image within this dir
    :return: A cv2 image instance
    """
    image = cv2.imread(image_name)
    # Get image dimensions
    height, width, _ = image.shape
    if height * width > 2073600:
        max_width = 1920
        max_height = 1080

        # Calculate scaling factor, to maintain aspect ratio
        scale_factor = min(max_width / width, max_height / height)

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        image = cv2.resize(image, (new_width, new_height))
        height = new_height
        width = new_width
    return image


def get_edges(image, settings):
    """
    Gets edges by:
    Gaussian Blur
    Canny edge detection

    :param image: An opencv2 instance of image opened
    :param setting: A 1D list from detection_params to set threshold and epsilon values
    :return: A 2D python list of binary edges
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ksize = settings[0]
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 1)
    return cv2.Canny(blurred, settings[1], settings[2])


def houghLine_detect(edges, corners=None, mask=None, threshold=150, corner_eps=5, line_eps=10):
    """

    :param edges: All the edges found from canny edge detection
    :param corners: A result from harris corner detection
    :param mask: A mask over the image to actually look for lines
    :param threshold: threshold in terms of votes for Hough line detection (used from settings list)
    :param corner_eps: Epsilon value when filtering lines by corner intersection (how many pixel off is still accepted)
    :param line_eps: Epsilon value when filtering out SIMILAR lines
    :return:
    """
    e = edges.copy()
    if mask is not None:
        # Remove edges where mask is set as 0
        e[mask == 0] = 0
    lines = ht.unpack_hough(cv2.HoughLines(e, 1, np.pi / 180, threshold=threshold))
    lines = fg.filter_similar_lines(lines, line_eps)
    if corners is not None:
        lines = sh.filter_hough_lines_by_corners(lines, corners, tolerance=corner_eps)
    return lines


def cluster_lines(image, lines, gap_eps):
    """

    :param image:
    :param lines:
    :param gap_eps:
    :return:
    """
    clusters = fg.cluster_lines(lines, gap_eps)
    for key in clusters:
        """
        Sorts each cluster by rho value, then linearly scans to group each line by consistent gaps

        gap_Interpolation returns potentially combined intervals if gaps are consistent or multiples of
        each other (possibly indicating missing grid lines)

        saves the largest found group of parallel lines with consistent gaps
        """
        sorted_list, gap_average = ht.sort_Rho(clusters[key], gap_eps)
        clusters[key] = max(ht.gap_Interpolation(sorted_list, gap_average, image.shape[:2]), key=len)
    return list(clusters.items())


def check_all_grids(image, cluster_list, corners):
    """

    :param image:
    :param cluster_list:
    :param corners:
    :return:
    """
    clusters = []
    scores = []
    """
    Brute force check each cluster conditions stated in filter_grids.check_grid_like() header
    each cluster is then combined and a score is saved corresponding to their indices

    Currenly, the highest score is considered the chessboard 
    """
    for i, theta in enumerate(cluster_list[1:], start=1):
        _, cluster1 = cluster_list[i - 1]
        clusters.append(cluster1)
        j = i
        while j < len(cluster_list):
            _, cluster2 = cluster_list[j]
            score = fg.check_grid_like(cluster1, cluster2, image.shape, corners)
            clusters[-1].extend(cluster2)
            scores.append(score)
            j += 1
    return clusters, scores


def present_lines(image, clusters, scores, corners):
    """

    :param image:
    :param clusters:
    :param scores:
    :param corners:
    :return:
    """
    if len(scores) > 0:
        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        for x, y in corners:
            cv2.circle(image, (x, y), 3, color=blue, thickness=-1)

        max_index = scores.index(max(scores))
        for i in range(len(clusters)):
            if i == max_index:
                ht.put_lines(clusters[i], image, green)
            else:
                ht.put_lines(clusters[i], image, red)
        ht.show_images(image)


def detect_chessboard(image_name, detection_mode):
    """

    :param image_name:
    :param detection_mode:
    :return:
    """
    image = image_load(image_name)
    height, width, _ = image.shape

    # Epsilon list (suggested by GPT and tested)
    settings = detection_params[detection_mode]
    image_diagonal = np.hypot(width, height)
    line_similarity_eps = int(image_diagonal * 0.01) * settings[6]
    corner_eps = int(image_diagonal * 0.05) * settings[5]
    gap_eps = 0.2 * settings[5]

    edges = get_edges(image, settings)
    corners = sh.harris(image, settings[0], settings[4])

    # Filter outer pixels by using a binary mask mask,
    # lines from those areas are generally not part of the chessboard
    # This will disgard 10% of the width and 6% of the height outer borders
    w_ratio = 0.1
    h_ratio = 0.06
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(
        mask,
        # top-left corner
        (int(width * w_ratio), int(height * h_ratio)),
        # bottom-right corner
        (int(width * (1 - w_ratio)), int(height * (1 - h_ratio))),
        255,
        thickness=-1
    )
    good_lines, lines = houghLine_detect(edges, corners, mask, settings[3], corner_eps, line_similarity_eps)
    clusters = cluster_lines(image, good_lines, gap_eps)
    clusters, scores = check_all_grids(image, clusters, corners)

    if len(scores) > 0 and max(scores) > 40:
        present_lines(image, clusters, scores, corners)
        return True
    else:
        clusters = cluster_lines(image, lines, gap_eps)
        clusters, scores = check_all_grids(image, clusters, corners)
    if len(scores) > 0 and max(scores) > 20:
        present_lines(image, clusters, scores, corners)
        return True

    print(scores)
    # NO FOUND GRID, TRY DIFFERENT SETTINGS
    return False


def main():
    easy = ["Taken_Photos/left,25angle.png",
            "Taken_Photos/left,random,25angle.png",
            "Taken_Photos/top,random,rotated.png",
            "Taken_Photos/top,random.png",
            "Taken_Photos/top,rotated.png",
            "Taken_Photos/top.png",
            "Real_Photos/Close,somewhat_angled,solid.jpg",
            "Real_Photos/Close,somewhat_angled,solid2.jpg"]
    medium = ["Taken_Photos/left,65angle.png",
              "Taken_Photos/left,random,45angle.png",
              "Taken_Photos/left,rotated,45angle.png",
              "Taken_Photos/left,rotated,random,45angle.png",
              "Real_Photos/Mid,not_angled,flat.jpeg",
              "Real_Photos/Mid,not_angled,flat2.jpg",
              "Real_Photos/Mid,somewhat_angled,flat.png"]
    hard = ["Taken_Photos/left,rotated,65angle.png",
            "Taken_Photos/left,rotated,random,65angle.png",
            "Real_Photos/Far,angled,solid.jpg",
            "Real_Photos/1-5.png",
            "Real_Photos/Far,somewhat_angled,flat.jpg",
            "Real_Photos/Mid,somewhat_angled,flat2.jpg",
            "Real_Photos/Mid,very_angled,solid.jpg"]
    detected = False
    i = 0
    while not detected:
        detected = detect_chessboard(hard[0], i)
        i += 1


if __name__ == "__main__":
    main()