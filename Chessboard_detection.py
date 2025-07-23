import sys
import math
import cv2
import numpy as np
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import CV_filter_groups as cvfg
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import HoughTransform as ht
from ChessBoardDetector import Grid_Completion as gc
from ChessBoardDetector import Vanishing_point as vp

"""
constant = the value is used as is
multiplier = a multipled value to an already constant epsilon
(larger == more lenient)
setting params are formatted as such:
0) ksize             [constant]   (blur intensity, must be odd)
1) edge_threshold1   [constant]   (Min threshold for edge to be a candidate)
2) edge_threshold2   [constant]   (threshold for automatic edge detection)
3) hline_thres       [constant]   (votes to accept a line)
4) similar_hline_eps [multiplier] (A epsilon treated as a threshold for difference in
rho, theta to still be accepted as similar hough lines)
5) eps_scalar        [multiplier] for filtering lines by:
intersection with corners (how off they can be)
clustering lines by consistent gaps (how off the gaps can be to be merged))
6) cluster_eps       [multiplier] for clustering lines together by theta
Either DBSCAN cluster (determines how close the theta values are to be merged)
Or KMEANs (this eps doesn't affect it)
7) gap_eps           [multiplier] for scaled epsilon between gaps, if the gap difference
is below the eps, it gets accepted within a line group
"""
ksize = 7
edge_thres1 = 100
edge_thres2 = 150
hline_thres = 100
similar_hline_eps = 1
eps_scalar = 0.55
cluster_eps = 0.5
gap_eps = 0.85


def image_load(image_name):
    """
    :param image_name: The path/name of the image within this dir
    :return: A downscaled cv2 image instance, and an original image (can be None if no downscale)
    """
    image = cv2.imread(image_name)
    old_image = None
    scale_factor = None
    height, width, _ = image.shape
    if height * width > 2073600:
        max_width = 1920
        max_height = 1080

        # Calculate scaling factor, to maintain aspect ratio
        scale_factor = min(max_width / width, max_height / height)

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        old_image = image.copy()
        image = cv2.resize(image, (new_width, new_height))
    return image, old_image, scale_factor if scale_factor is not None else 1


def rescale_all(scale_factor, lines=[], corners=[]):
    """

    :param scale_factor:
    :param lines:
    :param corners:
    :return: Returns rescaled lines, corners respectively determined by scale_factor
    """
    if scale_factor == 0 or scale_factor is None:
        return lines, corners
    inv_scale = 1 / scale_factor
    # rescale lines by adding to rho (theta remains)
    # Points are just multiplying both x and y by inverse of scale_factor
    rescaled_lines = [[rho * inv_scale, theta] for rho,theta in lines]
    rescaled_corners = [[x * inv_scale, y * inv_scale] for x,y in corners]
    return rescaled_lines, rescaled_corners


def houghLine_detect(image_shape, edges, corners, image, mask=None, threshold=4):
    """
    :param image_shape: [height, width] of the image
    :param edges: All the edges found from canny edge detection
    :param corners: A result from harris corner detection
    :param mask: A mask over the image to actually look for lines
    :param threshold: threshold in terms of votes for Hough line detection (used from settings list)
    :return:
    """
    e = edges.copy()
    if mask is not None:
        # Remove edges where mask is set as 0
        e[mask == 0] = 0
    lines = ht.unpack_hough(cv2.HoughLines(e, 1, np.pi / 720, threshold=threshold))

    find_exact_line(image, lines, 0, corners=corners, green=False)

    if corners is not None:
        lines = hcd.filter_hough_lines_by_corners(lines, corners)
    lines = fg.filter_similar_lines(lines, image_shape)
    return ht.normalize_rho(lines)


def cluster_lines(image, lines, gap_eps):
    """
    Clustering and filtering groups by their shared vanishing point

    :param image:
    :param lines:
    :param gap_eps:
    :return:
    """
    # Clusters by shared vanishing point
    temp = vp.has_vanishing_point(lines, image.shape[:2])

    # for group in temp:
    #     indices, dir = group
    #     got_lines = [lines[i] for i in indices]
    #     print(group)
    #     find_exact_line(image, got_lines, 0)

    final_clusters = []
    for group in temp:
        indices, dir = group
        got_lines = [lines[i] for i in indices]
        got_lines.sort(key= lambda x:x[0])
        got_lines = vp.enough_similar_theta(got_lines)
        clean_lines, gap_avg = cvfg.cv_clean_lines(got_lines, dir, image.shape[:2], image)

        if clean_lines is not None:
            find_exact_line(image, clean_lines, 0, green=False)

        final_clusters.append(clean_lines)
    return final_clusters
    #
    #     """
    #     Uses intersection by an external perpendicular line in order to gauge gap consistency
    #     """
    #
    #     sorted_list, gap_average = ht.sort_Rho(c, gap_eps)
    #     # for x in sorted_list:
    #     #     lz = x
    #     #     print(lz)
    #     #     for i in range(len(lz)):
    #     #         print(lz[i])
    #     #         find_exact_line(image, lz, i)
    #     """
    #     Should be sorted as
    #     [Theta clusters
    #         [As group of lines (with similar gap)
    #             [As lines
    #                 [As rho, theta]
    #             ]
    #         ]
    #     ]
    #     """
    #     candidates = ht.gap_Interpolation(sorted_list, gap_average, image.shape[:2])
    #     if len(candidates) != 0:
    #         final_clusters.append(candidates)
    # return final_clusters


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

    cluster_list = 4d array
    Each cluster stores similar theta groups that are stored in list of similar gaps by rho
    """
    for i in range(1, len(cluster_list)):
        c1 = cluster_list[i - 1]
        for g1 in c1:
            j = i
            while j < len(cluster_list):
                c2 = cluster_list[j]
                for g2 in c2:
                    score, intersect_list = fg.check_grid_like(g1, g2, image.shape, corners)
                    clusters.append([g1.copy()])
                    clusters[-1].append(g2.copy())
                    scores.append(score)
                    j += 1
    return clusters, scores


def present_lines(image, clusters, scores, corners):
    """
    :param image: The opencv instance of image to display the found grid lines
    :param clusters: The cluster list that contains combined groups that recieved grid like score
    :param scores:
    :param corners:
    :return:
    """
    if len(scores) > 0:
        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        for x, y in corners:
            cv2.circle(image, (int(x), int(y)), 3, color=blue, thickness=-1)

        max_index = scores.index(max(scores))
        for i in range(len(clusters)):
            g1, g2 = clusters[i]
            if i == max_index:
                ht.put_lines(g1, image, green, 2)
                ht.put_lines(g2, image, green, 2)
            else:
                ht.put_lines(g1, image, red, 1)
                ht.put_lines(g2, image, red, 1)
        ht.show_images(image)


def find_exact_line(image, lines, index, corners=[], green=True):
    """
    Function mainly to show each line value, not used for implementation

    :param image:
    :param lines:
    :param index:
    :param corners:
    :param green: Make the line at index green if true
    :return:
    """
    img = image.copy()
    blue = (255, 0, 0)
    if len(lines) == 0:
        for x, y in corners:
            cv2.circle(img, (int(x), int(y)), 3, color=blue, thickness=-1)
        ht.show_images(img)
        return
    l_copy = lines.copy()
    if green:
        line_x = [l_copy[index]]
        l_copy.pop(index)
        ht.put_lines(l_copy, img, (0, 0, 255))
        ht.put_lines(line_x, img, (0, 255, 0))
    else:
        ht.put_lines(l_copy, img, (0, 0, 255))
    for x, y in corners:
        cv2.circle(img, (int(x), int(y)), 3, color=blue, thickness=-1)
    ht.show_images(img)


def detect_chessboard(image_name, thres_config, d_mode):
    """

    :param image_name:
    :param thres_config:
    :return: An image with
    """

    ksize, edge_thres1, edge_thres2, hline_thres = thres_config
    image, origin_img, image_scale = image_load(image_name)
    if origin_img is None:
        origin_img = image
    height, width, _ = image.shape

    # Epsilon list (suggested by GPT and tested)
    image_diagonal = np.hypot(width, height)
    gap_eps = 0.1

    corners = hcd.harris(image, ksize)
    find_exact_line(image, [], None, corners, False)
    # GPT generate to make corner binary map
    binary_map = np.zeros((height, width), dtype=np.uint8)
    for x, y in corners:
        binary_map[int(y), int(x)] = 255  # Note: OpenCV uses (y, x) order for indexing

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
    lines = houghLine_detect([height, width],
                             binary_map,
                             corners,
                             image,
                             mask,
                             3)
    # Rounds up due to having extremely long decimals
    lines = [[round(rho, 3), round(theta, 5)] for rho, theta in lines]

    find_exact_line(image, lines, 0, corners=corners, green=False)
    print("Finding lines")
    for x, y in corners:
        cv2.circle(image, (x, y), radius=3, color=(0, 255, 255), thickness=1)

    # lines.sort(key=lambda x: x[1])
    # for x in range(len(lines)):
    #     print(lines[x])
    #     find_exact_line(image, lines, x)

    if len(lines) <= 4:
        return False
    clusters = cluster_lines(image, lines, gap_eps)

    # for l in range(len(clusters)):
    #     print(clusters[l])
    #     find_exact_line(image, clusters, l)

    # clusters, scores = check_all_grids(image, clusters, corners)
    # if len(scores) > 0 and max(scores) >= 30:
    #
    #     # 1 means scaled by 1 (image resolution is as is)
    #     if image_scale == 1:
    #         present_lines(image, clusters, scores, corners)
    #     else:
    #         new_clusters = []
    #         _, scaled_corners = rescale_all(image_scale, corners=corners)
    #         for i, group_pairs in enumerate(clusters):
    #             new_clusters.append([])
    #             for j, group in enumerate(group_pairs):
    #                 scaled_group, _ = rescale_all(image_scale, group)
    #                 new_clusters[i].append(scaled_group)
    #         present_lines(origin_img, new_clusters, scores, scaled_corners)
    #     return True
    #
    # return False


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

    test1 = ["Taken_Photos/left,25angle.png",
            "Taken_Photos/left,random,25angle.png",
            "Taken_Photos/top,random,rotated.png",
            "Taken_Photos/top,random.png",
            "Taken_Photos/top,rotated.png",
            "Taken_Photos/top.png",
            "Taken_Photos/left,65angle.png",
             "Taken_Photos/left,random,45angle.png",
             "Taken_Photos/left,rotated,45angle.png",
             "Taken_Photos/left,rotated,random,45angle.png",
             "Taken_Photos/left,rotated,65angle.png",
             "Taken_Photos/left,rotated,random,65angle.png"
            ]

    detected = False
    i = 1
    while not detected:
        thres_config = (ksize, edge_thres1, edge_thres2, hline_thres)
        # for j in range(len(easy)):
        #     detected = detect_chessboard(easy[j], thres_config, scalar_config, i)
        #     print('done: ', i)
        # for j in range(len(medium)):
        #     detected = detect_chessboard(medium[j], thres_config, scalar_config, i)
        #     print('done: ', i)
        for j in range(len(medium)):
            detect_chessboard(medium[3], thres_config, i)
        i += 1


if __name__ == "__main__":
    main()