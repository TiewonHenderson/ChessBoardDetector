import sys
import math
import cv2
import numpy as np
from collections import Counter
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import cv_filter_groups as cvfg
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
4) dir_eps
"""
ksize = 3
edge_thres1 = 100
edge_thres2 = 150
hline_thres = 100

dir_eps = {0: 0.3,
           45: 0.175,
           90: 0.1,
           135: 0.175,
           180: 0.3,
           225: 0.175,
           270: 0.1,
           315: 0.175}


def image_load(image_name):
    """
    makes aspect ratio 1:1 and downscaled if needed
    :param image_name: The path/name of the image within this dir
    :return: A downscaled cv2 image instance, and an original image (can be None if no downscale)
    """
    image = cv2.imread(image_name)
    old_image = None
    # For original points when scaling back
    scale_factor_x = scale_factor_y = 1
    height, width, _ = image.shape
    max_dim = 1000
    if height > max_dim or width > max_dim:
        # Calculate scaling factor, to maintain aspect ratio
        scale_factor = min(max_dim / width, max_dim / height)

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        old_image = image.copy()
        image = cv2.resize(image, (new_width, new_height))

        scale_factor_x = new_width / old_image.shape[1]
        scale_factor_y = new_height / old_image.shape[0]
    # Need 1:1 aspect ratio
    elif height/width != 1:
        size = max(height, width)
        old_image = image.copy()
        image = cv2.resize(image, (size, size))

        scale_factor_x *= size / width
        scale_factor_y *= size / height

    return image, old_image, (scale_factor_x, scale_factor_y)


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


def houghLine_detect(image_shape, edges, corners, mask=None, threshold=4):
    """
    :param image_shape: [height, width] of the image
    :param edges: All the edges found from canny edge detection
    :param corners: A result from harris corner detection
    :param mask: A mask over the image to actually look for lines
    :param threshold: threshold in terms of votes for Hough line detection (used from settings list)
    :return:
    """
    e = edges.copy()
    height, width = image_shape
    min_gap = 0.0264 * height
    if mask is not None:
        # Remove edges where mask is set as 0
        e[mask == 0] = 0
    lines = ht.unpack_hough(cv2.HoughLines(e, 1, 0.0175, threshold=threshold))
    lines = fg.filter_similar_lines(lines, image_shape)
    if corners is not None:
        lines, pt_by_line = hcd.filter_hough_lines_by_corners(lines, corners, min_gap)
        return lines, pt_by_line
    return lines, None


def cluster_lines(image, lines, corners, gap_eps, corner=[]):
    """
    Clustering and filtering groups by their shared vanishing point

    :param image:
    :param lines:
    :param gap_eps:
    :return:
    """

    temp = vp.has_vanishing_point(lines, image.shape[:2])
    # Added a clustering find as well with vp implementation
    clusters = fg.dbscan_cluster_lines(lines, indices=True, eps=0.1)
    for key in clusters:
        if len(clusters[key]) >= 4:
            clustered_lines = [lines[i] for i in clusters[key]]
            theta_to_dir = [fg.snap_to_cardinal_diagonal(np.rad2deg(l[1])) for l in clustered_lines]
            dir_counter = Counter(theta_to_dir)
            most_common_dir, _ = dir_counter.most_common(1)[0]
            temp.append((clusters[key], most_common_dir))

    final_clusters = []
    for group in temp:
        indices, dir = group
        got_lines = [lines[i] for i in indices]

        # print(dir)
        # print("before")
        # find_exact_line(image, got_lines, 0, green=False)

        clean_lines, new_dir = cvfg.cv_clean_lines(got_lines, corners, dir, image.shape[:2], image)

        if clean_lines is None or len(clean_lines) == 0:
            continue

        # print("after")
        # find_exact_line(image, clean_lines, -1, green=True)

        final_clusters.append((clean_lines, new_dir))
    return final_clusters


def check_all_grids(image, cluster_list, corners):
    """
    :param image:
    :param cluster_list:
    :param corners:
    :return:
    """
    clusters = []
    scores = []
    sect_list = []
    cardinal_vertical = {0, 180}
    cardinal_horizontal = {90, 270}
    """
    Brute force check each cluster conditions stated in filter_grids.check_grid_like() header
    each cluster is then combined and a score is saved corresponding to their indices

    cluster_list = 3d array
    Each cluster stores similar theta groups that are stored in list of similar gaps by rho
    """
    for i in range(1, len(cluster_list)):
        c1, dir1 = cluster_list[i - 1]
        j = i
        while j < len(cluster_list):
            c2, dir2 = cluster_list[j]
            # SHOULD HAVE CHECK BY PERPENDICULAR GROUPS, future improvement
            score, intersect_list = fg.check_grid_like(c1, c2, image.shape[:2], corners)
            clusters.append([cluster_list[i - 1]])
            clusters[-1].append(cluster_list[j])
            scores.append(score)
            sect_list.append(intersect_list)
            j += 1
    return clusters, scores, sect_list


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



def detect_chessboard(image_name, thres_config):
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
    min_gap = 0.0264 * height

    # Epsilon list (suggested by GPT and tested)
    image_diagonal = np.hypot(width, height)
    gap_eps = 0.1

    # Filter outer pixels by using a binary mask mask,
    # lines from those areas are generally not part of the chessboard
    # This will disgard 10% on both sides
    ratio = 0.10
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(
        mask,
        # top-left corner
        (int(width * ratio), int(height * ratio)),
        # bottom-right corner
        (int(width * (1 - ratio)), int(height * (1 - ratio))),
        1,
        thickness=-1
    )

    corners = hcd.harris(image, ksize, mask)
    # corners = hcd.shi_tomasi(image, ksize, min_gap)
    binary_map = np.zeros((height, width), dtype=np.uint8)

    for x, y in corners:
        x = int(x)
        y = int(y)
        binary_map[y, x] = 255

    lines, line_by_pts = houghLine_detect([height, width],
                                                    binary_map,
                                                    corners,
                                                    mask,
                                                    3)
    find_exact_line(image, lines, 0, corners=corners, green=False)
    # lines = sorted(lines, key=lambda x: x[1])
    # for i in range(len(lines)):
    #     print(i, lines[i])
    #     find_exact_line(image, lines, i, corners=corners, green=True)

    if len(lines) <= 4:
        print("Not enough lines for a grid")
        print("Failed houghline!!!")
        return None
    clusters = cluster_lines(image, lines, corners, gap_eps, corners)

    final_lines, scores, sect_list = check_all_grids(image, clusters, corners)

    if len(scores) > 0 and len(final_lines) > 0:
        print("best grid found")
        max_index = scores.index(max(scores))
        # Structured as g = (lines, params for curve fit, direction overall)
        # Should still be ordered by intersections
        g1_data, g2_data = final_lines[max_index]
        g1_lines, g1_dir = g1_data
        g2_lines, g2_dir = g2_data

        # print("score", scores[max_index])
        # find_exact_line(image, g1[0] + g2[0], 0, corners=corners, green=False)

        """
        sect_list = the intersection between the two groups represented as points
        line_pts = the intersection between lines and corner points
        corners = all corner points found by a corner detection function
        """
        gc.point_interpolate(g1_lines,
                             g2_lines,
                             sect_list[max_index],
                             line_by_pts,
                             corners,
                             image.shape[:2],
                             image=image,
                             lines=lines)
    else:
        print("No valid grid found")
        print("Failed check_all_grids")


def main():
    easy = ["Taken_Photos/left,25angle.png",
            "Taken_Photos/left,random,25angle.png",
            "Taken_Photos/top,random,rotated.png",
            "Taken_Photos/top,random.png",
            "Taken_Photos/top,rotated.png",
            "Taken_Photos/top.png",
            "Real_Photos/Close,somewhat_angled,solid.jpg",
            "Real_Photos/Close,somewhat_angled,solid2.jpg"
            ]
    medium = ["Taken_Photos/left,65angle.png",
              "Taken_Photos/left,random,45angle.png",
              "Taken_Photos/left,rotated,45angle.png",
              "Taken_Photos/left,rotated,random,45angle.png",
              "Real_Photos/Mid,not_angled,flat.jpeg",
              "Real_Photos/Mid,not_angled,flat2.jpg",
              "Real_Photos/Mid,somewhat_angled,flat.png"
              ]
    hard = ["Taken_Photos/left,rotated,65angle.png",
            "Taken_Photos/left,rotated,random,65angle.png",
            "Real_Photos/Far,angled,solid.jpg",
            "Real_Photos/1-5.png",
            "Real_Photos/Far,somewhat_angled,flat.jpg",
            "Real_Photos/Mid,somewhat_angled,flat2.jpg",
            "Real_Photos/Mid,very_angled,solid.jpg"
            ]

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

    thres_config = (ksize, edge_thres1, edge_thres2, hline_thres)
    # for j in range(len(easy)):
    #     detected = detect_chessboard(easy[j], thres_config, scalar_config, i)
    #     print('done: ', i)
    # for j in range(len(medium)):
    #     detected = detect_chessboard(medium[j], thres_config, scalar_config, i)
    #     print('done: ', i)
    for j in range(len(test1)):
        detect_chessboard(test1[j], thres_config)


if __name__ == "__main__":
    main()