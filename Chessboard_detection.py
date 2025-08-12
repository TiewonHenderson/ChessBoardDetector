import sys
import math
import cv2
import os
import numpy as np
from collections import Counter
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import cv_filter_groups as cvfg
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import HoughTransform as ht
from ChessBoardDetector import Grid_Completion as gc
from ChessBoardDetector import Vanishing_point as vp
from ChessBoardDetector import Line_Completion as lc

"""
constant = the value is used as is
multiplier = a multipled value to an already constant epsilon
(larger == more lenient)
setting params are formatted as such:
0) ksize             [constant]   (blur intensity, must be odd)
1) dir_eps           [constant]   epsilon for vanishing point within that direction, how off the theta can be
"""
ksize = 3
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
    needed_dim = 1000
    if height != needed_dim or width != needed_dim:
        old_image = image.copy()
        image = cv2.resize(image, (needed_dim, needed_dim))

        scale_factor_x = old_image.shape[1] / needed_dim
        scale_factor_y = old_image.shape[0] / needed_dim

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
    lines = ht.unpack_hough(cv2.HoughLines(e, 1, np.pi/360, threshold=threshold))
    lines = [(round(rho,3), round(theta,3)) for rho, theta in lines]
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

    def overlap_groups(candidate, final_clusters):
        """
        If groups having overlapping lines, combine into one group
        :param candidate: A 2d colleciton of lines and its dir
        :param found_lines: A set of tuples representing groups of lines
        :return:
        """
        lines, dir = candidate
        cand_tuple = list([tuple(x) for x in lines])
        cand_tuple_set = set(cand_tuple)
        cand_len = len(cand_tuple_set)
        for i, cluster in enumerate(final_clusters):
            group, _ = cluster
            group_tuple_set = set([tuple(x) for x in group])
            group_len = len(group)
            # Compare by set intersection
            intersect_sets = cand_tuple_set.intersection(group)
            if len(intersect_sets) == min(group_len, cand_len):
                if cand_len > group_len:
                    final_clusters.pop(i)
                    final_clusters.append((cand_tuple, dir))
                    return
                return
        final_clusters.append((cand_tuple, dir))
        return


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
        # print(clean_lines)
        # find_exact_line(image, clean_lines, -1, green=True)

        overlap_groups((clean_lines, new_dir), final_clusters)
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


def detect_chessboard(image_name, ksize=ksize):
    """
    :param image_name:
    :param ksize:
    :return: 4 corners or
    """

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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (ksize+2, ksize+2), 1)
    # edge by GPT
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)
    lines, line_by_pts = houghLine_detect([height, width],
                                                    edges,
                                                    corners,
                                                    threshold=max(height, width)//10)

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

        # find_exact_line(image, g1_lines + g2_lines, 0, corners=corners, green=False)
        """
        Build scoring system of points (only for line complete)
        """
        # verified, verified_lines = gc.intersect_verification(sect_list[max_index], corners)
        # flat_verified = [pt for row in verified for pt in row]
        # all_sects, line_corners = gc.get_points(sect_list[max_index], line_by_pts)
        # score_system, all_pt_tree, all_pts = gc.score_points(flat_verified, all_sects, line_corners, corners)

        """
        Line complete method (not enough time to finish + debug)
        """
        # g1_pts, g2_pts = lc.get_outer_points(sect_list[max_index])
        # print("g line points")
        # print(g1_pts)
        # print(g2_pts)
        # g1_pt1_list, g1_pt2_list = lc.insert_lines(g1_pts)
        # g2_pt1_list, g2_pt2_list = lc.insert_lines(g2_pts)
        # print("insert if needed", g1_pt1_list)
        # print(g1_pt2_list)
        # g1_pt1_inserts, g1_pt2_inserts, g1_need = lc.mask_lines(g1_pt1_list, g1_pt2_list)
        # g2_pt1_inserts, g2_pt2_inserts, g2_need = lc.mask_lines(g2_pt1_list, g2_pt2_list)
        # print("expand mask", g1_pt1_inserts, g1_pt2_inserts)
        # print(g2_pt1_inserts, g2_pt2_inserts)
        # left_1, left_2 = g1_pt1_inserts
        # right_1, right_2 = g1_pt2_inserts
        # g1_top_k = lc.get_best_k_lines(left_1, left_2, right_1, right_2, all_pts, score_system, g1_need)
        #
        # left_1, left_2 = g2_pt1_inserts
        # right_1, right_2 = g2_pt2_inserts
        # g2_top_k = lc.get_best_k_lines(left_1, left_2, right_1, right_2, all_pts, score_system, g2_need)
        # print("top_k", g1_top_k)
        # print(g2_top_k)
        # lc.show_lines(g1_top_k + g1_pts, image)
        # lc.show_lines(g2_top_k + g2_pts, image)

        """
        sect_list = the intersection between the two groups represented as points
        line_pts = the intersection between lines and corner points
        corners = all corner points found by a corner detection function
        """
        true_corners = gc.point_interpolate(g1_lines,
                                        g2_lines,
                                        sect_list[max_index],
                                        line_by_pts,
                                        corners,
                                        image.shape[:2],
                                        image=image,
                                        lines=lines)
        if true_corners is None:
            return None
        scaled_corners = []
        for pt in true_corners:
            x, y = pt
            scaled_corners.append((x * image_scale[0], y * image_scale[1]))
        return scaled_corners
    else:
        print("No valid grid found")
        print("Failed check_all_grids")
        return None


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
    taken = "./Taken_Photos"
    generated = "./Generated_photos"
    taken_files = [f for f in os.listdir(taken) if os.path.isfile(os.path.join(taken, f))]
    gen_files = [f for f in os.listdir(generated) if os.path.isfile(os.path.join(generated, f))]
    # for j in range(len(taken)):
    #     detect_chessboard(os.path.join(taken, taken_files[j]), ksize)

    for j in range(len(medium)):
        detect_chessboard(medium[j], ksize)


if __name__ == "__main__":
    main()