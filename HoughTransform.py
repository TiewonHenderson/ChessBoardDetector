import sys
import math
import cv2
import numpy as np
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import HarrisCornerDetection as hcd


def put_lines(lines, image, color, thickness=2):
    if len(lines) != 0:
        for rho, theta in lines:
            points = get_line(rho,theta)
            # Display line using openCV (GPT)
            cv2.line(image, points[0], points[1], color=color, thickness=thickness)


def show_images(image):
    # Display the result (GPT)
    cv2.imshow("",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_line(rho, theta):
    """
    Closest point of the found line formula:
    (x0,y0) = (ρ * cos(θ), ρ * sin(θ))

    Since cos = adj/hypt in respect of x axis, cos(theta) * rho (hypt) = x value
    Same applies to sin, sin(theta) * rho = y value
    """
    # (GPT)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # To expand the line, totalling to 3000 pixel length
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))
    return [(x1, y1), (x2, y2)]


def polar_theta(point):
    """
    Get the polar coordinate theta value from two lines
    point is a [[x1, y1, x2, y2]] type argument

    """
    x1, y1, x2, y2 = point[0]
    # Formula from GPT
    return np.arctan2(y2 - y1, x2 - x1) % np.pi


def polar_rho(point, theta):
    x1, y1, x2, y2 = point[0]
    # Formula from GPT
    return x1 * np.cos(theta) + y1 * np.sin(theta)


def unpack_hough(lines):
    """
    Unpacks the houghline 3D array into 2D array of [rho,theta] values
    """
    if lines is not None:
        return [[rho_theta[0][0], rho_theta[0][1]] for rho_theta in lines]
        # or using numpy
        # unpacked = lines[:, 0, :]  # shape (N, 2)
    else:
        return []


def orthogonal_gap(line1, line2):
    """
    Perpendicular spacing = Δρ/cos(θ)
    :param line1:
    :param line2:
    :return:
    """
    r1, t1 = line1
    r2, t2 = line2
    delta_rho = abs(r2 - r1)
    mean_theta = (t1 + t2) / 2

    denom = math.cos(mean_theta)
    if denom > 0.0001:
        return delta_rho / denom
    else:
        return delta_rho


def sort_Rho(lines, eps):
    """
    Sort lines by their rho (distance to the original, top left of images)
    Uses sliding window to get the best interval of similar gapped lines (gap is determined by rho)
    :param lines: unpacked 2D array of rho,theta values
    :param eps: how lenient the gap acceptance is (how off the difference can be to be accepted)
    :return: A 2D list that are grouped based off gap between them,
            with a 1D array representing gap average of each group of lines
    """
    lines.sort(key=lambda x: x[0])
    # temp = []
    # for i in range(0, len(lines) - 1):
    #     temp.append(abs(lines[i][0] - lines[i + 1][0]))
    # print("gaps: ", temp)

    # Not enough lines to be considered a grid
    if len(lines) <= 4:
        return [lines], []

    prev_gap = abs(orthogonal_gap(lines[0], lines[1]))
    candidates = []
    gap_average = [prev_gap]
    l = 0
    i = 1
    while i <= len(lines) - 1:
        gap_mean = gap_average[-1] / (i - l)
        if i == len(lines) - 1:
            gap_average[-1] = gap_mean
            candidates.append([l, i])
            break
        gap = abs(orthogonal_gap(lines[i], lines[i + 1]))
        # Since these are clustered as similar theta, same rho should be combined into 1 line
        if gap == 0:
            rho, t1 = lines[i]
            _, t2 = lines[i + 1]
            lines[i] = [rho, (t1 + t2)/2]
            lines.pop(i + 1)
            continue
        gap_eps = eps * gap_mean
        # End window if its above eps, continue otherwise
        if abs(gap - gap_mean) > gap_eps:
            gap_average[-1] = gap_mean
            gap_average.append(0)
            candidates.append([l, i])
            l = i
        prev_gap = gap
        gap_average[-1] += gap
        i += 1
    """
    Filters candidates, if there isn't any candidates totalling lines > 4
    The function will return the max lines candidate
    
    if there is candidates totalling lines > 4
    Include all in sorted_list
    """
    sorted_list = []
    max_cand = 0 # Stored by index
    for i, cand in enumerate(candidates):
        group_len = abs(cand[1] - cand[0])
        prev_max_len = abs(candidates[max_cand][1] - candidates[max_cand][0])
        # range() is [x,y), doesn't include y
        sorted_list.append([lines[i] for i in range(cand[0], cand[1] + 1)])
        if group_len > prev_max_len:
            max_cand = i
    if len(sorted_list) == 0:
        start, end = candidates[max_cand]
        return [[lines[i] for i in range(start, end + 1)]], gap_average
    return sorted_list, gap_average


def gap_Interpolation(sorted_list, gap_average, image_shape):
    """
    This function will determine if groups should be combined due to missing lines within a grid
    Some cases could be missing lines, causing a multiplied gap distance

    This would require these conditions:
    1)  The two groups have similar average gap value inside
    2)  The gap between the groups (not the gaps inside) is a multiple of the gap values
    3)  The sum or all lines WITH interpolated lines should not overcome a threshold,
        should never be over 18 lines total.
    :param: sorted_list/gap_average: should be straight from sort_Rhos
    :return:
    """
    line_groups = sorted_list.copy()
    group_gaps = gap_average.copy()

    # Group gap = the gap in between the two groups
    # Average gap = the average gap WITHIN a group
    if len(sorted_list) < 2 or len(gap_average) < 2:
        return sorted_list

    h, w = image_shape
    mean_eps = np.hypot(w,h) * 0.005
    group_gap_eps = 0.1

    i = 0
    while i < len(line_groups) - 1:
        groupA = line_groups[i]
        groupB = line_groups[i + 1]
        if len(groupA) == 0:
            line_groups.pop(i)
            group_gaps.pop(i)
            continue
        if len(groupB) == 0:
            line_groups.pop(i + 1)
            group_gaps.pop(i + 1)
            continue

        # Condition 1
        gap_diff = abs(group_gaps[i + 1] - group_gaps[i])
        if gap_diff <= mean_eps:

            # Condition 2
            groupA_end_rho = line_groups[i][-1][0]
            # Since the groups are inclusive both end, we must skip the index since
            # Last index of previous group == first index of next group
            groupB_start_rho = line_groups[i + 1][1][0]
            group_gap = abs(groupB_start_rho - groupA_end_rho)
            group_average = (group_gaps[i] + group_gaps[i + 1]) / 2
            # Calculates if the group gap diff is a multiple by eps
            ratio = group_gap / group_average
            m_value = round(ratio)  # Important, this would be the group gap multiplicity value
            diff = abs(ratio - m_value)
            # What this meant is only accept if <= 10% off a whole number multiplicity value
            if diff <= group_gap_eps:

                # Condition 3
                # The multiplicity of the group gap indicates the amount of interpolated lines inbetween
                total_lines = len(line_groups[i]) + (m_value - 1) + len(line_groups[i + 1])
                if total_lines <= 18:

                    # All conditions suffice
                    theta_avg = 0
                    for l in groupA:
                        theta_avg += l[1]
                    for l in groupB:
                        theta_avg += l[1]
                    theta_avg /= (len(groupA) + len(groupB))

                    for j in range(1, m_value):
                        line_groups[i].append([groupA_end_rho + (group_average * j), theta_avg])
                    line_groups[i].extend(groupB[1:])
                    line_groups.pop(i + 1)
                    group_gaps.pop(i + 1)
                    continue
        i += 1
    i = 0
    while i < len(line_groups):
        if len(line_groups[i]) < 3:
            line_groups.pop(i)
            continue
        i += 1
    return line_groups


def houghline_detect(edges, corners=None, mask=None, threshold=150, corner_eps=5, line_eps=10):
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
    lines = unpack_hough(cv2.HoughLines(e, 1, np.pi / 180, threshold=threshold))
    lines = fg.filter_similar_lines(lines, line_eps)
    if corners is not None:
        lines = hcd.filter_hough_lines_by_corners(lines, corners, tolerance=corner_eps)
    return lines


def image_load(image_name):
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

    # Create a binary mask with center region = 1, borders = 0
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

def main():
    """
    Load and grayscale image
    Gaussian blur
    Hough line transformation
    Harris corner detection
    Filter lines by harris corners and similarity
    Run grid detection on filtered lines
    """
    detection_params = [[11, 150, 200, 150, None, 1, 1.75],
                        [7, 100, 150, 100, 0.5, 0.55, 1],
                        [5, 85, 100, 80, 1.0, 0.25, 0.5]]
    image_name = "Real_Photos/Far,angled,solid.jpg"
    # 0 = strict, 1 = default
    detection_mode = 2
    image = cv2.imread(image_name)
    # Get image dimensions
    height, width, channels = image.shape
    if height * width > 2073600:
        image = cv2.resize(image, (1920, 1080))
        height = 1080
        width = 1920

    # Create a binary mask with center region = 1, borders = 0
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
    image_diagonal = np.hypot(width, height)
    # Epsilon list (suggested by GPT and tested)
    line_similarity_eps = int(image_diagonal * 0.01)
    corner_eps = int(image_diagonal * 0.05) * settings[5]
    gap_eps = 0.2 * settings[5]

    # Detection section
    corners = hcd.harris(image, settings[0])
    good_lines, lines = houghline_detect(edges, corners, mask, settings[3], corner_eps, line_similarity_eps)
    fg.static_cluster_lines(good_lines)
    # Too many lines breaks DBSCAN clustering
    # Clusters lines using DBSCAN with theta values
    clusters = fg.cluster_lines(good_lines, gap_eps)
    for key in clusters:
        """
        Sorts each cluster by rho value, then linearly scans to group each line by consistent gaps
        
        gap_Interpolation returns potentially combined intervals if gaps are consistent or multiples of
        each other (possibly indicating missing grid lines)
        
        saves the largest found group of parallel lines with consistent gaps
        """
        sorted_list, gap_average = sort_Rho(clusters[key], gap_eps)
        clusters[key] = max(gap_Interpolation(sorted_list, gap_average, [height, width]), key=len)
    cluster_list = list(clusters.items())
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

    if len(scores) > 0:
        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        for x, y in corners:
            cv2.circle(image, (x, y), 3, color=blue, thickness=-1)

        max_index = scores.index(max(scores))
        for i in range(len(clusters)):
            if i == max_index:
                put_lines(clusters[i], image, green)
            else:
                put_lines(clusters[i], image, red)
        show_images(image)


if __name__ == "__main__":
    main()