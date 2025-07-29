import sys
import math
import cv2
import numpy as np
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import cv_filter_groups as cvfg
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit
from collections import Counter


def flatten_lines(lines):
    """Helper function to flatten grouped overlapping lines"""
    flat = []
    for l in lines:
        for x in l:
            flat.append(x)
    return flat


def put_lines(lines, image, color, thickness=2):
    if len(lines) != 0:
        for rho, theta in lines:
            points = get_line(rho,theta)
            # Display line using openCV (GPT)
            cv2.line(image, points[0], points[1], color=color, thickness=thickness)


def mean_lines(line1, line2):

    """GPT suggested to double theta then use trig functions to get circular theta means"""
    rho1, theta1 = line1
    rho2, theta2 = line2
    rho1, rho2 = abs(rho1), abs(rho2)
    mean_rho = (rho1 + rho2) / 2
    x = np.cos(2 * np.array([theta1, theta2]))
    y = np.sin(2 * np.array([theta1, theta2]))
    mean_theta = 0.5 * np.arctan2(np.mean(y), np.mean(x))
    return mean_rho, mean_theta % np.pi


def normalize_rho(lines):
    for i, l in enumerate(lines):
        rho, theta = l
        if rho < 0:
            rho *= -1
            theta += np.pi
            theta = theta % (2 * np.pi)
            lines[i] = [rho, theta]
    return lines


def circle_theta(theta):
    """
    Adds 45 degree to theta, mainly used on directions being 0, 180
    :param theta:
    :return:
    """
    return (theta + np.pi/4) % np.pi


def slope_from_theta(theta, max):
    """
    The max slope for the lines would be the height of the image
    :param theta: Theta to convert to slope
    :param max: The height of the image
    :return: Slope of the line cooresponding to the theta inputted
    """
    if np.isclose(np.sin(theta), 0):
        return max
    return -np.cos(theta) / np.sin(theta)


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
    Update:
    Really only works when the thetas are similar to each other
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
