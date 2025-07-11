import sys
import math
import cv2
import numpy as np
from ChessBoardDetector import filter_grids as fg
from ChessBoardDetector import HarrisCornerDetection as hcd
from ChessBoardDetector import HoughTransform as ht


def cv_sort_rho(lines):
    """
    Sort and groups lines by their rho (distance to the original, top left of images)
    Uses coefficient variance in order to
    :param lines: unpacked 2D array of rho,theta values
    :return: A 2D list that are grouped based off gap between them,
            with a 1D array representing gap average of each group of lines
    """
    
