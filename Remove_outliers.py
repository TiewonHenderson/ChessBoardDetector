import sys
import cv2
import numpy as np
from scipy.optimize import curve_fit
from ChessBoardDetector import HoughTransform as ht
from ChessBoardDetector import cv_filter_groups as cvfg


def get_min_max_med_theta(indices, lines):
    """
    Helper function to get median to best represent overlapped lines
    Max theta in order see if the combined lines even fit the expected behavior
    """
    indices_len = len(indices)
    if indices_len == 0:
        return None, None
    thetas = sorted([(j, lines[j][1]) for j in indices], key=lambda x: x[1])
    return (thetas[0], thetas[-1], thetas[len(thetas) // 2])


def dp_find_longest_chain(interval, lines, threshold=0.05):
    n = len(interval)
    thetas = [get_min_max_med_theta(indices, lines) for indices in interval]
    """
    Quick GPT implemention using DP of find_longest_chain
    
    dp represents the len that index can create
    prev represents the last last index that represent the previous element as a chain
    
    so max(dp) index -> prev index, loop back through corresponding prev index's elements
    compare_type:
    False = med compare med, True = max > min
    """
    prev = [-1] * n
    dp = [1] * n
    compare_type = [False] * n

    for i in range(n):      # i shows the current items
        min_i, max_i, med_i = thetas[i]
        for j in range(i):  # j is the previous elements
            min_j, max_j, med_j = thetas[j]
            if med_j[-1] - min_i[-1] > -threshold and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
                compare_type[i] = False
            elif min_j[-1] - max_i[-1] > -threshold and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
                compare_type[i] = True

    # Reconstruct the longest decreasing chain
    max_len = max(dp)
    end_idx = dp.index(max_len)
    chain = []
    add_max = False
    while end_idx != -1:
        # True means add min, then max (overrides adding median)
        min_theta, max_theta, med_theta = get_min_max_med_theta(interval[end_idx], lines)
        if add_max:
            # first element gets index of lines itself
            chain.append(max_theta[0])
            add_max = False
        elif compare_type[end_idx]:
            chain.append(min_theta[0])
            add_max = True
        else:
            chain.append(med_theta[0])
        end_idx = prev[end_idx]

    return chain[::-1]  # reverse to get correct order


def remove_outlier_parallel(interval, lines, threshold=0.02):
    """
    Theta should remain relatively the same, so rho is considered more
    Gap should increase since its going towards the camera
    :param interval:
    :param lines:
    :param threshold: Maximum allowed deviation from expected trend
    :return:
    """

    def finalize_lines(final_indices, lines):
        """
        Helper function to combine all combination of indices
        Will not append None
        """
        i = 0
        finalize = []
        while i < len(final_indices) - 1:
            # start end index
            s1, e1 = final_indices[i]
            s2, e2 = final_indices[i + 1]
            if i == 0:
                if s1 is not None:
                    finalize.append(lines[s1])
            # With how lines are added, s2 is e1
            if e1 is not None:
                finalize.append(lines[e1])
            elif s2 is not None:
                finalize.append(lines[s2])
            if i == len(final_indices) - 2:
                if e1 is not None:
                    finalize.append(lines[e2])
                    break
            i += 1
        return finalize
    i = 0
    prev_change = -sys.maxsize
    final_indices = []
    while i < len(interval) - 1:
        if len(final_indices) == 0:
            lines1 = [lines[index] for index in interval[i]]
        else:
            lines1 = [lines[final_indices[-1][1]]]
        lines2 = [lines[index] for index in interval[i + 1]]
        rho_diffs = []
        indices = []
        for j, l1 in enumerate(lines1):
            for k, l2 in enumerate(lines2):
                # Check theta difference first, must be consistent
                t_diff = abs(l1[1] - l2[1])
                if t_diff > threshold:
                    continue
                rho_diffs.append(abs(l1[0] - l2[0]))
                indices.append((interval[i][j], interval[i + 1][k]))
        if len(rho_diffs) == 0:
            i += 1
            continue
        med = np.median(np.percentile(rho_diffs, [65]))
        """
        Needs further on why prev_change is used here (no trends)
        """
        if med >= prev_change:
            closest_index = min(range(len(rho_diffs)), key=lambda i: abs(rho_diffs[i] - med))
            final_indices.append(indices[closest_index])
            prev_change = med * (1 - threshold)
        else:
            # Fail safe
            final_indices.append(indices[0])
        i += 1

    # Finalize lines, we got best fit for each step, combine to one line
    return finalize_lines(final_indices, lines)


def brute_force_find(interval, lines, direction, threshold=0.1, deviation=2.5):
    """
    Uses a brute force chaining algorithm in order to find the longest chain of lines that suffices
    the expected behavior given the direction by degree

    Assuming interval and lines are sorted by x then y
    :param interval:
    :param lines:
    :param direction:
    :param threshold:
    :param deviation:
    :return:
    """

    """
    45, 225: Decreasing theta overall, so differences are positive
    Derivative of theta also decreases

    135, 315: Decreasing theta overall, so differences are positive
    Derivative of theta also decreases
    """
    decreasing_theta = {45, 135, 225, 315}

    # Curve fit functions
    def arctan_model(x, A, B, C, D):
        # Generated by GPT, arctan function fit
        return A * np.arctan(B * x + C) + D
    def linear(x, a, b):
        return a * x + b
    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    lines_c = lines
    if direction == 0 or direction == 180:
        lines_c = [(rho, (theta + np.pi/4) % np.pi) for rho, theta in lines]

    # Longest sequence of similar or decreasing theta
    max_chain = dp_find_longest_chain(interval, lines_c, threshold)
    if len(max_chain) < 3:
        return [lines[i] for i in max_chain]
    x = np.arange(len(max_chain))
    y = np.array([lines[i][-1] for i in max_chain])

    outlier_indices = None
    """
    All groups runs a linear fit if the camera is top down perspective
    90, 270 almost always fits a linear relationship if they didnt snap to decreasing_theta
    """
    params_linear, _ = curve_fit(linear, x, y)
    y_pred_lin = linear(x, *params_linear)
    resid_lin = np.abs(y - y_pred_lin)
    residuals = resid_lin
    final_param = params_linear
    if direction in decreasing_theta and len(max_chain) >= 4:
        """
        Curve fit an arctan function for persepctive skew cases
        """
        # x is just consecutive indices y is theta
        # p0 is the init guess as param, allows faster converge according to GPT
        try:
            params_arctan, _ = curve_fit(arctan_model, x, y, p0=[1.5, 1, 0, np.pi / 2])
            y_pred_arc = arctan_model(x, *params_arctan)

            # Compute residuals (idea by gpt over CV to determine parallel)
            resid_arc = np.abs(y - y_pred_arc)

            # Choose the better model based on sum of residuals
            if np.sum(resid_lin) >= np.sum(resid_arc):
                residuals = resid_arc
                final_param = params_arctan
        except:
            print("No found arctan params found, default to linear")
    elif direction == 0 or direction == 180 and len(max_chain) >= 3:
        """
        Curve fit an quadratic function for middle vanish point cases
        Same functionality as above but with quadratic fit
        """
        params_quad, _ = curve_fit(quadratic, x, y)
        y_pred_quad = quadratic(x, *params_quad)
        resid_quad = np.abs(y - y_pred_quad)
        if np.sum(resid_lin) >= np.sum(resid_quad):
            residuals = resid_quad
            final_param = params_quad

    # Now it evaluates the quadratic line of x with the found coefficients
    # We use it to compare to our theta (difference in residuals)
    mad, dist_med = cvfg.check_MAD(residuals, get_mad=True)
    outliers = dist_med > deviation * mad
    outlier_indices = np.where(outliers)[0]
    """
    TO-DO, make indices be kept track
    """
    finalize = []
    for i in range(len(max_chain)):
        if i in outlier_indices:
            continue
        finalize.append(lines[max_chain[i]])
    return finalize
