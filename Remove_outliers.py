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


def find_longest_chain(interval, lines, threshold=0.05):
    """
    TO-DO, implement memoization to search below, Bottom up prob works

    1) First DP searches through interval to get long chains of all thetas almosts always decreasing
    :param interval:
    :param lines:
    :param threshold:
    :return:
    """
    chains = set()
    """
    i represents the starting index of the chains
    """
    for i, indices in enumerate(interval):
        curr_chain = []
        add = False
        i_thetas = None
        for k in range(i + 1, len(interval)):
            if len(curr_chain) == 0:
                # 0 = min, 1 = max, 2/-1 = med
                i_thetas = get_min_max_med_theta(indices, lines)
            else:
                # Fakes having a max to avoid out of bounds
                i_thetas = (curr_chain[-1], curr_chain[-1])

            # Check med difference, then max difference if not within threshold
            # (index, theta) format
            k_thetas = get_min_max_med_theta(interval[k], lines)
            k_med = k_thetas[-1][-1]
            k_min = k_thetas[0][-1]
            k_max = k_thetas[1][-1]

            # add used to prevent duplicate adds
            if i_thetas[-1][-1] - k_med > -threshold:
                if not add:
                    curr_chain.append(i_thetas[-1])
                curr_chain.append(k_thetas[-1])
                add = True
            elif i_thetas[1][-1] - k_min > -threshold:
                if not add:
                    curr_chain.append(i_thetas[1])
                curr_chain.append(k_thetas[0])
                add = True
            else:
                add = False
        chains.add(tuple(curr_chain))

    return max(chains, key=len)


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


def remove_outlier_away(interval, lines, threshold=0.05, mad_thres=2.5):
    """
    Uses quadratic curve fitting in order to fit a best line that represents the theta
    Theta should still be decreasing overall
    :param interval:
    :param lines:
    :param threshold: Maximum allowed deviation from expected trend
    :param mad_thres: Threshold of how much mad* off the line can be from the fit quadratic func
    :return:
    """
    if len(interval) < 3:
        return []
    rep_lines = []
    x = np.arange(len(interval))
    y = []
    for l in interval:
        if len(l) > 1:
            overlap_ts = [ht.circle_theta(lines[i][1]) for i in l]
            # For overlapping lines, take median for now (could be improved on)
            med = np.median(overlap_ts)
            rep_lines.append(lines[min(l, key=lambda i: abs(lines[i][1] - med))])
            y.append(med)
        elif len(l) == 1:
            y.append(ht.circle_theta(lines[l[-1]][1]))
            rep_lines.append(lines[l[-1]])
    y = np.array(y)

    prune = []
    prev_index = 0
    # Theta should be decreasing overall, removing any major increases
    # Difference should be positive
    for i in range(1, len(rep_lines) - 1):
        t1 = ht.circle_theta(rep_lines[prev_index][1])
        t2 = ht.circle_theta(rep_lines[i][1])

        if t1 - t2 < -threshold:
            # if end, its most likely last line thats bad
            if i == len(rep_lines) - 1:
                prune.append(i)

            # if start, its most likely first line thats bad, but check neighbor
            elif prev_index == 0 and i + 1 < len(rep_lines):
                t3 = ht.circle_theta(rep_lines[i + 1][1])
                if t2 - t3 < threshold:
                    prune.append(i)
                if t1 - t3 < threshold:
                    prune.append(prev_index)
                continue

            # general case
            else:
                prune.append(i)
                continue
        prev_index = i

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    # Params is the parameters for a,b,c for quadratic function above
    params_quad, _ = curve_fit(quadratic, x, y)
    y_pred = quadratic(x, *params_quad)

    # Now it evaluates the quadratic line of x with the found coefficients
    # We use it to compare to our theta (difference in residuals)
    residuals = np.abs(y - y_pred)
    mad, dist_med = cvfg.check_MAD(residuals, get_mad=True)
    outliers = dist_med > mad_thres * mad
    outlier_indices = np.where(outliers)[0]

    finalize = []
    for i in range(len(interval)):
        if i in outlier_indices or i in prune:
            continue
        finalize.append(rep_lines[i])
    return finalize


def remove_outlier_parallel(interval, lines, threshold=0.02):
    """
    Theta should remain relatively the same, so rho is considered more
    Gap should increase since its going towards the camera
    :param interval:
    :param lines:
    :param threshold: Maximum allowed deviation from expected trend
    :return:
    """
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


def remove_outlier_norm(interval, lines, direction, threshold=0.05, deviation=3):
    """
    Using the direction of the group of lines' VP, we should expect some behaviors.
    Theta trends cannot be used for vanish points going towards 0 and 180

    Generally, the closer the grid lines is to the camera, the larger the theta change
    No matter 45, 135, 225, 315. Theta decreases and rate of change also decreases (if all lines are present)

    A static check for direction of vanishing points being 0 or 180
    The relationship must be:
    1) Growing theta that slows when approaching pi (smaller gaps of theta as continue)
    2) expect some near stable at the top
    3) Growing theta that speeds up afterwards (bigger gaps of theta as continue)

    :param interval:
    :param lines:
    :param direction: VP direction, doesn't work too well with direction == 0, 180
    :param threshold: Maximum allowed deviation from expected trend
    :param deviation: How much deviation is accepted off the found theta difference
    :return:
    """

    prev_change = sys.maxsize
    prev_miss = 1
    adaptive_thres = threshold
    final_indices = []
    i = 0

    while i < len(interval) - 1:
        # Uses previously found line if possible
        thetas1 = []
        if len(final_indices) == 0:
            thetas1 = [lines[index][1] for index in interval[i]]
        else:
            thetas1 = [lines[final_indices[-1][1]][1]]
        thetas2 = [lines[index][1] for index in interval[i + 1]]
        theta_diffs = []
        indices = []
        # get median THETA out of the interval (it only stores indices)
        for j, t1 in enumerate(thetas1):
            for k, t2 in enumerate(thetas2):
                t_diff = t1 - t2
                """
                characteristic:
                overall, theta decreases (expect positive t_diff)
                the rate of change for theta decreases
                """
                if t_diff > -adaptive_thres and t_diff <= prev_change and abs(t_diff) < abs(prev_change * deviation):
                    theta_diffs.append(t_diff)
                    indices.append((interval[i][j], interval[i + 1][k]))
        """
        median of theta_diffs -> closest to median or median itself
        median -> indices -> j, k of the two lines

        prev_change saves to med, NOW next theta change must exceed or be equal to it
        """
        if len(theta_diffs) == 0:
            # No theta_diff that suffices, skip and see if it fits forward
            prev_miss += 1
            adaptive_thres *= prev_miss
            i += 1
            continue
        best = min(theta_diffs, key=abs)
        score = abs(best)
        if score <= abs(prev_change):
            closest_index = min(range(len(theta_diffs)), key=lambda i: abs(theta_diffs[i] - best))
            final_indices.append(indices[closest_index])
            # prevent 0 degree changes (also expect some randomness for theta
            prev_change = best + threshold
            adaptive_thres = threshold
        else:
            # Fail safe
            final_indices.append(indices[0])
            prev_change = best * 2 + threshold
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

    print("residual", residuals)
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
