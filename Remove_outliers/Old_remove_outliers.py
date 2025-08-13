import sys
import cv2
import numpy as np
from scipy.optimize import curve_fit
from ChessBoardDetector.Line_Point_detection import HoughTransform as ht
from ChessBoardDetector.Remove_outliers import cv_filter_groups as cvfg
"""
DEPRECATED OUTLIER REMOVAL ALGORITHM

Old implementation that treated grouped lines separatedly

But noticed every group ALWAYS decreases in theta in someway.
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