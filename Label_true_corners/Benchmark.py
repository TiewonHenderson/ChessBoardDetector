import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ChessBoardDetector import Chessboard_detection as cd
from ChessBoardDetector.Sams_X_corner import FindChessboards as fcd


def order_corners(pts):
    """
    Given 4 points, return them ordered as:
    top-left, top-right, bottom-right, bottom-left
    """
    if pts is None:
        return None
    pts = sorted(pts, key=lambda p: p[0])  # sort by x

    left_pts = sorted(pts[:2], key=lambda p: p[1])   # sort left two by y
    right_pts = sorted(pts[2:], key=lambda p: p[1])  # sort right two by y

    ordered = [left_pts[0], right_pts[0], right_pts[1], left_pts[1]]
    return ordered


def read_points():
    INPUT_FILE = "points.csv"
    data = {}
    with open(INPUT_FILE, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row

        for row in reader:
            # row[0] is the image filename
            image_name = row[0]

            # The rest are coordinates: x1, y1, x2, y2, x3, y3, x4, y4
            coords = list(map(float, row[1:]))  # Convert to float (or int if guaranteed whole numbers)

            # Reshape coords into [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

            data[image_name] = points

    return data


def polygon_iou(gt_points, pred_points, img_width, img_height):
    """
    Compute IoU between two quadrilaterals defined by 4 corner points each.

    Parameters:
        gt_points:   list of 4 (x,y) tuples for ground truth polygon
        pred_points: list of 4 (x,y) tuples for predicted polygon
        img_width:   width of the image canvas
        img_height:  height of the image canvas

    Returns:
        iou: float between 0 and 1 representing polygon overlap
    """
    if pred_points is None or len(pred_points) != 4:
        return 0
    # Create empty binary masks
    gt_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    pred_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Convert points to numpy int32 arrays as required by fillPoly
    gt_poly = np.array(gt_points, dtype=np.int32).reshape((-1, 1, 2))
    pred_poly = np.array(pred_points, dtype=np.int32).reshape((-1, 1, 2))

    # Fill polygons on masks
    cv2.fillPoly(gt_mask, [gt_poly], 1)
    cv2.fillPoly(pred_mask, [pred_poly], 1)

    # Compute intersection and union
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()

    if union == 0:
        # Both polygons empty (rare), define IoU as 1
        return 1.0

    iou = intersection / union
    return int(iou * 100)


def save_with_two_scores(img, score1, score2, output_path):
    h, w = img.shape[:2]
    margin = 220  # wider margin for two scores

    canvas = np.ones((h, w + margin, 3), dtype=np.uint8) * 255
    canvas[:, :w] = img

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    padding = 10

    texts = [f"Sams: {score1:.1f}%", f"Ours: {score2:.1f}%"]

    # Calculate max text width and height for consistent box size
    sizes = [cv2.getTextSize(text, font, font_scale, thickness)[0] for text in texts]
    max_width = max(w for (w, h) in sizes)
    max_height = max(h for (w, h) in sizes)

    # Vertical positions for two boxes (centered vertically with spacing)
    total_height = 2 * (max_height + 2*padding) + padding  # two boxes + space between
    start_y = (h - total_height) // 2

    for i, text in enumerate(texts):
        box_x1 = w + 10
        box_y1 = start_y + i * (max_height + 2*padding + padding)
        box_x2 = box_x1 + max_width + 2*padding
        box_y2 = box_y1 + max_height + 2*padding

        # Draw white rectangle background
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (255,255,255), thickness=-1)

        # Text baseline to position text inside box
        baseline = cv2.getTextSize(text, font, font_scale, thickness)[1]

        text_x = box_x1 + padding
        text_y = box_y2 - padding - baseline

        # Put text over the rectangle
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (0,128,0), thickness)

    cv2.imwrite(output_path, canvas)


def draw_grid_and_polygons(
        image,  # Input image (BGR)
        gt_points=None,  # Ground truth points (list of 4 (x,y) tuples)
        corner_list_1=None,  # First polygon points (list of 4 (x,y))
        corner_list_2=None  # Second polygon points (list of 4 (x,y))
):
    img_h, img_w = image.shape[:2]

    overlay = image.copy()

    # --- Draw ground truth polygon ---
    if gt_points is not None and len(gt_points) >= 3:
        gt_pts_np = np.array(gt_points, dtype=np.int32).reshape((-1, 1, 2))

        # Filled transparent green (BGR: (0,255,0))
        cv2.fillPoly(overlay, [gt_pts_np], color=(0, 255, 0))

        # Solid green outline
        cv2.polylines(image, [gt_pts_np], isClosed=True, color=(0, 255, 0), thickness=2)

        # Blend transparent fill
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # --- Draw corner_list_1 polygon (blue, no fill) ---
    if corner_list_1 is not None and len(corner_list_1) >= 2:
        pts1_np = np.array(corner_list_1, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts1_np], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue BGR

    # --- Draw corner_list_2 polygon (yellow, no fill) ---
    if corner_list_2 is not None and len(corner_list_2) >= 2:
        pts2_np = np.array(corner_list_2, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts2_np], isClosed=True, color=(0, 255, 255), thickness=2)  # Yellow BGR

    return image


def plot_score_histograms(sams_score, our_score, output_path="scores_histogram.png"):
    bins = list(range(0, 101, 10))  # 0,10,20,...100
    bin_labels = [f"{i+1}-{i+10}" for i in range(0, 100, 10)]
    bin_labels[0] = "0-10"  # fix first bin label

    # Compute histogram counts
    sams_counts, _ = np.histogram(sams_score, bins=bins)
    our_counts, _ = np.histogram(our_score, bins=bins)

    x = np.arange(len(bin_labels))  # positions for groups

    width = 0.35  # bar width

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Bar plot for sams_score
    ax1.bar(x, sams_counts, width, color='tab:blue')
    ax1.set_title('Sams Score Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, rotation=45)
    ax1.set_xlabel('Score Intervals')
    ax1.set_ylabel('Count')

    # Bar plot for our_score
    ax2.bar(x, our_counts, width, color='tab:orange')
    ax2.set_title('Our Score Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, rotation=45)
    ax2.set_xlabel('Score Intervals')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


valid_pts = read_points()
all_photos = "../All_photos"
output = "./Scored_images"
files = [f for f in os.listdir(all_photos) if os.path.isfile(os.path.join(all_photos, f))]
sams_score = []
our_score = []
for f in files:
    file_name = os.path.join(all_photos, f)
    image = cv2.imread(file_name)
    h,w, _ = image.shape
    gt_corners = order_corners(valid_pts[f])
    sams_corner = order_corners(fcd.processSingleCustomWrapper(file_name))
    our_corner = order_corners(cd.detect_chessboard(file_name))

    sams_score.append(polygon_iou(gt_corners, sams_corner, h, w))
    our_score.append(polygon_iou(gt_corners, our_corner, h, w))

    display_score = draw_grid_and_polygons(image, gt_corners, sams_corner, our_corner)

    output_image_dir = os.path.join(output, f)
    save_with_two_scores(display_score, sams_score[-1], our_score[-1], output_image_dir)

plot_score_histograms(sams_score, our_score)