import cv2
import csv
import os

def manual_corner_mark(filename):
    # ==== SETTINGS ====
    OUTPUT_FILE = "points.csv"
    needed_dim = 1000
    # ==================

    # Read original image
    orig_img = cv2.imread(filename)
    if orig_img is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    orig_h, orig_w = orig_img.shape[:2]

    scale_x = orig_w / needed_dim
    scale_y = orig_h / needed_dim

    # Resize for display (downscaled, not cropped)
    resized_img = cv2.resize(orig_img, (needed_dim, needed_dim))

    clicked_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < 4:
                # Convert clicked coords back to original scale
                orig_x = x * scale_x
                orig_y = y * scale_y
                clicked_points.append((round(orig_x), round(orig_y)))
                print(f"Point {len(clicked_points)}: ({orig_x:.2f}, {orig_y:.2f})")

                # Draw a small circle on the clicked point
                cv2.circle(resized_img, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Image", resized_img)

    # OpenCV window
    cv2.imshow("Image", resized_img)
    cv2.setMouseCallback("Image", mouse_callback)

    print("Click 4 points on the image (Press 'q' to quit early)")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(clicked_points) == 4:
            break
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Save to CSV
    if clicked_points:
        file_exists = os.path.isfile(OUTPUT_FILE)

        with open(OUTPUT_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)

            # Write header only if file doesn't exist yet
            if not file_exists:
                writer.writerow(["image", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])

            # Flatten points into one row
            row = [os.path.basename(filename)]
            for x, y in clicked_points:
                row.extend([x, y])

            writer.writerow(row)

        print(f"Appended {len(clicked_points)} points for {filename} to {OUTPUT_FILE}")
    else:
        print("No points were selected.")


def main():
    all_photos = "../All_photos"
    files = [f for f in os.listdir(all_photos) if os.path.isfile(os.path.join(all_photos, f))]
    for f in files:
        print(os.path.join(all_photos, f))
        manual_corner_mark(os.path.join(all_photos, f))


if __name__ == "__main__":
    main()
