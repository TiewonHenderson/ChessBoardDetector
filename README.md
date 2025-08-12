## A prototype algorithm that finds the 4 chessboard corners

**Approach to this chessboard detector:**

1. HoughLine over a Canny edge mask over the image + Harris corner detection on the image.
2. Filter HoughLine results with known theta relationships.
3. Find two groups of lines that are consistent and relatively equal in theta.
4. Filter out noise lines that don't follow theta relationships, and don't follow gap consistency between lines.
5. Grade each group of lines by pairs (combination of intersection consistency, number of total lines, rho consistency).
6. Verify intersections with Harris corners.
7. Expand out the verified intersection to form a 9×9 point grid.
8. Run RANSAC on the found 9×9 point grid to get a grid homography based on grid square corners.
9. Offset the 4 corners of the homography by the relative distance between points to see which bounding box fits most grid corners.
10. Return the final found 4 corner points (None if anything went wrong in the middle steps).

To run the algorithm:
- Within Chessboard_detection.py
- Use the detect_chessboard() function with a valid image directory as an input
- Expected outcome:
1) A 4 element list that includes the 4 corner points found
2) None if no points could be found

---

###### Sam-o-bot's chessboard detection repository:
[https://github.com/Elucidation/ChessboardDetect](https://github.com/Elucidation/ChessboardDetect?tab=readme-ov-file#chessboard-detection)
