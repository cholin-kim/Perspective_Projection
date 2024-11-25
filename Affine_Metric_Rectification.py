import numpy as np
# from PIL import Image
import cv2
# img = np.array(Image.open("Align Example_screenshot_24.11.2024.png"))

def apply_homography(img, H):
    '''
    img 상의 모든 점을 homography transform
    '''
    # Convert the NumPy array H to a format compatible with OpenCV
    H_cv = np.array(H, dtype=np.float64)

    # Determine the size of the output image
    output_size = (img.shape[1] *1, img.shape[0] * 1)

    # Apply the homography
    img_out = cv2.warpPerspective(img, H_cv, output_size)

    return img_out


import cv2
import numpy as np


def main():
    img = cv2.imread("Align Example_screenshot_24.11.2024.png", 1)  # flags 1: read image as default(color)

    # Define the 4x3 matrix with line points(corners found by aruco marker center)
    line_points = np.array([
        [304, 191, 1],
        [866, 196, 1],
        [221, 435, 1],
        [927, 427, 1]
    ], dtype=np.float64)

    # Initialize a 4x3 matrix to store lines
    lines = np.zeros((4, 3), dtype=np.float64)

    # Compute the cross product of pairs of points to get lines
    line[0] = np.cross(line_points[0], line_points[2])
    line[1] = np.cross(line_points[1], line_points[3])
    line[2] = np.cross(line_points[2], line_points[3])
    line[3] = np.cross(line_points[1], line_points[2])

    return img, line_points, lines



# Run the main function
if __name__ == "__main__":
    img, line_points, lines = main()
    print("Line Points:")
    print(line_points)
    print("Lines:")
    print(lines)