import numpy as np
import cv2
import scipy

def apply_homography(img, H):
    # Convert the NumPy array H to a format compatible with OpenCV
    H_cv = np.array(H, dtype=np.float64)

    # Determine the size of the output image
    output_size = (img.shape[1] * 1, img.shape[0] * 1)

    # Apply the homography
    img_out = cv2.warpPerspective(img, H_cv, output_size)

    return img_out


def extract_lines(line_points):
    # Define the 4x3 matrix with line points(corners found by aruco marker center)

    # Initialize a 4x3 matrix to store lines
    lines = np.zeros((4, 3), dtype=np.float64)

    # Compute the cross product of pairs of points to get lines
    lines[0] = np.cross(line_points[0], line_points[2])
    lines[1] = np.cross(line_points[1], line_points[3])
    lines[2] = np.cross(line_points[2], line_points[3])
    lines[3] = np.cross(line_points[0], line_points[1])

    return line_points, lines


def H_affine_rect(lines):
    # Intersection of line #1 and #2
    A1 = np.vstack([lines[0], lines[1]])
    vanishing_pt1 = scipy.linalg.null_space(A1).ravel()    # ravel() returns 1d contigurous flattened array, flatten() copies the original array
    # vanishing_pt1 /= vanishing_pt1[2]   # no need for normalization. scipy.linalg.null_space returns orthonormal basis for the null space using SVD
    A2 = np.vstack([lines[2], lines[3]])
    vanishing_pt2 = scipy.linalg.null_space(A2).ravel()

    # image of line at infinity l_prime
    A3 = np.vstack([vanishing_pt1, vanishing_pt2])
    image_of_line_inf = scipy.linalg.null_space(A3).ravel()

    H_ar = np.eye(3)
    H_ar[2, :] = image_of_line_inf
    return vanishing_pt1, vanishing_pt2, H_ar


# def H_metric_rect(vp1, vp2, H_ar):
#     # affine transformed vanishing point
#     vp1_prime = H_ar @ vp1
#     vp1_prime /= np.linalg.norm(vp1_prime)
#     vp2_prime = H_ar @ vp2
#     vp2_prime /= np.linalg.norm(vp2_prime)
#
#     # Directions
#     dir_vectors = np.array([
#         [vp1_prime[0], -vp1_prime[0], vp2_prime[0], -vp2_prime[0]],
#         [vp1_prime[1], -vp1_prime[1], vp2_prime[1], -vp2_prime[1]]
#     ])
#
#     thetas = [
#         abs(np.arctan2(dir_vectors[0, i], dir_vectors[1, i])) for i in range(4)
#     ]
#     thetas2 = [
#         np.arctan2(dir_vectors[0, 2], dir_vectors[1, 2]),
#         np.arctan2(dir_vectors[0, 3], dir_vectors[1, 3])
#     ]
#
#     hidx = np.argmin(thetas)
#     vidx = np.argmax(thetas2)
#     if hidx <= 2:
#         vidx += 2
#
#     # Metric rectification matrix
#     H_mr = np.array([
#         [dir_vectors[0, vidx], dir_vectors[0, hidx], 0],
#         [dir_vectors[1, vidx], dir_vectors[1, hidx], 0],
#         [0, 0, 1]
#     ])
#
#     if np.linalg.det(H_mr) < 0:
#         H_mr[0, :] *= -1
#     return H_mr



def H_metric_rect(lines):
    """Performs metric rectification using orthogonality constraints."""
    # Normalize lines by the third coordinate
    for i in range(4):
        lines[i] /= lines[i][2]

    # Extract lines
    l1 = lines[0]
    l2 = lines[1]
    l3 = lines[2]
    l4 = lines[3]

    # Formulate orthogonality constraints
    ortho_constraint = np.array([
        [l1[0] * l2[0], l1[0] * l2[1] + l1[1] * l2[0], l1[1] * l2[1]],
        [l3[0] * l4[0], l3[0] * l4[1] + l3[1] * l4[0], l3[1] * l4[1]]
    ])

    # Solve for s using SVD
    _, _, Vt = scipy.linalg.svd(ortho_constraint)
    s = Vt[-1]

    # Reconstruct symmetric matrix S
    S = np.array([
        [s[0], s[1]],
        [s[1], s[2]]
    ])

    # Decompose S into K
    U, D, _ = scipy.linalg.svd(S)
    K = U @ np.diag(np.sqrt(D)) @ U.T

    # Construct the metric rectification homography
    H_mr = np.array([
        [K[0, 0], K[0, 1], 0],
        [K[1, 0], K[1, 1], 0],
        [0, 0, 1]
    ])

    return H_mr





# Run the main function
if __name__ == "__main__":
    # img = cv2.imread("Align Example_screenshot_24.11.2024.png", 1)  # flags 1: read image as default(color)
    img = cv2.imread("original_screenshot2_25.11.2024.png")
    from Detect_Aruco import Detect_Aruco
    da = Detect_Aruco()

    pointlist = da.find_aruco(img = img)
    pointlist = np.concatenate((pointlist, np.ones(4).reshape(4, 1)), axis=1)

    line_points, lines = extract_lines(pointlist)
    vp1, vp2, H_ar = H_affine_rect(lines)

    img_aff = apply_homography(img, H_ar)
    img_aff_resized = cv2.resize(img_aff, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)


    # Apply metric rectification
    # H_mr = H_metric_rect(vp1, vp2, H_ar)
    H_mr = H_metric_rect(lines)
    img_metric = apply_homography(img, np.linalg.inv(H_mr) @ H_ar)
    img_metric_resized = cv2.resize(img_metric, (0, 0), fx=0.5, fy=0.5)


    print("Line Points:")
    print(line_points)
    print("Lines:")
    print(lines)
    print("Homography matrix for Affine Rectification:")
    print(H_ar)

    # Display the rectified image
    cv2.imshow("Affine Rectification", img_aff_resized)
    cv2.imshow("Metric Rectification", img_metric_resized)
    # cv2.moveWindow("Metric Rectification", 0, 500)
    # cv2.moveWindow("Affine Rectification", 100, 500)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

