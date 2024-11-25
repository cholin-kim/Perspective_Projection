import pyrealsense2 as rs
import cv2
import numpy as np


print("reset start")
ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()
print("reset done")

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)


from Detect_Aruco import Detect_Aruco
da = Detect_Aruco()
from Affine_Metric_Rectification import *


a = np.ones(4).reshape(4, 1)

try:
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        pointlist = da.find_aruco(img=color_image)
        print(pointlist)

        cv2.namedWindow("original", cv2.WINDOW_NORMAL)
        cv2.namedWindow('Affine Rectification', cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('Metric Rectification', cv2.WINDOW_KEEPRATIO)



        if len(pointlist) == 4:
            cv2.imshow("original", da.display_points(pointlist, color_image))

            pointlist = np.concatenate((pointlist, a), axis=1)
            line_points, lines = extract_lines(line_points=pointlist)
            vp1, vp2, H_ar = H_affine_rect(lines)
            img_aff = apply_homography(color_image, H_ar)
            img_aff_resized = cv2.resize(img_aff, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

            # Apply metric rectification
            H_mr = H_metric_rect(vp1, vp2, H_ar)
            img_metric = apply_homography(color_image, np.linalg.inv(H_mr) @ H_ar)
            img_metric_resized = cv2.resize(img_metric, (0, 0), fx=0.25, fy=0.25)

            cv2.imshow("Affine Rectification", img_aff_resized)
            cv2.imshow("Metric Rectification", img_metric_resized)
            cv2.moveWindow("Metric Rectification", 0, 500)
            cv2.moveWindow("Affine Rectification", 100, 500)

        else:
            # img = da.display_points(pointlist, color_image)
            cv2.imshow("original", color_image)


        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()