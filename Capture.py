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

try:
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', color_image)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()