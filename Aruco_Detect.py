import cv2
import numpy as np
from cv_bridge import CvBridge
import CamParams as cam
class Aruco_Detect():
    def __init__(self):

        # self.distCoeffs = cam.distCoeffs
        # self.camMatrix = cam.camMatrix


        ## Utils ##
        self.bridge = CvBridge()


        ## Initialize ##
        from PIL import Image
        self.cv2_img = np.array(Image.open("Align Example_screenshot_24.11.2024.png"))
        # self.cv2_img = 0
        self.corners = 0
        self.ids = 0
        self.rejected = 0
        self.aruco_visualization = 0
        self.detect_flag = False

        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        arucoParam = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(arucoDict, arucoParam)


    def find_aruco(self):
        self.corners, self.ids, self.rejected = self.detector.detectMarkers(self.cv2_img)
        # print(len(self.corners))
        # assert len(self.corners) == 4
        self.ids = self.ids.flatten()  ## {NoneType} object has no attribute 'flatten', convert to shape (n,)
        print("Detected markers: ", self.ids, "#", len(self.corners))
        pointlist = self.extract_corner()
        return pointlist

    def extract_corner(self):
        Pointlist = []
        for corner in self.corners:
            corner = np.array(corner).reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corner

            stackedPoint = np.vstack((topRight, topLeft, bottomRight, bottomLeft))
            middlePoint = [np.mean(stackedPoint[:, 0]), np.mean(stackedPoint[:, 1])]
            middlePoint = np.array([int(middlePoint[0]), int(middlePoint[1])])
            Pointlist.append(middlePoint)
        return Pointlist

    def display_points(self, points, img=None):
        if img is None: img = self.cv2_img
        for point in points:
            u, v = point[0], point[1]
            cv2.circle(img,(u, v), 1, color=(0, 255, 0), thickness=-1)
        cv2.imshow("", img)
        cv2.waitKey()




if __name__ == '__main__':
    ad = Aruco_Detect()


    # while True:
    pointlist = ad.find_aruco()
    print(pointlist)
    ad.display_points(pointlist)


