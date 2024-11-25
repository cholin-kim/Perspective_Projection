import cv2
import numpy as np
from cv_bridge import CvBridge
import CamParams as cam
class Detect_Aruco:
    def __init__(self):

        # self.distCoeffs = cam.distCoeffs
        # self.camMatrix = cam.camMatrix


        ## Utils ##
        self.bridge = CvBridge()


        ## Initialize ##
        self.corners = 0
        self.ids = 0
        self.rejected = 0
        self.aruco_visualization = 0
        self.detect_flag = False

        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        arucoParam = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(arucoDict, arucoParam)


    def find_aruco(self, img):
        # if img is None: img = self.cv2_img
        self.corners, self.ids, self.rejected = self.detector.detectMarkers(img)
        # print(len(self.corners))
        # assert len(self.corners) == 4
        if self.ids is None:
            self.ids = []
        else:
            self.ids = self.ids.ravel()
        # self.ids = self.ids.flatten()  ## {NoneType} object has no attribute 'flatten', convert to shape (n,)
        print("Detected markers: ", self.ids, "#", len(self.corners))
        self.corners = np.array(self.corners).reshape(len(self.corners), 4, 2)
        # self.ids = np.sort(self.ids)
        self.corners = self.corners[np.argsort(self.ids)]
        pointlist = self.extract_corner()
        return pointlist

    def extract_corner(self):
        Pointlist = []
        for corner in self.corners:
            # corner = np.array(corner).reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corner

            stackedPoint = np.vstack((topRight, topLeft, bottomRight, bottomLeft))
            middlePoint = [np.mean(stackedPoint[:, 0]), np.mean(stackedPoint[:, 1])]
            middlePoint = np.array([int(middlePoint[0]), int(middlePoint[1])])
            Pointlist.append(middlePoint)
        return Pointlist

    def display_points(self, points, img):
        # if img is None: img = self.cv2_img
        for point in points:
            u, v = point[0], point[1]
            cv2.circle(img,(u, v), 1, color=(0, 255, 0), thickness=-1)
        # cv2.imshow("", img)
        # cv2.waitKey()
        return img




if __name__ == '__main__':
    da = Detect_Aruco()
    from PIL import Image

    img = np.array(Image.open("Align Example_screenshot_24.11.2024.png"))


    # while True:
    pointlist = da.find_aruco(img)
    print(pointlist)
    # da.display_points(pointlist, img)


