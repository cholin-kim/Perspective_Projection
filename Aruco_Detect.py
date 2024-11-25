import cv2
import numpy as np
from cv_bridge import CvBridge
import CamParams as cam
class Aruco_Detect():
    def __init__(self):

        self.distCoeffs = cam.distCoeffs
        self.camMatrix = cam.camMatrix


        ## Utils ##
        self.bridge = CvBridge()


        ## Initialize ##
        self.cv2_img = 0
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
        print("Detected markers: ", self.ids, "#", len(self.corners))
        assert len(self.corners) == 4
        self.ids = self.ids.flatten()  ## {NoneType} object has no attribute 'flatten', convert to shape (n,)
        pointlist = self.extract_corner()
        return pointlist

    def extract_corner(self):
        Pointlist = []
        for corner in self.corners:
            corner = np.array(corner).reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corner

            topRightPoint    = np.array([int(topRight[0]),      int(topRight[1])])
            topLeftPoint     = np.array([int(topLeft[0]),       int(topLeft[1])])
            bottomRightPoint = np.array([int(bottomRight[0]),   int(bottomRight[1])])
            bottomLeftPoint  = np.array([int(bottomLeft[0]),    int(bottomLeft[1])])
            stackedPoint = np.hstack(topRightPoint, topLeftPoint, bottomRightPoint, bottomLeftPoint)
            middlePoint = [np.mean(stackedPoint[0]), np.mean(stackedPoint[1])]
            Pointlist.append(middlePoint)
        return Pointlist




if __name__ == '__main__':
    ad = Aruco_Detect()

    while True:
        pointlist = ad.find_aruco()
        print("_________________________________")

