import numpy as np

'''
input: pixel coords of 4 rectangular placed aruco marker 
'''

topLeftid = 1
topRightid = 2
bottomLeftid = 3
bottomRightid = 4


m1 = (y1 - y3) / (x1 - x3)
m2 = (y2 - y4) / (x2 - x4)
c1 = -m1 * x3 + y3
c2 = -m2 * x4 + x4

intersect_x = (c2 - c1) / (m1 - m2)
intersect_y = m1 * intersect_x + c1

check if vanishing point inside the img or it locates outside of the img
