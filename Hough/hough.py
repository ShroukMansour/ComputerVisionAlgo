
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('check_rot.bmp')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
# plt.imshow(edges)
# plt.show()

lines = cv2.HoughLines(edges,1,np.pi/180,50)
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# cv2.imwrite('houghlines5.jpg',img)
cv2.imshow("hello", img)
cv2.waitKey(0)