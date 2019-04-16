import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

img_ = cv2.cvtColor(cv2.imread('2.jpg'),cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(cv2.imread('1.jpg'),cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)[0:20]

matches = np.asarray(matches)

src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

print(H)
dst = cv2.warpPerspective(img_, H, (img.shape[1] + img_.shape[1], img.shape[0]))
plt.subplot(122),plt.imshow(dst), plt.title('Warped Image')
plt.show()
plt.figure()
dst[0:img.shape[0], 0:img.shape[1]] = img
cv2.imwrite('output.jpg',dst)
plt.imshow(dst)
plt.show()