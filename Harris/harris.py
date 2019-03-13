import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_corners(img, ksize, w, sigma):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    Ixx = Ix * Ix                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    rows, cols = gray.shape
    R_matrix = np.zeros_like(gray)
    w = int((w-1)/2)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            sub_Ixx = Ixx[i - w:i + w + 1, j - w:j + w + 1]
            sub_Ixy = Ixy[i - w:i + w+1, j - w:j + w+1]
            sub_Iyy = Iyy[i - w:i + w+1, j - w:j + w+1]
            M = np.array([[np.sum(sub_Ixx), np.sum(sub_Ixy)], [np.sum(sub_Ixy), np.sum(sub_Iyy)]])
            R_matrix[i][j] = np.linalg.det(M) - sigma * (np.trace(M) ** 2)
    img[R_matrix > 10000] = [0, 0, 255]
    img = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


img = cv2.imread('check_rot.bmp')
find_corners(img, 3, 3, 0.05)
img = cv2.imread('simA.jpg')
find_corners(img, 3, 3, 0.05)
img = cv2.imread('simB.jpg')
find_corners(img, 3, 3, 0.05)
img = cv2.imread('transA.jpg')
find_corners(img, 3, 3, 0.05)
img = cv2.imread('transB.jpg')
find_corners(img, 3, 3, 0.05)
