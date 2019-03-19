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
    w = int((w - 1) / 2)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            sub_Ixx = Ixx[i - w:i + w + 1, j - w:j + w + 1]
            sub_Ixy = Ixy[i - w:i + w + 1, j - w:j + w + 1]
            sub_Iyy = Iyy[i - w:i + w + 1, j - w:j + w + 1]
            M = np.array([[np.sum(sub_Ixx), np.sum(sub_Ixy)], [np.sum(sub_Ixy), np.sum(sub_Iyy)]])
            R_matrix[i][j] = np.linalg.det(M) - sigma * (np.trace(M) ** 2)
    img[R_matrix > 7000000] = [0, 0, 255]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def find_corners2(img, ksize, w, sigma):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gray = cv2.GaussianBlur(gray, ksize=(ksize, ksize), sigmaX=0.5, sigmaY=0.5)
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    Ixx = cv2.GaussianBlur(Ix * Ix, (w, w), sigmaX=0.5, sigmaY=0.5)
    Ixy = cv2.GaussianBlur(Ix * Iy, (w, w), sigmaX=0.5, sigmaY=0.5)
    Iyy = cv2.GaussianBlur(Iy * Iy, (w, w), sigmaX=0.5, sigmaY=0.5)

    rows, cols = gray.shape
    R_matrix = np.zeros_like(gray)
    for i in range(0, rows):
        for j in range(0, cols):
            M = np.array([[Ixx[i, j], Ixy[i, j]], [Ixy[i, j], Iyy[i, j]]])
            R_matrix[i][j] = np.linalg.det(M) - sigma * (np.trace(M) ** 2)
    R_matrix[R_matrix < 700000] = 0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = np.array(R_matrix[i - 1:i + 2, j - 1:j + 2])
            arg = np.argmax(window)
            index = np.unravel_index(arg, window.shape)
            if index != (1, 1):
                R_matrix[i, j] = 0

    img[R_matrix > 0] = [0, 0, 255]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


img = cv2.imread('check_rot.bmp')
find_corners2(img, 3, 3, 0.05)
img = cv2.imread('simA.jpg')
find_corners2(img, 3, 3, 0.05)
img = cv2.imread('simB.jpg')
find_corners2(img, 3, 3, 0.05)
img = cv2.imread('transA.jpg')
find_corners2(img, 3, 3, 0.05)
img = cv2.imread('transB.jpg')
find_corners2(img, 3, 3, 0.05)
