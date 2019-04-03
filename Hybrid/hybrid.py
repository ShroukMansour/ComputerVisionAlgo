from scipy import ndimage, misc
import matplotlib.pyplot as plt
import cv2
import numpy as np

def hybrid(img1, img2):

    lo_img = cv2.GaussianBlur(img2, sigmaX=100, sigmaY=100, ksize=(9,9))
    # hi_img = ndimage.gaussian_laplace(img2, sigma=1)
    hi_img = cv2.Laplacian(cv2.GaussianBlur(img1, sigmaX=3, sigmaY=3, ksize=(3,3)), ddepth=-1)
    hi_img = hi_img/np.max(hi_img)
    hi_img = hi_img*img1
    hi_img = hi_img.astype(int)
    plt.imshow(lo_img)
    plt.show()
    plt.imshow(hi_img, cmap='gray')
    plt.show()
    final = (lo_img + hi_img)%255
    plt.imshow(final)
    plt.show()


img1 = plt.imread("cat.jpg")
img2 = plt.imread("dog.jpg")
hybrid(img1, img2)
