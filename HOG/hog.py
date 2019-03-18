import cv2
from PIL import Image
import numpy as np


def resize_image(img_path, input_shape=(128, 64)):
    image = Image.open(img_path).convert('L')
    iw, ih = image.size
    h, w = input_shape
    # resize image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('L', (w, h), (128))
    new_image.paste(image, (dx, dy))
    return np.asarray(new_image)


def smooth_img(img, size, sigma):
    return cv2.GaussianBlur(img, (size, size), sigmaX=sigma, sigmaY=sigma)


def apply_gradient(img, size):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=size)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=size)
    gradient, angle = cv2.cartToPolar(sobelx, sobely)
    angle = angle * 180 / np.pi
    angle[angle > 180] = angle[angle > 180] - 180
    return gradient, angle


def get_cell_fv(cell_gradient, cell_angle):
    fv = np.zeros(9)
    for i in range(cell_gradient.shape[0]):
        for j in range(cell_gradient.shape[1]):
            bin = int(cell_angle[i, j] // 20)
            current_bin_start = bin * 20
            current_bin_mid = current_bin_start + 10
            prev_bin_mid = (current_bin_mid - 20) % 180
            next_bin_mid = (current_bin_mid + 20) % 180
            if abs(cell_angle[i, j] - prev_bin_mid) < 20:
                fv[bin - 1] = fv[bin - 1] + cell_gradient[i, j] * (abs(cell_angle[i, j] - current_bin_mid) / 20)
                fv[bin] = fv[bin] + cell_gradient[i, j] * (abs(cell_angle[i, j] - prev_bin_mid) / 20)
            elif abs(cell_angle[i, j] - next_bin_mid) < 20:
                fv[bin + 1] = fv[bin + 1] + cell_gradient[i, j] * (abs(cell_angle[i, j] - current_bin_mid) / 20)
                fv[bin] = fv[bin] + cell_gradient[i, j] * (abs(cell_angle[i, j] - next_bin_mid) / 20)
    return fv


def get_block_fv(block_gradient, block_angle):
    fv = np.zeros((4, 9))
    x = 0
    for i in range(0, block_gradient.shape[0], 8):
        for j in range(0, block_gradient.shape[1], 8):
            fv[x, :] = get_cell_fv(block_gradient[i:i + 8, j:j + 8], block_angle[i:i + 8, j:j + 8])
            x = x + 1
    total = fv.sum()
    if total != 0:
        fv = fv / total
    return fv.flatten()


def get_img_fv(gradient, angle):
    fv = np.zeros((32, 36))
    x = 0
    for i in range(0, gradient.shape[0], 16):
        for j in range(0, gradient.shape[1], 16):
            fv[x, :] = get_block_fv(gradient[i:i + 16, j:j + 16], angle[i:i + 16, j:j + 16])
            x = x + 1
    return fv.flatten()


def apply_hog(img_path):
    img = resize_image(img_path, (128, 64))
    smoothed_img = smooth_img(img, 3, 0.5)
    gradient, angle = apply_gradient(smoothed_img, 3)
    fv = get_img_fv(gradient, angle)
    return fv


fv = apply_hog('sudoku.jpg')
