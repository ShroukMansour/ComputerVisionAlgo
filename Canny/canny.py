import cv2
import numpy as np
import matplotlib.pyplot as plt


def smooth_img(img, size, sigma):
    return cv2.GaussianBlur(img, (size, size), sigmaX=0.5, sigmaY=0.5)


def quantize_angle(angle):
    angle[(angle <= 22.5) | (angle > 157.5)] = 0
    angle[(22.5 < angle) & (angle <= 67.5)] = 45
    angle[(67.5 < angle) & (angle <= 112.5)] = 90
    angle[(112.5 < angle) & (angle <= 157.5)] = 135
    return angle


def apply_gradient(img, size):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=size)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=size)
    gradient, angle = cv2.cartToPolar(sobelx, sobely)
    angle = angle * 180 / np.pi
    angle[angle > 180] = angle[angle > 180] - 180
    angle = quantize_angle(angle)
    return gradient, angle


def is_local_maximum(gradient, angle, i, j):
    if angle[i, j] == 0:
        if gradient[i, j] >= gradient[i, j - 1] and gradient[i, j] >= gradient[i, j + 1]:
            return True
        else:
            return False
    elif angle[i, j] == 45:
        if gradient[i, j] >= gradient[i + 1, j - 1] and gradient[i, j] >= gradient[i - 1, j + 1]:
            return True
        else:
            return False
    elif angle[i, j] == 135:
        if gradient[i, j] >= gradient[i - 1, j - 1] and gradient[i, j] >= gradient[i + 1, j + 1]:
            return True
        else:
            return False
    elif angle[i, j] == 90:
        if gradient[i, j] >= gradient[i - 1, j] and gradient[i, j] >= gradient[i + 1, j]:
            return True
        else:
            return False


def non_max_suppression(gradient, angle):
    thinned = np.array(gradient)
    for i in range(1, gradient.shape[0] - 1):
        for j in range(1, gradient.shape[1] - 1):
            if not is_local_maximum(gradient, angle, i, j):
                thinned[i, j] = 0
    return thinned


def is_strong_edge(edges, angle, i, j):
    if angle[i, j] == 90:
        if edges[i, j - 1] == 255 or edges[i, j + 1] == 255:
            return True
        else:
            return False
    elif angle[i, j] == 135:
        if edges[i + 1, j - 1] == 255 or edges[i - 1, j + 1] == 255:
            return True
        else:
            return False
    elif angle[i, j] == 45:
        if edges[i - 1, j - 1] == 255 or edges[i + 1, j + 1] == 255:
            return True
        else:
            return False
    elif angle[i, j] == 0:
        if edges[i - 1, j] == 255 or edges[i + 1, j] == 255:
            return True
        else:
            return False


def classify_edges(gradient):
    edges = gradient[gradient > 0]
    min_val = np.percentile(edges, 20)
    max_val = np.percentile(edges, 60)
    edges_classified = np.full(gradient.shape, 128)
    edges_classified[gradient >= max_val] = 255
    edges_classified[gradient <= min_val] = 0
    return edges_classified


def get_neighbor_points(point_angle, point_pos):
    i = point_pos[0]
    j = point_pos[1]
    if point_angle == 90:
        return (i, j - 1), (i, j + 1)

    elif point_angle == 135:
        return (i + 1, j - 1), (i - 1, j + 1)

    elif point_angle == 45:
        return (i - 1, j - 1), (i + 1, j + 1)

    elif point_angle == 0:
        return (i - 1, j), (i + 1, j)


def mark_pixel(edges_classified, edge_angle, point_pos, after):
    if after:
        _, next_point = get_neighbor_points(edge_angle, point_pos)
    else:
        next_point, _ = get_neighbor_points(edge_angle, point_pos)
        # print(edge_angle)
    try:
        if edges_classified[next_point] == 255:
            mark_pixel(edges_classified, edge_angle, next_point, after)
        elif edges_classified[next_point] == 0:
            return
        elif edges_classified[next_point] == 128:
            edges_classified[next_point] = 255
            mark_pixel(edges_classified, edge_angle, next_point, after)
            return

    except IndexError:
        return


def connect_edge(edges_classified, edge_angle, point_pos):
    mark_pixel(edges_classified, edge_angle, point_pos, after=True)
    mark_pixel(edges_classified, edge_angle, point_pos, after=False)


def hysteresis_thresholding_bonus(gradient, angle):
    edges_classified = classify_edges(gradient)
    canny_edges = np.array(edges_classified)
    for i in range(0, edges_classified.shape[0]):
        for j in range(0, edges_classified.shape[1]):
            if edges_classified[i, j] == 255:
                connect_edge(canny_edges, angle[i,j], (i,j))
    canny_edges[canny_edges == 128] = 0
    return canny_edges


def hysteresis_thresholding(gradient, angle):
    edges_classified = classify_edges(gradient)
    canny_edges = edges_classified
    for i in range(1, edges_classified.shape[0] - 1):
        for j in range(1, edges_classified.shape[1] - 1):
            if edges_classified[i, j] == 128:
                if is_strong_edge(edges_classified, angle, i, j):
                    canny_edges[i, j] = 255
                else:
                    canny_edges[i, j] = 0

    return canny_edges


def apply_canny(img):
    smoothed = smooth_img(img, size = 3, sigma=0.5)
    gradient, angle = apply_gradient(smoothed, size=3)
    plt.imshow(gradient, cmap='gray')
    plt.show()
    thinned_image = non_max_suppression(gradient, angle)
    plt.imshow(thinned_image, cmap='gray')
    plt.show()
    canny_edges = hysteresis_thresholding_bonus(thinned_image, angle)
    plt.imshow(canny_edges, cmap='gray')
    plt.show()
    plt.imshow(cv2.Canny(img, 100, 200), cmap='gray')
    plt.show()


# img = plt.imread("dog.jpg")
img = cv2.imread("sudoku.jpg", 0)
apply_canny(img)
