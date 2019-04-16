import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
    estimate trans:
        1-get p1, p2 (perfectly aligned)
        2-loop over 50 itr:
            a- get random 3 points from p1, 3 points from p2
            b- estimate T 
            c- p2` = p1 * T 
            d- calc num of inliers between p2` and p2 (threshold = 10)
            e- if (num of inliers > max_num_inliers) = best_T = T; 
            
    
"""


def get_matched_keypoints(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches


def draw_matches(kp1, kp2, matches):
    keypoints_img1 = cv2.drawKeypoints(img1, kp1, None)
    keypoints_img2 = cv2.drawKeypoints(img2, kp2, None)

    # Draw first 20 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

    plt.imshow(img3)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(keypoints_img1)
    plt.subplot(1, 2, 2)
    plt.imshow(keypoints_img2)
    plt.show()


def get_aligned_matched_points(kp1, kp2, matched):
    interest_pts1 = []
    interest_pts2 = []
    for match in matched:
        interest_pts1.append(kp1[match.queryIdx].pt)
        interest_pts2.append(kp2[match.trainIdx].pt)
    return np.array(interest_pts1, dtype=int), np.array(interest_pts2, dtype=int)


def get_num_of_iterations():
    return 50


def get_X_matrix(points):
    X = []
    for i in range(len(points)):
        r1 = [points[i][0], points[i][1], 0, 0, 1, 0]
        r2 = [0, 0, points[i][0], points[i][1], 0, 1]
        X.append(r1)
        X.append(r2)
    return np.array(X)


def get_transformation(points1, points2):
    X = get_X_matrix(points1)
    y = np.array(points2).flatten()
    Xinv = np.linalg.inv(X)
    T = np.matmul(Xinv, y)
    return T


def get_num_inliers(img2, p2, p2_dash, threshold):
    p2_dash = p2_dash.reshape(p2.shape)
    num_inliers = 0
    for i in range(0, len(p2), 2):
        distance = np.linalg.norm(np.array([p2[i], p2[i+1]]) - np.array([p2_dash[i], p2_dash[i+1]]))
        if distance < threshold:
            num_inliers = num_inliers + 1
    return num_inliers


def estimate_transformation(img2, interest_points1, interest_points2):
    iterations = get_num_of_iterations()
    max_num_inliers = -1; best_transformation = []
    for i in range(iterations):
        idx = np.random.randint(len(interest_points1), size=3)
        points1 = interest_points1[idx, :]
        points2 = interest_points2[idx, :]
        if np.linalg.det(get_X_matrix(points1)) == 0:
            continue
        transformation = get_transformation(points1, points2)
        interest_points2_transformed = apply_transformation(interest_points1, transformation)
        num_of_inliers = get_num_inliers(img2, interest_points2, interest_points2_transformed, 10)
        if num_of_inliers > max_num_inliers:
            max_num_inliers = num_of_inliers
            best_transformation = transformation
    print("max num of inliers ", max_num_inliers)
    return best_transformation


def apply_transformation(points, T):
    X = get_X_matrix(points)
    y = np.array(np.matmul(X, T), dtype=int)
    return y


def transform_img(img, T):
    transformed_img = np.zeros(img.shape)
    img_points = [[x, y] for x in range(img.shape[0]) for y in range(img.shape[1])]
    y = apply_transformation(img_points, T)
    idx = 0
    for i in range(img.shape[0]-2):
        for j in range(img.shape[1]-2):
            transformed_img[i][j] = img[y[idx]][y[idx+1]]
            idx = idx + 2
    return transformed_img


img1 = cv2.cvtColor(cv2.imread('1.jpg'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('2.jpg'), cv2.COLOR_BGR2RGB)
keypoints1, keypoints2, matched_keypoints = get_matched_keypoints(img1, img2)
# draw_matches(keypoints1, keypoints2, matched_keypoints)

best_matched = matched_keypoints[0:20]
interest_points1, interest_points2 = get_aligned_matched_points(keypoints1, keypoints2, best_matched)
transformation = estimate_transformation(img2, interest_points1, interest_points2)

img2_transformed = transform_img(img2, transformation).astype(np.uint8)
plt.imshow(img2_transformed), plt.title('Warped Image')
plt.show()
