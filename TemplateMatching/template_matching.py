import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import cv2
from TemplateMatching.crop_img import crop_img


img = cv2.imread('img.PNG')
img2 = img.copy()
template = crop_img("img.PNG")
# template = cv2.imread('template.PNG')
w, h = template.shape[1], template.shape[0]

res = cv2.matchTemplate(img, template, 3)  # 3 for 'cv.TM_CCORR_NORMED'
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, 255, 2)

plt.subplot(121), plt.imshow(res, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle('Normed cross correlation')
plt.show()