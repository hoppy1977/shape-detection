# https://www.geeksforgeeks.org/how-to-detect-shapes-in-images-in-python-using-opencv/

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random as rng

rng.seed(12345)

# reading image
#image = cv2.imread('IMG_5959.JPG')
image = cv2.imread('IMG_6101.JPG')
  
# converting image into grayscale image
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayscale = cv2.blur(grayscale, (3,3))

plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(grayscale,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

# perform edge detection
canny_output = cv2.Canny(grayscale, 10, 200)

contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
centers = [None]*len(contours)
radius = [None]*len(contours)
for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(contours_poly[i])
    centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

# Draw polygonal contour + bonding rects + circles
for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(drawing, contours_poly, i, color)
    cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

cv2.imshow('Contours', drawing)


# detect lines in the image using hough lines technique
lines = cv2.HoughLinesP(canny_output, 1, np.pi/180, 60, np.array([]), 50, 5)

# iterate over the output lines and draw them
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), (20, 220, 20), 10)

# show the image
#plt.imshow(image)
#plt.show()

plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(canny_output,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
