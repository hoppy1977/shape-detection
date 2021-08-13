import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes around lego pieces.')
parser.add_argument('--input', help='Path to input image.', default='IMG_6101.JPG')
args = parser.parse_args()

src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))

# Calculate contours
canny_threshold = 50
canny_output = cv.Canny(src_gray, canny_threshold, canny_threshold * 3)

contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
for i, c in enumerate(contours):
    contours_poly[i] = cv.approxPolyDP(c, 3, True)
    boundRect[i] = cv.boundingRect(contours_poly[i])

# Draw the contours
contours_drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (255, 255, 255)
    cv.drawContours(contours_drawing, contours_poly, i, color)

# Draw the original with bounding rectangle
rectangle_drawing = src.copy()
for i in range(len(contours)):
    color = (255, 0, 0)
    cv.rectangle(rectangle_drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)


plt.subplot(231)
plt.imshow(src)
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(232)
plt.imshow(src_gray, cmap = 'gray')
plt.title('Greyed and blured')
plt.xticks([]), plt.yticks([])

plt.subplot(233)
plt.imshow(canny_output)
plt.title('Canny output')
plt.xticks([]), plt.yticks([])

plt.subplot(234)
plt.imshow(contours_drawing, cmap = 'gray')
plt.title('Contours')
plt.xticks([]), plt.yticks([])

plt.subplot(235)
plt.imshow(rectangle_drawing)
plt.title('Contours with rectangles')
plt.xticks([]), plt.yticks([])

plt.show()