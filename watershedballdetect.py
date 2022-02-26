import math

from cv2 import cv2
import time
import numpy as np

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

lowerG = (50,70, 60)
upperG = (80, 255, 255)

bounds = [[lowerG,upperG]]

while cam.isOpened():
    ret, image = cam.read()
    image = cv2.flip(image, 1)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    masks = [cv2.inRange(hsv, bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    masks = [cv2.erode(mask, None, iterations=2) for mask in masks]
    masks = [cv2.dilate(mask, None, iterations=2) for mask in masks]



    dist = cv2.distanceTransform(masks[0], cv2.DIST_L2, 5)
    ret, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    fg = np.uint8(fg)
    confuse = cv2.subtract(masks[0], fg)
    ret, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[confuse == 255] = 0

    wmarkers = cv2.watershed(image, markers.copy())

    contours, heirarchy = cv2.findContours(wmarkers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    num = 0
    areas = []
    for i in range(len(contours)):
        if heirarchy[0][i][3] == -1 and heirarchy[0][i][0] != -1 and cv2.contourArea(contours[i])>3000:
            areas.append(cv2.contourArea(contours[i]))
            num+=1
            cv2.drawContours(image, contours, i, (0, 255, 0), 10)

    # print(num)
    print((num))

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    cv2.imshow("Camera", image)
    cv2.imshow("Binary", np.uint8(wmarkers))

cam.release()
cv2.destroyAllWindows()