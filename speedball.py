import math

from cv2 import cv2
import time
import numpy as np

position = []

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

lowerG = (40,70, 140)
upperG = (80, 255, 255)

lowerR = (0,150, 150)
upperR = (10, 255, 255)

lowerY = (20, 80, 160)
upperY = (35, 255, 255)

lowerB = (90, 160, 160)
upperB = (110, 255, 255)

bounds = [[lowerR,upperR],[lowerG,upperG],[lowerY,upperY],[lowerB,upperB]]
colors = [(0,0,255),(0,255,0),(0,255,255),(255,255,0)]
color_names = ['r','g','y','b']

prev_time = time.time()
curr_time = time.time()
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
d = 5.6 * 10 ** -2
radius = 1

guess = np.array(['r','g','y','b'])
np.random.shuffle(guess)
guess = guess.reshape((2,2))

guess = [list(g) for g in guess]

print(guess)

while cam.isOpened():
    ret, image = cam.read()
    curr_time =  time.time()
    image = cv2.flip(image, 1)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    masks = [cv2.inRange(hsv, bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    masks = [cv2.erode(mask, None, iterations=2) for mask in masks]
    masks = [cv2.dilate(mask, None, iterations=2) for mask in masks]

    ball_order = []

    for i in range(len(masks)):
        cnts = cv2.findContours(masks[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts)>0:
            c = max(cnts, key=cv2.contourArea)
            (curr_x, curr_y), radius = cv2.minEnclosingCircle(c)
            ball_order.append([curr_x,curr_y,color_names[i]])
            if radius > 10:
                cv2.circle(image, (int(curr_x), int(curr_y)), int(radius), colors[i], 2)
                cv2.circle(image, (int(curr_x), int(curr_y)), 5, colors[i], 2)


    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    time_diff = curr_time - prev_time
    pxl_per_m = d / radius
    dist = ((prev_x-curr_x)**2 + (prev_y - curr_y)**2) ** 0.5
    speed = dist/time_diff * pxl_per_m

    if len(ball_order)>0:
        ball_order = sorted(ball_order, key=lambda x: x[1])
        if len(ball_order)==4:
            ball_order = np.array(ball_order).reshape((2, 2, -1))
            ball_order = [sorted(row, key= lambda x: x[0]) for row in ball_order]
            ball_order = [[i[2] for i in row] for row in ball_order]
        else:
            ball_order = [i[2] for i in ball_order]

        found = True
        for b,g in zip(ball_order,guess):
            if b != g:
                found = False

        if found:
            cv2.putText(image, "CORRECT", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, "FALSE", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(image, str(ball_order), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Camera", image)


    prev_time = curr_time
    prev_x = curr_x
    prev_y = curr_y

cam.release()
cv2.destroyAllWindows()