import math

from cv2 import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    angles = np.array([line[0][1] for line in lines])
    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                    for angle in angles], dtype=np.float32)

    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def segmented_intersections(lines):
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


# paper_image = cv2.imread('img/gun.jpg')
# rows, cols, _ = paper_image.shape
# pts1 = np.float32([[0,0], [0, rows], [cols, 0], [cols, rows]])

paper_text = 'Hello world!!!'
rows, cols = (480, 640)
pts1 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_EXPOSURE, 0)

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Cnts", cv2.WINDOW_KEEPRATIO)

# out = cv2.VideoWriter('outpy3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))

while cam.isOpened():
    _, image = cam.read()
    image = cv2.flip(image, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    cntsImg = cv2.Canny(gray, 100, 200, None, 3)
    kernel = np.ones((5, 5))
    cntsImg = cv2.dilate(cntsImg, kernel, iterations=2)
    cntsImg = cv2.erode(cntsImg, kernel, iterations=1)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    cnts, _ = cv2.findContours(cntsImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find all the lines
    lines = cv2.HoughLines(cntsImg, 1, np.pi / 180, 120, None, 0, 0)
    # array for rect corners
    coords = []

    if lines is not None:
        # draw all the lines
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        # segment the lines into parallels and find intersections
        segmented = segment_by_angle_kmeans(lines)
        intersections = segmented_intersections(segmented)
        intersections = np.array(intersections)

        if len(intersections) > 3:
            # group intersections into similar groups
            model = DBSCAN(eps=15, min_samples=5)
            yhat = model.fit_predict(intersections)
            clusters = np.unique(yhat)
            # find mean intersection in each group and draw it
            if len(clusters) > 0:
                for cluster in clusters:
                    row_ix = np.where(yhat == cluster)
                    coords.append(list(intersections[row_ix].mean(axis=0).astype(int)))
                    cv2.circle(image, coords[-1], 7, (0, 255, 255), 3)
            # if 4 intersections found put text there
            if len(coords) == 4:
                coords = sorted(coords, key=lambda x: x[0])
                coords = sorted(coords[:2], key=lambda x: x[1]) + sorted(coords[2:], key=lambda x: x[1])

                matrix = cv2.getPerspectiveTransform(np.array(coords, dtype='float32'), pts1)
                aff_img = cv2.warpPerspective(image, matrix, (cols, rows))

                font = cv2.FONT_HERSHEY_SIMPLEX

                textsize = cv2.getTextSize(paper_text, font, 3, 5)[0]

                textX = (aff_img.shape[1] - textsize[0]) // 2
                textY = (aff_img.shape[0] + textsize[1]) // 2

                cv2.putText(aff_img, paper_text, (textX, textY), font, 3, (0, 255, 0), 5)

                matrix = cv2.getPerspectiveTransform(pts1, np.array(coords, dtype='float32'))
                aff_img = cv2.warpPerspective(aff_img, matrix, (cols, rows))

                gray = cv2.cvtColor(aff_img, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

                roi = image[:aff_img.shape[0], :aff_img.shape[1]]

                mask_inv = cv2.bitwise_not(mask)
                bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                p = cv2.add(aff_img, bg)
                image[:p.shape[0], :p.shape[1]] = p

                # # code for putting an image onto paper
                # coords = sorted(coords,key=lambda x: x[0])
                # coords = sorted(coords[:2], key=lambda x: x[1]) + sorted(coords[2:], key=lambda x: x[1])
                #
                # matrix = cv2.getPerspectiveTransform(pts1, np.array(coords,dtype='float32'))
                # aff_img = cv2.warpPerspective(paper_image, matrix, (640, 480))
                # gray = cv2.cvtColor(aff_img, cv2.COLOR_BGR2GRAY)
                # ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
                #
                # roi = image[:aff_img.shape[0], :aff_img.shape[1]]
                #
                # mask_inv = cv2.bitwise_not(mask)
                # bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                # p = cv2.add(aff_img, bg)
                # image[:p.shape[0], :p.shape[1]] = p

    cv2.imshow('Camera', image)
    cv2.imshow('Cnts', cntsImg)
    # out.write(image)

cam.release()
cv2.destroyAllWindows()
