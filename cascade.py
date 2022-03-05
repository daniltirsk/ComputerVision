from cv2 import cv2
import numpy as np

def put_glasses(img, face_classifier, eye_classifier, glasses_img, scaleFactor=None, minNeighbors=None):
    result = img.copy()
    rects = face_classifier.detectMultiScale(result, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    rects = sorted(rects, reverse=True, key=lambda x: x[2])
    rects = rects[:1]

    faces = []

    for (x, y, w, h) in rects:
        face = np.zeros_like(result)
        face[y:y + h, x:x + h] = result[y:y + h, x:x + h]
        faces.append(face)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255))

    for face in faces:
        rects = eye_classifier.detectMultiScale(face, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        if len(rects)>1:
            rects = sorted(rects, reverse=True, key=lambda x: x[2])
            rects = rects[:2]
            for (x, y, w, h) in rects:
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255))

            rects = sorted(rects, reverse=True, key=lambda x: x[0])

            right = rects[0][0] + rects[0][2]*2
            left = rects[1][0] - rects[1][2]

            scaling = (right-left)/glasses_img.shape[1]
            glasses_img_resized = cv2.resize(glasses_img.copy(), (right-left, int(glasses_img.shape[0] * scaling)))

            y1 = rects[0][1] - int(h / 1.5)
            y2 = rects[1][1] - int(h / 1.5)

            y = min(y1, y2)

            result[y:y+glasses_img_resized.shape[0],left:right] = result[y:y+glasses_img_resized.shape[0],left:right] * \
                                                                             (1 - glasses_img_resized[:, :, 3:] / 255) + \
                                                                             glasses_img_resized[:, :, :3] * (glasses_img_resized[:, :, 3:] / 255)



    return result

glasses = cv2.imread('img/dealwithit.png',-1)

face_cascade = 'haarcascades/haarcascade_frontalface_default.xml'
eye_cascade = 'haarcascades/haarcascade_eye.xml'

eye_classifier = cv2.CascadeClassifier(eye_cascade)
face_classifier = cv2.CascadeClassifier(face_cascade)

cam = cv2.VideoCapture(0)

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

while cam.isOpened():
    ret, frame = cam.read()

    result = put_glasses(frame, face_classifier, eye_classifier, glasses, 1.2, 5)

    cv2.imshow('Camera', result)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()