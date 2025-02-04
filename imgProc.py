import random

import cv2 as cv
import dlib
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

currPoint = None
currPointPos = None


def check_wraps(lips):

    global currPointPos

    if not lips:
        return

    # check if lips wrap around the burger
    for lip in lips:

        result = cv.pointPolygonTest(lip, currPointPos, False)

        if result >= 0:
            print("Point eaten.")
            currPointPos = None
            break

    return


# we only need lip landmarks. these are landmarks 48-68
def get_lips(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    lips_list = []

    for face in faces:
        shape = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)  # converts dlib shape object to numpy array ezpz
        lips = shape_np[48:68]
        lips_list.append(lips)

    return lips_list


def img_process(frame):

    global currPointPos

    lips = get_lips(frame)

    if currPointPos is None:
        h, w = frame.shape[:2]
        # pick a random position for the burger
        currPointPos = (random.randint(70, w-70), random.randint(70, h-70))
    else:
        check_wraps(lips)

    cv.circle(frame, currPointPos, 10, (255, 0, 0), -1)

    for lip in lips:
        for point in lip:
            cv.circle(frame, (point[0], point[1]), 1, (0, 0, 255), -1)
        cv.polylines(frame, [lip], isClosed=True, color=(0, 255, 0), thickness=1)

    return
