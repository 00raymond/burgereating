import cv2 as cv
import dlib
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

currBurgerPos = (0, 0)
currBurger = None


def check_wraps_burger(lips):
    # check if lips wrap around the burger
    return


def generate_burger(frame):
    # spawn a random point on the screen.
    # this point will be the center of the burger.
    # put the burger.png image on the screen at that point.
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

    lips = get_lips(frame)

    if currBurger is None:
        generate_burger(frame)
    else:
        check_wraps_burger(lips)

    for lip in lips:
        for point in lip:
            cv.circle(frame, (point[0], point[1]), 1, (0, 0, 255), -1)
        cv.polylines(frame, [lip], isClosed=True, color=(0, 255, 0), thickness=1)

    return
