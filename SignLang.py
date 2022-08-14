import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawingUtils = mp.solutions.drawing_utils
cam = cv2.VideoCapture(0)
fingertips = [8, 12, 16, 20]
thumbtip = 4
while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    height, width, channels = img.shape
    results = hands.process(img)
    if (results.multi_hand_landmarks):
        landmarks = results.multi_hand_landmarks
        for eachHand in landmarks:
            lmList = []
            for handmark in eachHand.landmark:
                lmList.append(handmark)
        fingerFoldStatus = []
        for ft in fingertips:
            x, y = int(lmList[ft].x*width), int(lmList[ft].y*height)
            cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)
            if (lmList[ft].x < lmList[ft-3].x):
                fingerFoldStatus.append(True)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)
            else:
                fingerFoldStatus.append(False)
        if (all(fingerFoldStatus)):
            if (lmList[thumbtip].y < lmList[thumbtip-1].y < lmList[thumbtip-2].y):
                cv2.putText(img, 'like', (100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif (lmList[thumbtip].y > lmList[thumbtip-1].y > lmList[thumbtip-2].y):
                cv2.putText(img, 'dislike', (100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        mp_drawingUtils.draw_landmarks(img, eachHand, mp_hands.HAND_CONNECTIONS, mp_drawingUtils.DrawingSpec(
            (0, 0, 255), 2, 2), mp_drawingUtils.DrawingSpec((0, 255, 0), 4, 2))
    cv2.imshow('signLanguage',img)
    cv2.waitKey(1)

