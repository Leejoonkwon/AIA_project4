import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('C:\\team_project\AIA_project4/haarcascade_frontalface_default.xml')
eye_casecade = cv2.CascadeClassifier('C:\\team_project\AIA_project4/haarcascade_eye.xml')

ff = np.fromfile('C:\\team_project\AIA_project4\\정우성.jpg',np.uint8)
# img = cv2.imread('C:\\team_project\AIA_project4\\test12.jpg')
img = cv2.imdecode(ff,cv2.IMREAD_UNCHANGED)
print(cv2.__version__)
print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3,5)
imgNum  = 0
for (x,y,w,h) in faces:
    cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
    # 이미지를 저장
    cv2.imwrite("thumbnail" + str(imgNum) + ".png", cropped)
    imgNum += 1

    # cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
    # roi_gray = gray[y:y+h, x:x+w]
    # roi_color = img[y:y+h, x:x+w]
    # eyes = eye_casecade.detectMultiScale(roi_gray)
    # for (ex, ey, ew, eh) in eyes:
    #     cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0,255,0),2)

# cv2.imshow('Image view', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


