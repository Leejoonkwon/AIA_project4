import cv2
import numpy as np
import os

path_dir = "D:\study_data\image\image/"
file_list = os.listdir(path_dir)

file_list[0]
len(file_list)
print(len(file_list))

file_name_list = []

for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".png",""))
print(file_name_list)


print(file_name_list[0])
def Cutting_face_save(image, name):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_casecade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
        cropped = img[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]
        # eyes = eye_casecade.detectMultiScale(roi_gray)    
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0,255,0),2)
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
    #     cropped = image[y: y+h, x: x+w]
    #     resize = cv2.resize(cropped, (250,250))
    #     # cv2.imshow("crop&resize", resize)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

        # 이미지 저장하기
        cv2.imwrite(f"D:\study_data\image\image/{name}.png", cropped)
        
for name in file_name_list:
    img = cv2.imread("D:\study_data\image\image/"+name+".png")
    Cutting_face_save(img, name)

