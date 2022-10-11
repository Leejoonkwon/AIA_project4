from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from urllib.request import urlopen
import time
import urllib.request
import os

# options = webdriver.ChromeOptions()
# options.add_argument('headless')
# options.add_argument('window-size=1920x1080')
# options.add_argument('disable-gpu')
# options.add_argument('User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36')
# options.add_argument('window-size=1920x1080')
# options.add_argument('ignore-certificate-errors')
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_experimental_option("excludeSwitches",["enable-logging"])
driver = webdriver.Chrome('C://chromedriver.exe',chrome_options=options)

if not os.path.isdir("이정재/"):
    os.makedirs("이정재/")

# driver = webdriver.Chrome('C://chromedriver.exe')
driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")

keywords = "이정재 얼굴"
elem = driver.find_element(By.XPATH,"/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input")

elem.send_keys(keywords)
elem.send_keys(Keys.RETURN)

SCROLL_PAUSE_TIME = 1

last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)
    new_height = driver.execute_script("return document.body.scrollHeight")

    if new_height == last_height:
        try:
            driver.find_element(By.CSS_SELECTOR,".mye4qd").click()
        except:
            break
    last_height = new_height

images = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd")
count = 1

for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl = driver.find_element(By.XPATH,"//*[@id='Sva75c']/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img").get_attribute('src')
        urllib.request.urlretrieve(imgUrl, "이정재/" + keywords + "_" + str(count) + ".jpg")
        print("Image saved: 이정재_{}.jpg".format(count))
        count += 1
    except:
        pass

driver.close()

import cv2
import numpy as np
import os

path_dir = "이정재/"
file_list = os.listdir(path_dir)

file_list[0]
len(file_list)
print(len(file_list))

file_name_list = []

for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg",""))
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
        cv2.imwrite(f"이정재/{name}.jpg", cropped)
                         
for name in file_name_list:
    img = cv2.imread("이정재/"+name+".jpg")
    Cutting_face_save(img, name)



