import cv2


vidcap = cv2.VideoCapture("D:\study_data/1.mp4")
count = 0

while(vidcap.isOpened()):
    ret, image = vidcap.read()
    cv2.imwrite("D:\study_data/frame%d.jpg" % count,image)
    print("Saved fram%d.jpg" % count)
    count += 1
    
vidcap.release()
################## 1/20 프레임 단위로 저장  
'''   
while(vidcap.isOpened()):                       
    ret, image = vidcap.read()
    
    if(int(vidcap.get(1))% 20 == 0):
        cv2.imwrite("D:\study_data/frame%d.jpg" % count,image)
        print("Saved fram%d.jpg" % count)
        count += 1
'''
########################################