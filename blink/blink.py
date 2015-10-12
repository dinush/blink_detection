import numpy as np
import cv2
import time

DOWNSCALE = 1
facecascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')
eyecascade = cv2.CascadeClassifier('haar/eye_tree.xml')

cam = cv2.VideoCapture(0)
cv2.namedWindow("preview")

if cam.isOpened(): # try to get the first frame
    rval, frame = cam.read()
else:
    rval = False

def minute_passed(oldepoch):
    return time.time() - oldepoch >= 60

last_state = True 
blink_count = 0
epoch = time.time()

while rval:
    ret, img = cam.read()

    minisize = (img.shape[1]/DOWNSCALE,img.shape[0]/DOWNSCALE)
    img = cv2.resize(img, minisize)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = facecascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face:
	img = img[y:y+h, x:x+w]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if (len(face) > 0):
        eyes = eyecascade.detectMultiScale(gray2, 1.1, 5)

        print("no of eyes ", len(eyes))
	if len(eyes) > 0 and last_state == False:
	    print("blink")
	    blink_count += 1
	last_state = len(eyes) > 0
        for (x,y,w,h) in eyes:
            #img = img[x:x+w, y:y+h]
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    else:
	last_state = True

    cv2.imshow('img',img)
    cv2.waitKey(1)

    if minute_passed(epoch):
	epoch = time.time()
	f = open('blink_log.txt', 'a')
	localtime = time.asctime(time.localtime(time.time()))
	f.write(localtime)
	f.write(" -> ")
	f.write(str(blink_count))
	f.write('\n')
	f.close()
	blink_count = 0



