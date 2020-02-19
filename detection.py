import cv2
import numpy as np
from matplotlib import pyplot as plt

x = 1189
y = 726

img = cv2.imread('./Images/test3.png')
img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('./Images/test3.png')
gray = cv2.imread('./Images/test3.png', 0)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
img = cv2.GaussianBlur(img, (3, 3), 0)
#ret,img = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#lower = np.array([140, 106, 106])
#upper = np.array([167, 255, 255])
#lower = np.array([124, 83, 54])
#upper = np.array([156, 245, 251])
lower = np.array([140, 50, 100])
upper = np.array([152, 255, 120])
mask = cv2.inRange(hsv, lower, upper)
mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
mask1 = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5),np.uint8))

th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,5,2)
th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,5,2)

edges = cv2.Canny(img,150,200)

#x, y, w, h = cv2.boundingRect(mask)
#rect1 = cv2.rectangle(img.copy(),(x,y),(x+w,y+h),(0,255,0),3) # not copying here will throw an error

canny = cv2.Canny(gray, 120, 180, 1)
kernel = np.ones((1,1),np.uint8)
dilate = cv2.dilate(mask1, kernel, iterations=1)

# Find contours 
cnts = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if w * h > 500:
        cv2.rectangle(img0, (x, y), (x + w, y + h), (36,255,12), 2)

#res = cv2.bitwise_and(img, img, mask = mask)
#print(hsv[x, y])
plt.imshow(img0)
#plt.imshow(mask)
#plt.xticks([]), plt.yticks([])
plt.show()
#cv2.imshow('test', hsv)


k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()



