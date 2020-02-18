import cv2
import numpy as np
from matplotlib import pyplot as plt

x = 1200
y = 725

img = cv2.imread('./Images/test1.png')
img2 = cv2.imread('./Images/test1.png')
gray = cv2.imread('./Images/test1.png', 0)

corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)


#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#lower = np.array([161, 155, 84])
#upper = np.array([179, 255, 255])
#mask = cv2.inRange(hsv, lower, upper)
#res = cv2.bitwise_and(img, img, mask = mask)
#print(hsv[y, x])
plt.imshow(img2)
#plt.xticks([]), plt.yticks([])
plt.show()
#cv2.imshow('test', hsv)


k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()



