import cv2
import numpy as np
img = cv2.imread("rosbags/145.png")
left = 424
right = 440
top = 390
bot  = 403
print "Actual light is red"
img = img[top:bot, left:right]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print hsv
cv2.imshow('Img',img)
cv2.waitKey(0)

print "Actual light is yellow"
img = cv2.imread("rosbags/126.png")
left = 542
right = 554
top = 438
bot = 446
img = img[top:bot, left:right]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print hsv
cv2.imshow('Img',img)
cv2.waitKey(0)

print "Actual light is GREEN"
img = cv2.imread("rosbags/299.png")
left = 608
right = 620
top = 470
bot = 483
img = img[top:bot, left:right]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print hsv
cv2.imshow('Img',img)
cv2.waitKey(0)

cv2.destroyAllWindows()
