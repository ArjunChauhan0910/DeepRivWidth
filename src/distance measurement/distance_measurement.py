import cv2
import numpy as np 
import os
import imutils
from matplotlib import pyplot as plt
from skimage.morphology import medial_axis, skeletonize
setboxes = []
global measureboxes
measureboxes =[]

def crop_img(event, x, y, flags, params):
	if event == cv2.EVENT_LBUTTONDOWN:
		#print("Point 1: "+str(x)+","+str(y))
		sbox = np.asarray([x,y])
		measureboxes.append(sbox)
		
	elif event == cv2.EVENT_LBUTTONUP:
		#print("Ending line at"+str(x)+","+str(y))
		ebox = np.asarray([x,y])
		measureboxes.append(ebox)
		print(measureboxes)
		clone = image.copy()
		cv2.rectangle(image, tuple(measureboxes[0]), tuple(measureboxes[1]), (0,0,255), 1)
		crop = clone[measureboxes[0][1]:measureboxes[1][1],measureboxes[0][0]:measureboxes[1][0]]
		cv2.imshow("Crop",crop)
		dist = distance_measure(crop)
		print("{INFO} Distance is :" + str(dist)+"Kms")

def distance_measure(img):
	gray = cv2.bilateralFilter(img,11,17,17)
	ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	dist = cv2.distanceTransform(th, cv2.DIST_L2, 3)
	#cv2.imshow("distance_matrix",dist)
		
	a = np.max(dist)
	############################################################################################################
	#################################SET SCALE HERE ############################################################
	scale = 0.017666666666666667
	############################################################################################################
	############################################################################################################
	distance = a*scale

	return distance 

image = cv2.imread('/home/arjun/sat_research/testbench_final/distance_verification/_7.png',0)
clone = image.copy()
cv2.namedWindow("image")

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
		#global measureboxes
		measureboxes = []
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		cv2.setMouseCallback("image", crop_img)
	elif key == ord("e"):
		break
# if there are two reference points, then crop the region of interest
# from teh image and display it
'''if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)'''
# close all open windows
cv2.destroyAllWindows()


#use this pipeline for distance measurement
'''img = cv2.imread('/home/arjun/sat_research/testbench_final/distance_verification/crop_9.png',0)
gray = cv2.bilateralFilter(img,11,17,17)
plt.imshow(gray)
ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.imshow(th)
dist = cv2.distanceTransform(th, cv2.DIST_L2, 3)
a = np.where(dist > 0)
average = np.average(a)
plt.imshow(dist)
print(dist)
print(np.max(dist)*scale)
print(average*scale)'''


#Failed pipeline
'''img = cv2.imread('/home/arjun/sat_research/testbench_final/distance_verification/crop_9.png',0)
gray = cv2.bilateralFilter(img,11,17,17)
plt.imshow(gray)
ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.imshow(th)

cnts = cv2.findContours(th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
final = np.zeros([img.shape[0], img.shape[1],3],dtype=np.uint8)
final = cv2.fillPoly(final, cnts, (0,255,0))
plt.imshow(final)
#final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)#hape[0], gt_img.shape[1],3],dtype=np.uint8)
#final = cv2.fillPoly(final, cnts, (0,255,0))
final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
skel, distance = medial_axis(final, return_distance=True)
skel = np.array(skel, dtype=np.uint8)
dist = cv2.distanceTransform(255 - (255*skel), cv2.DIST_L2, 3)
distance_on_skel = distance*skel
print(distance)
a = np.where(distance_on_skel > 0)
points = np.asarray(distance_on_skel[a])
average = np.average(points)
plt.imshow(skel)

scale = 0.017666666666666667
print(average*scale)
print(distance)
'''
