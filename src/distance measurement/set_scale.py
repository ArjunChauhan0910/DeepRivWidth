import cv2
import argparse
from time import time
import numpy as np

setboxes = []
measureboxes = []


def setScale(event, x, y, flags, params):

	if event == cv2.EVENT_LBUTTONDOWN:
		#print("Point 1: "+str(x)+","+str(y))
		sbox = np.asarray([x,y])
		setboxes.append(sbox)
		
	elif event == cv2.EVENT_LBUTTONUP:
		#print("Ending line at"+str(x)+","+str(y))
		ebox = np.asarray([x,y])
		setboxes.append(ebox)
		dist = np.linalg.norm(setboxes[0]-setboxes[1])
		global scale
		scale = dist/3000 #get scale measurement for a KM
		print("[INFO] Scale is :"+ str(scale))
		print("[INFO] Distance is :"+str(dist))
		#print("[INFO] Starting box" +str(sbox))
		#print("[INFO] Ending coord "+ str(ebox) )
		cv2.line(image, tuple(setboxes[0]), tuple(setboxes[1]), (0,255,0), 2)
		print(str(tuple(setboxes[0]))+"  "+str(tuple(setboxes[1])))

def measure(event, x, y, flags, params):
	if event == cv2.EVENT_LBUTTONDOWN:
		#print("Point 1: "+str(x)+","+str(y))
		sbox = np.asarray([x,y])
		measureboxes.append(sbox)
		
	elif event == cv2.EVENT_LBUTTONUP:
		#print("Ending line at"+str(x)+","+str(y))
		ebox = np.asarray([x,y])
		measureboxes.append(ebox)
		difference = np.linalg.norm(measureboxes[0]-measureboxes[1])
		print("[INFO DIFFERECE] "+str(difference))
		dist = difference*scale
		print("Distance is : ", dist)
		cv2.line(image, tuple(measureboxes[0]), tuple(measureboxes[1]), (0,0,255), 2)

def reset():
	global image
	global clone 
	global setboxes
	global measureboxes
	image = clone.copy()
	setboxes = []
	measureboxes = []

def new():
	global measureboxes
	global image
	measureboxes = []
	image = clone.copy()
	cv2.line(image, tuple(setboxes[0]), tuple(setboxes[1]), (0,255,0), 2)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("Measurement")

while True:
	# display the image and wait for a keypress
	cv2.imshow("Measurement", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("s"):
		cv2.setMouseCallback("Measurement", setScale)
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("m"):
		cv2.setMouseCallback("Measurement", measure)
	
	elif key == ord("r"):
		reset()
	
	elif key == ord("n"):
		new()
	
	elif key == ord("q"):
		break

cv2.destroyAllWindows()

