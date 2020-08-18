import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from matplotlib import pyplot as plt

def imgr(img1, ip=True):   
    img = cv2.imread(img1)
    img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img,(512,512,1))
    img = np.float16(img)
    img = np.expand_dims(img, axis=0)
    if ip:
        img = model.predict(img)
        img = np.squeeze(img, axis=0)
        img = img*255
    img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
    img = np.reshape(img,(256,256,1))
    return img

def calculateClass(img,n_labels):
    h=256
    w=256
    labels_c=np.zeros([h,w])
    for i in range (h):
        for j in range (w):
            l=img[i][j]
            if l==0 :#land
                labels_c[i][j]=0
            elif l==255:#water
                labels_c[i][j]=1
            
                        
    return labels_c*255
                

def catelab(img,  n_labels):
    dims=[256,256]
    labels_c=calculateClass(img,n_labels)
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            v=int(labels_c[i][j])
            x[i][j][v]=1
    x = x.reshape(dims[0] , dims[1], n_labels)
    if n_labels == 2:
        y = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
        x = np.dstack((x,y))
    return x


