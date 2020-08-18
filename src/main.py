import numpy as np
import os 
import cv2
from backend import Deeplabv3,imgr_deeplab, imgr_unet, sliding_window
import tensorflow as tf
from PIL import Image

src = os.getcwd()
src, _ = os.path.split(src)
ip_main = src+"/input_imgs/"
#gt_path = "/home/arjun/sat_research/testbench_final/input_original/19-10/"


#Reading images and creating sub-images
file = os.listdir(ip_main)
for img in file:
    name = img
    img = cv2.imread(ip_main+img,0)

height = img.shape[0]
width = img.shape[1]

height_factor = height//256
height_factor = height_factor+1
width_factor = width//256
width_factor = width_factor+1

pad_width = 256*width_factor-width
pad_height = 256*height_factor - height

img = np.pad(img,((0,pad_height),(0,pad_width)),'constant',constant_values=(0,0))

#save segments
save_path = src+"/sub_input_imgs/"
winW, winH = (256,256)
i =0
for (x,y, window) in sliding_window(img, stepSize=256, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        
        #filename = saving_path+"/gt/1_"+str(i)+"_s.png"
        filename = save_path+str(i)+(".png") #for making distance measurement data
        print("SEGMENT FORMED " + filename)
        frame = window
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(frame)
        result.save(filename)
        i += 1


#Predicting
print("Choose Model: 1) DeepLabv3 2) UNet")
choice = int(input())
# print(choice)
ip_path = src+"/sub_input_imgs/"

if choice==1:   
    img_sz = (256, 256, 3)
    model = Deeplabv3(classes = 2, input_shape=img_sz)
    checkpoint_path = src+"/weights/DeepLab_Adam_1e-4/cp.ckpt"
    model.load_weights(checkpoint_path)
    print("[INFO] DeepLab weights loaded")

    save_path = src+"/sub_output_imgs/deeplab/"
    name = name+"_deeplab.png"
    read_path = src+"/sub_output_imgs/deeplab/"
    ip_files = os.listdir(ip_path)

    for files in ip_files:
        pred = imgr_deeplab(ip_path+files,model,1)
        #CROP FINAL IMAGE OF THE PADDED REGION
        cv2.imwrite(save_path+files, pred)
        print("Written Deeplab predicted file -> "+save_path+files)

if choice ==2:
    model = tf.keras.models.load_model(src+"/weights/unet/unet_baseline_512px_500steps_10epochs.h5")
    print("[INFO] UNet weights loaded")

    save_path = src+"/sub_output_imgs/unet/"
    name = name+"_unet.png"
    read_path = src+"/sub_output_imgs/unet/"
    ip_files = os.listdir(ip_path)

    for files in ip_files:
        pred = imgr_unet(ip_path+files,model, 1)
        #CROP FINAL IMAGE OF THE PADDED REGION
        cv2.imwrite(save_path+files, pred)
        print("Written UNet Predicted file -> "+save_path+files)

#Combine Images    
file = []
file = os.listdir(read_path)
file.sort()
file
final = np.zeros(( 256*height_factor,256*width_factor,3),dtype = np.uint8)

file_int= []
for files in file:
    file_int.append(int(files.strip(".png")))
    file_int.sort()

img = []
for img_read in file_int:
    #print(path+folder+str(name)+".png")
    img1 = cv2.imread(read_path+str(img_read)+".png")
    img.append(img1)

i = 0
for y in range(0, final.shape[0], 256):
    for x in range(0, final.shape[1], 256):
        final[y:y + 256, x:x + 256] = img[i]
        i+=1

save_path = src+"/output_imgs/"
cv2.imwrite(save_path+name, final[0:height,0:width])
print("IMAGE WRITTEN -> "+save_path+name)