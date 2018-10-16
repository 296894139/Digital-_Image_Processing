import numpy as np
import cv2 as cv
from skimage import io,data
import matplotlib.pyplot as plt
img=data.imread('2.jpg')
rows=img.shape[0]
cols=img.shape[1]
img2=img.copy()
min_cha=abs(img[0][0][0]+img[0][0][1]+img[0][0][2]-img[0][1][0]-img[0][1][1]-img[0][1][2])*2
index=0
for i in range(1,cols-1):
    tem_cha=abs(2*img[0][i][0]+2*img[0][i][1]+2*img[0][i][2]-img[0][1+i][0]-img[0][1+i][1]-img[0][1+i][2]-img[0][i-1][0]-img[0][i-1][1]-img[0][i-1][2])
    if(tem_cha<min_cha):
        min_cha=tem_cha
        index=i
img2[0][index]=[255,0,255]
for i in range(1,rows-1):
    first=index-1
    if(first>=0):
        min_first=abs(img[i-1][index][0]+img[i-1][index][1]+img[i-1][index][2]-img[i][first][0]-img[i][first][1]-img[i][first][2])
    else:
        min_first=1000000
    second=index
    min_second=abs(img[i-1][index][0]+img[i-1][index][1]+img[i-1][index][2]-img[i][second][0]-img[i][second][1]-img[i][second][2])
    third=index+1
    if(third<rows):
        min_third=abs(img[i-1][index][0]+img[i-1][index][1]+img[i-1][index][2]-img[i][third][0]-img[i][third][1]-img[i][third][2])
    else:
        min_third=1000000
    if(min_first<min_second and min_first<min_third):
        img2[i][first]=[255,0,255]
        index=first
    elif(min_third<min_first and min_third<min_second):
        img2[i][third]=[255,0,255]
        index=third
    else:
        img2[i][index]=[255,0,255]
plt.figure()
plt.axis('off')
plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(img2)


plt.show()
