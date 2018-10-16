import numpy as np
import cv2 as cv
from skimage import io,data
import matplotlib.pyplot as plt
img=data.imread('2.jpg','gray')
rows=img.shape[0]
cols=img.shape[1]
print(rows)
print(cols)
img2=img.copy()
max_piexl=img.max()
mean_piexl=img.mean()

f=np.fft.fft2(img)
f_shift=np.fft.fftshift(f)
s2=np.log(np.abs(f_shift))
for i in range(200):
    for u in range(140,160):
        s2[u][i]=(s2[u-1][i]+s2[u+1][i])/2
for i in range(302):
    for u in range(90,110):
        s2[i][u]=(s2[i][u-1]+s2[i][u+1])/2

'''for i in range(rows):
    for u in range(cols):
        img2[i,u]=((img[i,u]/255)**0.5)*255'''
        #img2[i,u,1]=img[i,u,1]*255/mean_piexl
       # img2[i,u,2]=img[i,u,2]*255/mean_piexl
f2=np.fft.fft2(img2)
f2_shift=np.fft.fftshift(f2)
s1=np.log(np.abs(f2_shift))
plt.figure()
plt.axis('off')
plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(img2)
plt.subplot(2,2,3)
plt.imshow(s2,'gray')
plt.subplot(2,2,3)
plt.imshow(s2,'gray')


plt.show()
