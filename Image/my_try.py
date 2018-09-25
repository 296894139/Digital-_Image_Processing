import numpy as np
import cv2
from skimage import io,data
import matplotlib.pyplot as plt
img=data.imread('timg.jpg')
rows=img.shape[0]
cols=img.shape[1]
print(rows)
print(cols)
img2=img.copy()
max_piexl=img.max()
mean_piexl=img.mean()
f=np.fft.fft(img)
f_shift=np.fft.fftshift(f)
plt.plot(f_shift)
plt.show()
for i in range(rows):
    for u in range(cols):
        img2[i,u]=((img[i,u]/255)**0.5)*255
        #img2[i,u,1]=img[i,u,1]*255/mean_piexl
       # img2[i,u,2]=img[i,u,2]*255/mean_piexl

plt.figure()
plt.axis('off')
plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(img2)
plt.subplot(2,2,3)
plt.imshow(f)

plt.show()
