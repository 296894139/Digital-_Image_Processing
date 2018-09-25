import numpy as np
import cv2 as cv
import math

from Image.imconv import imconv
from skimage import io,data
from matplotlib import pyplot as plt
img = cv.imread('232.jpg', 0)
kernel_3x3 = np.array([[0.0924, 0.1192, 0.0924],
                       [0.1192, 0.1538, 0.1192],
                       [0.0924, 0.1192, 0.0924]])


#img2=img.copy()
#img3=img.copy()
img2=img.copy();
img2=imconv(img,kernel_3x3)

dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
rows, cols = img.shape
crow,ccol = rows/2 , cols/2
#for i in range(1,rows-1):
 #   for u in range(1,cols-1):
  #      img2[i][u]=0.0924*(img[i-1][u-1]+img[i+1][u-1]+img[i-1][u+1]+img[i+1][u+1])+0.1192*(img[i+1][u]+img[i-1][u]+img[i][u+1]+img[i-1][u])+0.1538*img[i][u]

# create a mask first, center square is 1, remaining all zeros
#mask = np.zeros((rows,cols,2),np.uint8)
#mask[crow-30:crow+30, ccol-30:ccol+30] = 1
# apply mask and inverse DFT
fshift = dft_shift
for i in range(dft_shift.shape[0]):
    for u in range(dft_shift.shape[1]):
        fshift[i][u]=dft_shift[i][u]/(1+10000/(i**2+u**2+0.01))
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img2,cmap='gray')
plt.title('juanji 3'),plt.xticks([]),plt.yticks([])

plt.show()

