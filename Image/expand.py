from skimage import io,data
import matplotlib.pyplot as plt
import cv2 as cv
img=cv.imread('002.jpg')
img2=img.copy()
row=img.shape[0]
col=img.shape[1]
for i in range(1,row-1):
    for u in range(1,col-1):
        tem=float(img[i][u])*0.6+(float(img[i-1][u]+img[i+1][u]+img[i][u-1]+img[i][u+1]))*0.1
        img2[i][u]=[int(tem[0]),int(tem[1]),int(tem[2])]
plt.figure()
plt.axis('off')
plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(img2)
plt.show()
