# program to rotate image

import cv2

import numpy as np

import matplotlib.pyplot as plt

img = cv2.imread('thumbs.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w = img.shape[:2]


T = cv2.getRotationMatrix2D((w/2, h/2), 90, .5)

img = cv2.warpAffine(img, T, (w, h))

fig, ax = plt.subplots(1, figsize=(12,200))

ax.axis('off')  

plt.imshow(img)

# output

![image](https://user-images.githubusercontent.com/96527272/148208929-6ba1c2a0-b2e9-4d08-8194-081563ec01a0.png)


![image](https://user-images.githubusercontent.com/96527272/148208770-45fca79a-3673-4403-95e8-52f7c3e14b8d.png)
