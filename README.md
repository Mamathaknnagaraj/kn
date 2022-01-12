1.# program to rotate image

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

2.# Write a program to contrast stretching of a low contrast image, Histogram, and
Histogram Equalization
 
 import cv2
 
# import Numpy 

import numpy as np 

from matplotlib import pyplot as plt 

# reading an image using imreadmethod

my_img = cv2.imread('air.jpg', 0)

equ = cv2.equalizeHist(my_img)

# stacking both the images side-by-side orientation

res = np.hstack((my_img, equ))

# showing image input vs output

cv2.imshow('image', res)

cv2.waitKey(0)

cv2.destroyAllWindows()

hist,bins = np.histogram(equ.flatten(),256,[0,256])

cdf = hist.cumsum()

cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color = 'b')

plt.hist(equ.flatten(),256,[0,256], color = 'r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'upper left')

plt.show()

#output

![image](https://user-images.githubusercontent.com/96527272/149118758-2e25d482-0666-46c8-9d7b-df714526ac38.png)

# 3. Write a program simulation and display of an image, negative of an image(Binary
& Gray Scale)

import cv2

import matplotlib.pyplot as plt
  
  
# Read an image

img_bgr = cv2.imread('air.jpg', 1)

plt.imshow(img_bgr)

plt.show()

  
# Histogram plotting of the image

color = ('b', 'g', 'r')
  
for i, col in enumerate(color):
      
    histr = cv2.calcHist([img_bgr], 
    
                         [i], None,
                         
                         [256], 
                         
                         [0, 256])
      
    plt.plot(histr, color = col)
      
    # Limit X - axis to 256
    
    plt.xlim([0, 256])
      
plt.show()
  
# get height and width of the image

height, width, _ = img_bgr.shape
  
for i in range(0, height - 1):

    for j in range(0, width - 1):
          
        # Get the pixel value
        
        pixel = img_bgr[i, j]
          
        # Negate each channel by 
        
        # subtracting it from 255
          
        # 1st index contains red pixel
        
        pixel[0] = 255 - pixel[0]
          
        # 2nd index contains green pixel
        
        pixel[1] = 255 - pixel[1]
          
        # 3rd index contains blue pixel
        
        pixel[2] = 255 - pixel[2]
          
        # Store new values in the pixel
        
        img_bgr[i, j] = pixel
  
# Display the negative transformed image

plt.imshow(img_bgr)

plt.show()
  
# Histogram plotting of the

# negative transformed image

color = ('b', 'g', 'r')
  
for i, col in enumerate(color):
      
    histr = cv2.calcHist([img_bgr], 
    
                         [i], None,
                         
                         [256],
                         
                         [0, 256])
      
    plt.plot(histr, color = col)
    
    plt.xlim([0, 256])
      
plt.show()

#output
![image](https://user-images.githubusercontent.com/96527272/149121869-ebf15edf-6637-44eb-bf07-f5a28cd7f478.png)

![image](https://user-images.githubusercontent.com/96527272/149121925-569d73c4-ecf2-4845-b6db-7ede69008cf9.png)

![image](https://user-images.githubusercontent.com/96527272/149122030-20b6fdb6-5ed1-4306-ac72-cd913eb9c417.png)

![image](https://user-images.githubusercontent.com/96527272/149122063-934abecb-b84a-41b3-b85f-1c54348f48fa.png)
