# 1. program to rotate image:

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

# output:

![image](https://user-images.githubusercontent.com/96527272/148208929-6ba1c2a0-b2e9-4d08-8194-081563ec01a0.png)


![image](https://user-images.githubusercontent.com/96527272/148208770-45fca79a-3673-4403-95e8-52f7c3e14b8d.png)

# 2. Write a program to contrast stretching of a low contrast image, Histogram, and
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

# output

![image](https://user-images.githubusercontent.com/96527272/149118758-2e25d482-0666-46c8-9d7b-df714526ac38.png)

 # 3. Write a program simulation and display of an image, negative of an image(Binary
& Gray Scale):

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

# output:

![image](https://user-images.githubusercontent.com/96527272/149121869-ebf15edf-6637-44eb-bf07-f5a28cd7f478.png)

![image](https://user-images.githubusercontent.com/96527272/149121925-569d73c4-ecf2-4845-b6db-7ede69008cf9.png)

![image](https://user-images.githubusercontent.com/96527272/149122030-20b6fdb6-5ed1-4306-ac72-cd913eb9c417.png)

![image](https://user-images.githubusercontent.com/96527272/149122063-934abecb-b84a-41b3-b85f-1c54348f48fa.png)

# 4. program to perform transformation on image:

# rotation

img = cv.imread('air.jpg',0)

rows,cols = img.shape

# cols-1 and rows-1 are the coordinate limits.

M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)

dst = cv.warpAffine(img,M,(cols,rows))

#cv.imshow('img',dst)

plt.subplot(121),plt.imshow(img),plt.title('Input')

plt.subplot(122),plt.imshow(dst),plt.title('Output')

plt.show()

cv.waitKey(0)

cv.destroyAllWindows()

# output:

![image](https://user-images.githubusercontent.com/96527272/149316158-efb58808-0ea3-47fb-8b8c-629e452a9fc1.png)

![image](https://user-images.githubusercontent.com/96527272/149316270-9e7cbceb-4dde-4c2a-9cd2-93708da5caeb.png)

# scaling

img=cv.imread('air.jpg',cv.IMREAD_COLOR)

resized=cv.resize(img,None,fx=1,fy=2,interpolation=cv.INTER_CUBIC)

#cv.imshow("original pic",img)

#cv.imshow("resized pic",resized)

plt.subplot(121),plt.imshow(img),plt.title('Input')

plt.subplot(122),plt.imshow(resized),plt.title('Output')

plt.show()

cv.waitKey()

cv.destroyAllWindows()

# output:

![image](https://user-images.githubusercontent.com/96527272/149316556-6dd379e1-a75b-4c1c-8e1f-bfe5c90b6c75.png)

![image](https://user-images.githubusercontent.com/96527272/149316607-db7013fe-68b7-4e1a-bc79-f897ab8dc96f.png)

import numpy as np

import cv2 as cv

from matplotlib import pyplot as plt

# translation

img = cv.imread('air.jpg',0)

rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])

dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('img',dst)

cv.waitKey(0)

cv.destroyAllWindows()
# output:


# 5. python program to read multiple images in directory:

# import the modules

import os

from os import listdir
 
# get the path or directory

folder_dir = "C:\img"

for images in os.listdir(folder_dir):
 
    # check if the image end swith png or jpg or jpeg
    
    if (images.endswith(".png") or images.endswith(".jpg")\
    
        or images.endswith(".jpeg")):
        
        # display
        
        print(images)
        
# output:

![air](https://user-images.githubusercontent.com/96527272/149464266-4243a27e-8ec8-464c-bfcf-14f110dfe86e.jpg)
![goku](https://user-images.githubusercontent.com/96527272/149464335-1eaaf08c-16f4-41c2-b755-4756480a970e.jpg)
![nature](https://user-images.githubusercontent.com/96527272/149464373-82416a78-c822-4e5b-ab81-03e3d6d1c299.jpg)



# 6. mearging an images:

from PIL import Image
  
img_01 = Image.open("goku.jpg")

img_02 = Image.open("nature.jpg")

  
img_01_size = img_01.size

img_02_size = img_02.size


  
print('img 1 size: ', img_01_size)

print('img 2 size: ', img_02_size)

  
new_im = Image.new('RGB', (2*img_01_size[0],2*img_01_size[1]), (250,250,250))
  
new_im.paste(img_01, (0,0))

new_im.paste(img_02, (img_01_size[0],0))

  
new_im.save("merged_images.png", "PNG")

new_im.show()

output:

img 1 size:  (922, 565)
img 2 size:  (771, 480)

# 7. program to find a mean value of the image:

import cv2
  
# Save image in set directory

# Read RGB image

img = cv2.imread('nature.jpg') 

 # getting mean value
 
mean = img.mean()
  
# printing mean value

print("Mean Value for 0 channel : " + str(mean))

# Output img with window name as 'image'

cv2.imshow('image', img) 
  
# Maintain output window utill

# user presses a key

cv2.waitKey(0)        
  
# Destroying present windows on screen

cv2.destroyAllWindows() 

# output:

Mean Value for 0 channel : 51.81252612047845

# 7. python code for sudoku:
import cv2 import numpy as np

#Loading image contains lines

img = cv2.imread('sudoku.jpg')

#Convert to grayscale

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Apply Canny edge detection, return will be a binary image

edges = cv2.Canny(gray,50,100,apertureSize = 3)

#Apply Hough Line Transform, minimum lenght of line is 200 pixels

lines = cv2.HoughLines(edges,1,np.pi/180,200)

#Print and draw line on the original image

for rho,theta in lines[0]:

print(rho, theta)

a = np.cos(theta)

b = np.sin(theta)

x0 = a*rho

y0 = b*rho

x1 = int(x0 + 1000*(-b))

y1 = int(y0 + 1000*(a))

x2 = int(x0 - 1000*(-b))

y2 = int(y0 - 1000*(a))

cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

#Show the result

cv2.imshow("Line Detection", img)

cv2.waitKey(0)

cv2.destroyAllWindows()

# output:

# 8. python program to find total number of circle:

import cv2
import numpy as np

img = cv2.imread('circle.jpg')
mask = cv2.threshold(img[:, :, 0], 255, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

stats = cv2.connectedComponentsWithStats(mask, 8)[2]
label_area = stats[1:, cv2.CC_STAT_AREA]

min_area, max_area = 50, 350  # min/max for a single circle
singular_mask = (min_area < label_area) & (label_area <= max_area)
circle_area = np.mean(label_area[singular_mask])

n_circles = int(np.sum(np.round(label_area / circle_area)))

print('Total circles:', n_circles)
cv2.imshow('image', img)
# output:


