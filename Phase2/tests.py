import os
import cv2
from skimage.feature import hog
from skimage.io import imread,imshow
from skimage import exposure
from skimage.transform import rescale
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
image = cv2.imread('train\\accordian\\image_0018.jpg')
plt.imshow(image)

# image_gray = rgb2gray(image)
# imshow(image_gray)
# hogfv,hog_image = hog(image_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),visualize=True, multichannel=True)
# hog_scale = exposure.rescale_intensity(hog_image,in_range=(0,5))
# imshow(hog_scale)