import numpy as np
import cv2
from matplotlib import pyplot as plt

img_idx = 0
img_idx = str(img_idx).zfill(10)

imgL = cv2.imread(f'left/{img_idx}.png',0)
imgR = cv2.imread(f'right/{img_idx}.png',0)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(imgL,imgR)
print(disparity.shape)
plt.imshow(disparity,'gray')
plt.show()