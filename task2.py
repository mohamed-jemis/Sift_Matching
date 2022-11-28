import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
Images = []
path = 'assignment data/'
Files = os.listdir(path)
for file in Files:
    img_path = os.path.join(path, file)
    # print(img_path)
    img = cv.imread(img_path)
    Images.append(img)
    print('X_data shape:', np.array(Images).shape)
print('X_data shape:', np.array(Images[0]).shape)