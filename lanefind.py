################################################################################
## Advanced Lane Finding Project
## Prabath Peiris
## peiris.prabath@gmail.com
################################################################################

import numpy as np
import cv2
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import _pickle as cPickle
from calibrateCam import *
from utils import *
from thresholding import applyThreshold
from perspective_transform import perspectiveTransform
from locatelanelines import locatelanes, locatelanes_slidingwindow

distortedImageLoc = 'test_images'
# Compute the camera calibration matrix and distortion coefficients

def main():
    # [x] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    mtx, dist = calibrateCamera()

    # generateCameraCalibrationImages(mtx, dist)
    # img = 'camera_cal/calibration1.jpg'
    # undistimg = applyDistortionCorrection(mtx, dist, img)
    # saveimage(undistimg, img.split('/')[-1].split('.')[0] + '_cam_cal')
    # return

    Images = glob.glob(distortedImageLoc + "/*.jpg")
    for img in Images:
        '''
            [x] Apply a distortion correction to raw color images.
        '''
        undistimg = applyDistortionCorrection(mtx, dist, img)
        saveimage(undistimg, img.split('/')[-1].split('.')[0] + '_undist')

        '''
            [x] Apply threshold to color image and get a binary image
        '''
        binaryImage = applyThreshold(undistimg)
        saveimageplt(binaryImage, img.split('/')[-1].split('.')[0] + '_combined')

        # """
        #     [x] Apply a perspective transform to rectify binary image ("birds-eye view").
        # """

        warped_img, Minv = perspectiveTransform(binaryImage, img)
        saveimageplt(warped_img, img.split('/')[-1].split('.')[0] + '_warped')
        # break

        # img1 = locatelanes(img, warped_img)
        finalimg  = locatelanes_slidingwindow(img, warped_img, undistimg, Minv, True)
        break
        saveimage(finalimg, img.split('/')[-1].split('.')[0] + '_final')

        break

    # Use color transforms, gradients, etc., to create a thresholded binary image.
    # applyThreshold(image)


if __name__ == '__main__':
    main()
