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
from Lanes import Line
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


distortedImageLoc = 'test_images'
# Compute the camera calibration matrix and distortion coefficients

def processTestImages():
    # [x] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    mtx, dist = calibrateCamera()

    # generateCameraCalibrationImages(mtx, dist)
    # img = 'camera_cal/calibration1.jpg'
    # undistimg = applyDistortionCorrection(mtx, dist, img)
    # saveimage(undistimg, img.split('/')[-1].split('.')[0] + '_cam_cal')
    # return

    Images = glob.glob(distortedImageLoc + "/*.jpg")

    leftLane = Line()
    rightLane = Line()
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
        # break
        saveimage(finalimg, img.split('/')[-1].split('.')[0] + '_final')

        # break

    # Use color transforms, gradients, etc., to create a thresholded binary image.
    # applyThreshold(image)

def processImage(img, imagepath="test_images/test.jpg"):
    mtx, dist = calibrateCamera()
    # '''
    #     [x] Apply a distortion correction to raw color images.
    # '''
    undistimg = applyDistortionCorrection(mtx, dist, img)

    # saveimage(undistimg, img.split('/')[-1].split('.')[0] + '_undist')

    # '''
    #     [x] Apply threshold to color image and get a binary image
    # '''
    binaryImage = applyThreshold(undistimg)
    # saveimageplt(binaryImage, img.split('/')[-1].split('.')[0] + '_combined')

    # # """
    # #     [x] Apply a perspective transform to rectify binary image ("birds-eye view").
    # # """

    warped_img, Minv = perspectiveTransform(binaryImage, imagepath)
    # saveimageplt(warped_img, img.split('/')[-1].split('.')[0] + '_warped')
    # # break

    # img1 = locatelanes(img, warped_img)
    finalimg  = locatelanes_slidingwindow(imagepath, warped_img, undistimg, Minv, False)
    # # break
    # saveimage(finalimg, img.split('/')[-1].split('.')[0] + '_final')
    return finalimg

def processImagePipleline(image):
    try:
        # img = addLaneLines(image)
        timg = processImage(image)
        # print("good")
        return timg
    except Exception as e:
        # print("bad")
        # print(e)
        return image

def processVideo():
    # process video
    # white_output = 'data/testing/project_video_output.mp4'
    white_output = 'data/testing/challenge_video.mp4'
    clip1 = VideoFileClip("challenge_video.mp4")
    white_clip = clip1.fl_image(processImagePipleline)
    white_clip.write_videofile(white_output, audio=False)

if __name__ == '__main__':
    # processTestImages()
    processVideo()
