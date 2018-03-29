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
from thresholding import applyThreshold
from perspective_transform import perspectiveTransform
from locatelanelines import locatelanes, locatelanes_slidingwindow

calImgLoc = 'camera_cal'
distortedImageLoc = 'test_images'
# Compute the camera calibration matrix and distortion coefficients

def saveimage(image, name='', loc="data/testing"):
    """
    Save given numpy array as an png image
    """
    iname = name + '.png'
    cv2.imwrite(loc + "/" + iname, image)
    print("[save:] %s" %(loc + "/" + iname))


def saveimageplt(image, name='', loc="data/testing"):

    iname = name + '.png'
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap = 'gray')
    fig.savefig(loc + "/" + iname, bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    print("[save:] %s" %(loc + "/" + iname))



'''load the camera matrix from the pickle file

once camera matrix has been computed the matrix has been saved for later use.
this method load the saved data
'''
def _loadCameraCalibration(mtx='mtx', dist='dist'):

    mtxfile = "data/camera_calibration/%s.pck" % (mtx)
    distfile = "data/camera_calibration/%s.pck" % (dist)
    if (os.path.isfile(mtxfile)):
        # load the file
        with open(mtxfile, 'rb') as _mtxfile:
            fmtx = cPickle.load(_mtxfile)
        with open(distfile, 'rb') as _distfile:
            fdist = cPickle.load(_distfile)
        return fmtx, fdist
    return None, None

''' Saved the camera matrix to a pickle file
Save the camera Matrix to a file
'''
def _saveCameraCalibration(mtx, dist):

    print('Camera calibration matrix has been saved to a file')

    mtxfile = "data/camera_calibration/mtx.pck"
    distfile = "data/camera_calibration/dist.pck"

    with open(mtxfile, 'wb') as f:
        p = cPickle.Pickler(f)
        p.fast = True
        p.dump(mtx)

    with open(distfile, 'wb') as f:
        p = cPickle.Pickler(f)
        p.fast = True
        p.dump(dist)


'''
Generate camera calibrate data
'''
def calibrateCamera():

    mtx, dist = _loadCameraCalibration()
    if mtx is None:
        print('Calibrating the camera')
        objpoints = []
        imgpoints = []
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        # read calibration image list
        calImages = glob.glob(calImgLoc + "/*.jpg")
        # img_size = (calImages[0].shape[1], img.shape[0])
        # Iterate image list
        img_size = False
        for calimg in calImages:
            # convert image to gray
            img = cv2.imread(calimg)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if not img_size:
                img_size = (img.shape[1], img.shape[0])
            # find chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                # saveimage(img, calimg.split('/')[1].split('.')[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                                            objpoints,
                                            imgpoints,
                                            img_size,
                                            None,
                                            None
                                        )

        _saveCameraCalibration(mtx, dist)
    else:
        print('Camera calibration matrix has been loaded from file')

    return mtx, dist

def applyDistortionCorrection(mtx, dist, imgpath):

    undistimg = cv2.undistort(cv2.imread(imgpath), mtx, dist, None, mtx)
    return undistimg

def main():
    # [x] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    mtx, dist = calibrateCamera()

    Images = glob.glob(distortedImageLoc + "/*.jpg")
    for img in Images:
        '''
            [x] Apply a distortion correction to raw color images.
        '''
        undistimg = applyDistortionCorrection(mtx, dist, img)
        # saveimage(undistimg, img.split('/')[-1].split('.')[0] + '_undist', True)

        '''
            [x] Apply threshold to color image and get a binary image
        '''
        binaryImage = applyThreshold(undistimg)
        # saveimageplt(binaryImage, img.split('/')[-1].split('.')[0] + '_binary')

        """
            [x] Apply a perspective transform to rectify binary image ("birds-eye view").
        """

        warped_img, Minv = perspectiveTransform(binaryImage, img)

        saveimageplt(warped_img, img.split('/')[-1].split('.')[0] + '_warped')

        locatelanes(img, warped_img)
        finalimg  = locatelanes_slidingwindow(img, warped_img, undistimg, Minv)
        saveimage(finalimg, img.split('/')[-1].split('.')[0] + '_final')



    # Use color transforms, gradients, etc., to create a thresholded binary image.
    # applyThreshold(image)


if __name__ == '__main__':
    main()
