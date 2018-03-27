import numpy as np
import cv2

"""
load the image from the file and convert to gray scale
"""
def _loadImage(image):
    # image is already loaded in to array format
    if type(image) is np.ndarray:
        # image has 3 color channels
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return gray
        # image is has only one color channel and assume this is a in gray scale
        if image.shape[2] == 1:
            return image

    # assuming the image variable contains the path to image file
    if type(image) is str:
        return cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_RGB2GRAY)

    raise Exception('failed to load the gray scale image')


# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    # Convert to grayscale
    gray = _loadImage(img)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = _loadImage(img)
    # 2) Take the gradient in x and y separately
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    # gradmag = np.sqrt(sobelx**2 + sobely**2)
    gradmag = sobelx
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = _loadImage(img)
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction_img = np.arctan2(abs_sobel_y, abs_sobel_x)
    # 5) Create a binary mask where direction thresholds are met
    sbinary = np.zeros_like(direction_img)
    sbinary[(direction_img >= thresh[0]) & (direction_img <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    return sbinary


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    Sbinary = np.zeros_like(S)
    Sbinary[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    # binary_output = np.copy(img) # placeholder line
    return Sbinary


def applyThreshold(undistimg):
    ksize = 3
    gradx = abs_sobel_thresh(undistimg, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    # grady = abs_sobel_thresh(undistimg, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    magthresh_img = mag_thresh(undistimg, sobel_kernel=ksize, mag_thresh=(20, 100))
    directionthres_img = dir_threshold(undistimg, sobel_kernel=ksize, thresh=(0.7, 1.3))
    s_binary = hls_select(undistimg, thresh=(170, 255))
    combined = np.zeros_like(directionthres_img)
    combined[
        (gradx == 1) |
        ((magthresh_img == 1) & (directionthres_img == 1)) |
        (s_binary == 1)] = 1


    # color_binary = np.dstack(( np.zeros_like(gradx), gradx, s_binary)) * 255

    return combined
