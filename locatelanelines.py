import numpy as np
import cv2
import matplotlib.pyplot as plt
from Lanes import *
# Read in a thresholded image
# warped = mpimg.imread('warped_example.jpg')
# window settings
window_width = 50
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):

    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids


def locatelanes(name, warped):

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels

        warpage = warpage.astype(np.uint8)
        template = template.astype(np.uint8)
        output = cv2.addWeighted(warpage, 1.0, template, 0.5, 0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    # Display the final results
    loc="data/testing"
    plt.title('window fitting results')
    plt.show()
    name = name.split('/')[-1].split('.')[0] + '_lanes'
    iname = name + '.png'
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(output)
    fig.savefig(loc + "/" + iname, bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    print("[save:] %s" %(loc + "/" + iname))


    return output

def locatelanes_slidingwindow(name, binary_warped, undist, Minv, fsave=False):

    leftl, rightl = getLanes()
    loc="data/testing"
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    histname = name.split('/')[-1].split('.')[0] + '_hist'
    np.save(loc + "/" + histname, histogram)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) *255
    # Save the image
    if fsave:
        tmpname = name.split('/')[-1].split('.')[0] + '_StackImage'
        iname = tmpname + '.png'
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.imshow(out_img)
        # ax.plot(left_fitx, ploty, color='red')
        # ax.plot(right_fitx, ploty, color='red')
        fig.savefig(loc + "/" + iname, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        print("[save:] %s" %(loc + "/" + iname))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # print(len(ploty), len(leftx))
    # print("--------------------")

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])


    # compute the different between left and right curverad
    diff_curverad = np.abs(left_curverad - right_curverad)

    # # print(diff_curverad)
    # if diff_curverad > 30000:
    #     left_curverad = leftl.getLastCurve()
    #     right_curverad = rightl.getLastCurve()
    #     left_fitx = leftl.getfit()
    #     right_fitx = rightl.getfit()
    #     print("Restore previous fits")
    # else:
    #     leftl.setLastCurve(left_curverad)
    #     rightl.getLastCurve(right_curverad)
    #     leftl.setfit(left_fitx)
    #     rightl.setfit(right_fitx)


    # Save the image
    if fsave:
        tmpname = name.split('/')[-1].split('.')[0] + '_lanes_sliding_window_1'
        iname = tmpname + '.png'
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.imshow(out_img)
        ax.plot(left_fitx, ploty, color='red')
        ax.plot(right_fitx, ploty, color='red')
        fig.savefig(loc + "/" + iname, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        print("[save:] %s" %(loc + "/" + iname))

    # Save the image
    if fsave:
        tmpname = name.split('/')[-1].split('.')[0] + '_lanes_sliding_window_2'
        iname = tmpname + '.png'
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.imshow(out_img)
        ax.plot(left_fitx, ploty, color='red')
        ax.plot(right_fitx, ploty, color='red')
        fig.savefig(loc + "/" + iname, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        print("[save:] %s" %(loc + "/" + iname))

    # print(left_curverad, right_curverad)
    # print("-----------")
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Compute the offset
    h = binary_warped.shape[0]
    r_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
    l_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
    lane_ceter_position = (r_fit_x_int + l_fit_x_int) /2
    car_position = binary_warped.shape[1]/2.0
    center_offset = (car_position - lane_ceter_position) * xm_per_pix

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    ltxt = (left_curverad+right_curverad)/2.0


    # cv2.line(undisttmp,(srcpts[0][0], srcpts[0][1]),(srcpts[1][0], srcpts[1][1]),(0,0,255),1)
    # saveimageplt(undisttmp, srcpts, name.split('/')[-1].split('.')[0] + '_linewidth')


    try:
        tmpmid = int(len(left_fitx)/2)
        tmp_pt1_x = int(left_fitx[tmpmid])
        tmp_pt1_y = int(ploty[tmpmid])

        tmp_pt2_x = int(right_fitx[tmpmid])
        tmp_pt2_y = int(ploty[tmpmid])

        current_width = (right_fitx[tmpmid] - left_fitx[tmpmid])
        prev_width = leftl.getPrevWidth()

        widthRatio = 1.0
        restored = False
        if prev_width:
            # compute the ratio
            widthRatio = prev_width/current_width

        widthRatiodisplay = "Width ratio %s" %(widthRatio)
        ltxtdisplay = "Radius of Curvature:%sm" %(round((left_curverad+right_curverad)/2.0, 2))

        if (widthRatio < 0.9) or (widthRatio > 1.08):
            # use the previous detection
            left_curverad = leftl.getLastCurve()
            right_curverad = rightl.getLastCurve()
            left_fitx = leftl.getfit()
            right_fitx = rightl.getfit()
            widthRatiodisplay = "%s (restore from previous)" %(widthRatiodisplay)
            ltxtdisplay = "%s (restore from previous)" %(ltxtdisplay)
            restored = True
        else:
            if (ltxt < 1250.0) or (ltxt > 8000.0):
                left_curverad = leftl.getLastCurve()
                right_curverad = rightl.getLastCurve()
                left_fitx = leftl.getfit()
                right_fitx = rightl.getfit()
                ltxtdisplay = "%s (restore from previous)" %(ltxtdisplay)
                # print("Restore previous fits: %s" %(ltxt))
            else:
                leftl.setPrevWidth(current_width)
                leftl.setLastCurve(left_curverad)
                rightl.setLastCurve(right_curverad)
                leftl.setfit(left_fitx)
                rightl.setfit(right_fitx)
    except Exception as e:
        raise e





    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    centertxt = "Vehicle is %sm left of center" %(round(center_offset, 4))
    diff_curverad = "Curve diff %s" %(diff_curverad)
    current_width = "Lane width %s" %(current_width)

    cv2.putText(result,ltxtdisplay,(20,100), font, 1,(255,0,0),1,cv2.LINE_AA)
    cv2.putText(result,centertxt,(20,150), font, 1,(255,0,0),1,cv2.LINE_AA)
    # cv2.putText(result,diff_curverad,(20,250), font, 1,(255,0,0),1,cv2.LINE_AA)
    # cv2.putText(result,current_width,(20,300), font, 1,(255,0,0),1,cv2.LINE_AA)
    # cv2.putText(result,widthRatiodisplay,(20,350), font, 1,(255,0,0),1,cv2.LINE_AA)
    # cv2.line(result, (tmp_pt1_x, tmp_pt1_y), (tmp_pt2_x, tmp_pt2_y), (0,255,0), 2)

    return result
