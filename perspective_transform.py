import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = None
Minv = None

def saveimageplt(image, pts, name='', loc="data/testing"):

    iname = name + '.png'
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap = 'gray')

    ax.plot(pts[0][0], pts[0][1], '.')
    ax.plot(pts[1][0], pts[1][1], '.')
    ax.plot(pts[2][0], pts[2][1], '.')
    ax.plot(pts[3][0], pts[3][1], '.')

    fig.savefig(loc + "/" + iname, bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    print("[save:] %s" %(loc + "/" + iname))


"""
Compute the transformation and inverse transformation matrix
"""
def _computeM(img, name):

    # print("Computing Transformation Matrix")

    global M
    global Minv

    srcpts = np.float32([
        [100, img.shape[0]-5],
        [550, 450],
        [750, 450],
        [1250, img.shape[0]-5]]
    )

    destpts = np.float32([
                    [100, img.shape[0]-5],
                    [150, 50],
                    [1100, 50],
                    [1100, img.shape[0]-5]
                ])

    # img = cv2.line(img, (srcpts[0][0], srcpts[0][1]), (srcpts[1][0], srcpts[1][1]), (0,255,0), 2)
    # cv2.line(img,(srcpts[0][0], srcpts[0][1]),(srcpts[1][0], srcpts[1][1]),(0,0,255),1)
    # saveimageplt(img, srcpts, name.split('/')[-1].split('.')[0] + '_point')

    M = cv2.getPerspectiveTransform(srcpts, destpts)
    Minv = cv2.getPerspectiveTransform(destpts, srcpts)


def perspectiveTransform(img, name):

    global M
    global Minv

    if M is None:
        _computeM(img, name)

    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, Minv
