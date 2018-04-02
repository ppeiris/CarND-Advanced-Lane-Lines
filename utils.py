import numpy as np
import cv2
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

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
