import numpy as np
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = 0
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        self.curverad = None
        self.fits = None

    def add(self):
        self.bestx +=1

    def getLastCurve(self):
        return self.curverad

    def setLastCurve(self, c):
        self.curverad = c

    def getfit(self):
        return self.fits

    def setfit(self, fit):
        self.fits = fit


leftLane = None
rightLane = None
def initLanes():
    global leftLane
    global rightLane
    leftLane = Line()
    rightLane = Line()


def getLanes():
    global leftLane
    global rightLane
    return leftLane, rightLane

