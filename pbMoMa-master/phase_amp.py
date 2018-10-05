import numpy as np
from pyPyrTools import SCFpyr
from skimage import color
from skimage import transform
from perceptual.filterbank import *

import numpy as np
import cv2
import colorsys
import matplotlib.pyplot as plt
import sys
from scipy import signal
import pyPyrTools as ppt
from pyPyrTools import JBhelpers as jbh
from scipy.signal import butter, lfilter, iirfilter
import sys
from frame_interp import *
try:
    import cv2.cv as cv
    USE_CV2 = True
except ImportError:
    # OpenCV 3.x does not have cv2.cv submodule
    USE_CV2 = False

def yiq_to_rgb_new(y, i, q):
    r = y + 0.948262*i + 0.624013*q
    g = y - 0.276066*i - 0.639810*q
    b = y - 1.105450*i + 1.729860*q
    return (r, g, b)

def butter_bandpass(lowcut, highcut, fs, order=9):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=9):
    # b, a = butter(order, [lowcut, highcut], btype='band')
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq):
    print "Reading:", vidFname,

    # get vid properties
    vidReader = cv2.VideoCapture(vidFname)
    if USE_CV2:
        # OpenCV 2.x interface
        vidFrames = int(vidReader.get(cv.CV_CAP_PROP_FRAME_COUNT))    
        width = int(vidReader.get(cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(vidReader.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        fps = int(vidReader.get(cv.CV_CAP_PROP_FPS))
        func_fourcc = cv.CV_FOURCC
    else:
        # OpenCV 3.x interface
        vidFrames = int(vidReader.get(cv2.CAP_PROP_FRAME_COUNT))    
        width = int(vidReader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vidReader.get(cv2.CAP_PROP_FPS))
        func_fourcc = cv2.VideoWriter_fourcc

    if np.isnan(fps):
        fps = 30

    print ' %d frames' % vidFrames,
    print ' (%d x %d)' % (width, height),
    print ' FPS:%d' % fps

    # video Writer
    fourcc = func_fourcc('M', 'J', 'P', 'G')
    vidWriter = cv2.VideoWriter(vidFnameOut, fourcc, int(fps), (width,height), 1)
    print 'Writing:', vidFnameOut

    # how many frames
    nrFrames = min(vidFrames, maxFrames) - 250  # less frame
    print 'FrameNr:', 
    for frameNr in range( nrFrames + windowSize ):
        print frameNr,
        sys.stdout.flush() 

        if frameNr < nrFrames:
            # read frame
            _, im = vidReader.read()
            yiqIm = np.empty((im.shape[0], im.shape[1], 3))

            [yiqIm[:, :, 0], yiqIm[:, :, 1], yiqIm[:, :, 2]] = colorsys.rgb_to_yiq(im[:, :, 0], im[:, :, 1], im[:, :, 2])
            #im = yiqIm[:, :, 0]
            plt.imshow(yiqIm)
            if frameNr==1:
                plt.show()
            L, my_pyr= decompose(im)
            nL = L
            print '*****complete decomposition******'
            #print len(L['phase'][0])

           # channel=0
            scale = 0.5
            limit = 4.0
            phase_diff = compute_phase_difference(L,L,scale, limit)
            print phase_diff
            


            out_img = out_img.reshape(im.shape)


            # reconstruct pyramid
            out = out_img  

            # clip values out of range
            
            out[out>255] = 255
            out[out<0] = 0
            # rgbIm[:, :, i] = out

            # make a RGB image
            #rgbIm = np.empty((im.shape[0], im.shape[1], 3))
            # rgbIm[:,:,0] = out
            # rgbIm[:,:,1] = out
            # rgbIm[:,:,2] = out

            #[rgbIm[:, :, 0], rgbIm[:, :, 1], rgbIm[:, :, 2]] = yiq_to_rgb_new(out, yiqIm[:, :, 1], yiqIm[:, :, 2])
            #rgbIm[rgbIm > 255] = 255
            #rgbIm[rgbIm < 0] = 0
            #write to disk
            res = cv2.convertScaleAbs(out)
            vidWriter.write(res)
    vidReader.release()
    vidWriter.release()

vidFname = './media/guitar.mp4'

# maximum nr of frames to process
maxFrames = 60000
# the size of the sliding window
windowSize = 30
# the magnifaction factor
factor = 20
# the fps used for the bandpass
fpsForBandPass = 600 # use -1 for input video fps
# low ideal filter
lowFreq = 72
# high ideal filter
highFreq = 92
# output video filename
vidFnameOut = vidFname + '-Mag%dIdeal-lo%d-hi%d.avi' % (factor, lowFreq, highFreq)


phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq)
