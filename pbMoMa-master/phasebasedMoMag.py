from perceptual.filterbank import *

import cv2
import colorsys
from scipy.signal import butter, lfilter, iirfilter
import matplotlib.pyplot as plt
# determine what OpenCV version we are using
try:
    import cv2.cv as cv
    USE_CV2 = True
except ImportError:
    # OpenCV 3.x does not have cv2.cv submodule
    USE_CV2 = False
    
import sys
import numpy as np

from pyr2arr import Pyramid2arr
from temporal_filters import IdealFilterWindowed, ButterBandpassFilter
import scipy.fftpack as fftpack

def yiq_to_rgb_new(y, i, q):
    r = y + 0.948262*i + 0.624013*q
    g = y - 0.276066*i - 0.639810*q
    b = y - 1.105450*i + 1.729860*q
    return (r, g, b)

def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    # b, a = butter(order, [lowcut, highcut], btype='band')
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def phase_filter_(arr, refarr, lowcut, highcut, fpsForBandPass):
 
    trans = fftpack.fft((np.angle(arr) - np.angle(refarr)))
    trans = fftpack.fftshift(trans)
    phase_diff_fft = trans


    filtered_phase = butter_bandpass_filter(phase_diff_fft, lowcut, highcut, fpsForBandPass)

    filtered_phase = np.resize(filtered_phase,(1,filtered_phase.shape[0]))

    phase_diff_trans = np.real(fftpack.ifft2(fftpack.ifftshift(filtered_phase)))
    phase_diff_trans = np.resize(phase_diff_trans,(phase_diff_trans.shape[1],))

    return phase_diff_trans


    

def get_ref_coeff(refNum,vidFname):
    steer = Steerable(5)
    cap = cv2.VideoCapture(vidFname)
    cap.set(1, refNum)
    ret, refIm = cap.read()
    yiqRef = np.empty((refIm.shape[0], refIm.shape[1], 3))

    [yiqRef[:, :, 0], refIm[:, :, 1], refIm[:, :, 2]] = colorsys.rgb_to_yiq(refIm[:, :, 0], refIm[:, :, 1], refIm[:, :, 2])

    lumRef = yiqRef[:, :, 0]
    refcoeff = steer.buildSCFpyr(lumRef)
    return refcoeff

def phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq):

    # initialize the steerable complex pyramid
    steer = Steerable(5)
    pyArr = Pyramid2arr(steer)

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
    nrFrames = min(vidFrames, maxFrames) - 200

    filter = ButterBandpassFilter(1, lowFreq, highFreq, fps=fpsForBandPass)


    print 'FrameNr:', 
    for frameNr in range( nrFrames + windowSize ):
        print frameNr,
        sys.stdout.flush() 

        if frameNr < nrFrames:
            # read frame
            _, im = vidReader.read()
               
            if im is None:
                # if unexpected, quit
                break
            # convert to gray image
            if len(im.shape) > 2:
                grayIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            else:
                # already a grayscale image?
                grayIm = im

#            yiqIm = np.empty((im.shape[0], im.shape[1], 3))
#            [yiqIm[:, :, 0], yiqIm[:, :, 1], yiqIm[:, :, 2]] = colorsys.rgb_to_yiq(im[:, :, 0], im[:, :, 1], im[:, :, 2])
#            grayImIm = yiqIm[:, :, 0]

            # get coeffs for pyramid
            coeff = steer.buildSCFpyr(grayIm)

            refcoeff = get_ref_coeff(0,vidFname)
            
            arr = pyArr.p2a(coeff)
            refarr = pyArr.p2a(refcoeff)
            phases = np.angle(arr)

            filter_phase = phase_filter_(arr, refarr, lowFreq, highFreq, fpsForBandPass)
            
            magnified_phase = (phases - filter_phase) +filter_phase*factor
            new_Arr = np.abs(arr)*np.exp(magnified_phase*1j)

            print 'shape of arr'
            print new_Arr.shape
            new_coeff = pyArr.a2p(new_Arr)

            out = steer.reconSCFpyr(new_coeff)

            # make a RGB image
            rgbIm = np.empty((im.shape[0], im.shape[1], 3))
            rgbIm[:,:,0] = out
            rgbIm[:,:,1] = out
            rgbIm[:,:,2] = out

            #[rgbIm[:, :, 0], rgbIm[:, :, 1], rgbIm[:, :, 2]] = yiq_to_rgb_new(out, yiqIm[:, :, 1], yiqIm[:, :, 2])
            rgbIm[rgbIm > 255] = 255
            rgbIm[rgbIm < 0] = 0

            res = cv2.convertScaleAbs(rgbIm)
            vidWriter.write(res)

    # free the video reader/writer
    vidReader.release()
    vidWriter.release()   


################# main script


vidFname = './media/ewi_tree.mp4'

# maximum nr of frames to process
maxFrames = 60000
# the size of the sliding window
windowSize = 30
# the magnifaction factor
factor = 80
# the fps used for the bandpass
fpsForBandPass = 30  # use -1 for input video fps
# low ideal filter
lowFreq = 6
# high ideal filter
highFreq = 10
# output video filename
vidFnameOut = vidFname + '-Mag%dIdeal-lo%d-hi%d.avi' % (factor, lowFreq, highFreq)

phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq)



