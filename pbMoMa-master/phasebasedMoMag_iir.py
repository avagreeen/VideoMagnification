from perceptual.filterbank import *

import cv2
import colorsys
from scipy.signal import butter, lfilter, iirfilter
import matplotlib.pyplot as plt
import math
# determine what OpenCV version we are using
try:
    import cv2.cv as cv
    USE_CV2 = True
except ImportError:
    # OpenCV 3.x does not have cv2.cv submodule
    USE_CV2 = False
    
import sys
import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pyr2arr import Pyramid2arr
from temporal_filters import IdealFilterWindowed, ButterBandpassFilter
import scipy.fftpack as fftpack

def temporal_ideal_filter(tensor,low,high,fps,axis=0):
    fft=fftpack.fft(tensor,axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)

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

def flatten_pyr(coeff, steer):
    level = range(1, steer.height - 1)
    band = range(steer.nbands)
    init = [coeff[0]]
    for lvl in level:
        for b in band:
            init.append(coeff[lvl][b].shape)
    Array = np.hstack([coeff[lvl][b].flatten() for lvl in level for b in band])
    Array = np.hstack((coeff[0].flatten(), Array, coeff[-1].flatten()))
    return Array
def reverse_arr(arr, steer, coeff):
    rev_coeff = []
    for item in coeff:
        if len((np.array(item).shape)) == 2:
            pop = arr[:np.size(item)]
            arr = arr[np.size(item):]
            reverse = pop.reshape(item.shape)

        else:
            reverse = []
            for j in range(np.array(item).shape[0]):
                pop = arr[:np.size(item[j])]
                arr = arr[np.size(item[j]):]
                reverse.append(pop.reshape(item[j].shape))
        rev_coeff.append(reverse)
    return rev_coeff

def diff_iir(ref, phase, low, high, fps):
    low_cut = low / (0.5 * fps)
    high_cut = high / (0.5 * fps)
    filter_phase1 = (1-low_cut) * ref + low_cut * phase
    filter_phase2 = (1-high_cut) * ref + high_cut * phase
    #filter_phase2 - filter_phase1= ref*(low-high)+phase*high-low=(high-low)*(phase-ref)/0.5*fps
    return filter_phase2 - filter_phase1




def phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq):

    # initialize the steerable complex pyramid
    steer = Steerable(4)
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
    nrFrames = min(vidFrames, maxFrames)


    print 'FrameNr:', 
    for frameNr in range( nrFrames + windowSize ):
        print frameNr,
        sys.stdout.flush()

        if frameNr == 0:
            refNum = 0
        else:
            refNum = 0 # frameNr - 1

        cap = cv2.VideoCapture(vidFname)
        cap.set(1, refNum)
        ret, refIm = cap.read()
        yiqRef = np.empty((refIm.shape[0], refIm.shape[1], 3))

        [yiqRef[:, :, 0], refIm[:, :, 1], refIm[:, :, 2]] = colorsys.rgb_to_yiq(refIm[:, :, 0], refIm[:, :, 1],
                                                                                refIm[:, :, 2])
        lumRef = yiqRef[:, :, 0]
        refcoeff = steer.buildSCFpyr(lumRef)

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

            yiqIm = np.empty((im.shape[0], im.shape[1], 3))
            [yiqIm[:, :, 0], yiqIm[:, :, 1], yiqIm[:, :, 2]] = colorsys.rgb_to_yiq(im[:, :, 0], im[:, :, 1], im[:, :, 2])
            grayIm = yiqIm[:, :, 0]
            # get coeffs for pyramid
            coeff = steer.buildSCFpyr(grayIm)
            arr = pyArr.p2a(coeff)

            fil_phase = []


            # # add image pyramid to video array
            # # NOTE: on first frame, this will init rotating array to store the pyramid coeffs
            # arr = pyArr.p2a(coeff)
            ref_arr = pyArr.p2a(refcoeff)
            test_arr = flatten_pyr(coeff, steer)
            ref_arr = flatten_pyr(refcoeff, steer)
            phase_diff = np.mod((np.angle(arr) - np.angle(ref_arr) + math.pi), 2*math.pi) - math.pi
            fft_phase = np.fft.fft(phase_diff)
            fft_phase = np.fft.fftshift(fft_phase)
            # # filter = firwin(200, [lowFreq / (0.5 * fpsForBandPass), highFreq / (0.5 * fpsForBandPass)],
            # #                 pass_zero = False, window='blackmanharris')
            # # filter_phase = lfilter(filter, 1.0, fft_phase)
            ref_phase = np.fft.fftshift(np.fft.fft(np.angle(ref_arr)-np.angle(ref_arr)))
            filter_phase = diff_iir(ref_phase, fft_phase, lowFreq, highFreq, fpsForBandPass)
            # filter_phase = butter_bandpass_filter(fft_phase, lowFreq, highFreq, fpsForBandPass)


            filter_phase = np.fft.ifftshift(filter_phase)
            ifft_phase = np.real(np.fft.ifft(filter_phase))
            magnified_phase = np.angle(arr) + (factor - 1) * ifft_phase
            new_arr = np.abs(test_arr) * np.exp(magnified_phase * 1j)
            new_coeff = pyArr.a2p(new_arr)

            out = steer.reconSCFpyr(new_coeff)
            # break
            # clip values out of range
            # out[out>255] = 255
            # out[out<0] = 0
            
            # make a RGB image
            rgbIm = np.empty((im.shape[0], im.shape[1], 3))
            # rgbIm[:,:,0] = out
            # rgbIm[:,:,1] = out
            # rgbIm[:,:,2] = out

            [rgbIm[:, :, 0], rgbIm[:, :, 1], rgbIm[:, :, 2]] = yiq_to_rgb_new(out, yiqIm[:, :, 1], yiqIm[:, :, 2])
            rgbIm[rgbIm > 255] = 255
            rgbIm[rgbIm < 0] = 0
            # plt.imshow(out)
            # plt.show()
            # plt.imshow(yiqIm[:, :, 0])
            # plt.show()
            #write to disk
            res = cv2.convertScaleAbs(rgbIm)
            vidWriter.write(res)

    # free the video reader/writer
    vidReader.release()
    vidWriter.release()   


################# main script


vidFname = './media/iron_origin.mp4'

# maximum nr of frames to process
maxFrames = 60000
# the size of the sliding
windowSize = 20
# the magnifaction factor
factor = 120
# the fps used for the bandpass
fpsForBandPass = 100  # use -1 for input video fps
# low ideal filter
lowFreq = 1.5
# high ideal filter
highFreq = 4
# output video filename
vidFnameOut = vidFname + '-Mag%dIdeal-lo%d-hi%d-pre.avi' % (factor, lowFreq, highFreq)

phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq)



