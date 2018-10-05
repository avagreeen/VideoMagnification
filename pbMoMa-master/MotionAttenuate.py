from perceptual.filterbank import *

import cv2
import colorsys
import matplotlib.pyplot as plt
# from colorutils import rgb_to_yiq, yiq_to_rgb

# determine what OpenCV version we are using
try:
    import cv2.cv as cv

    USE_CV2 = True
except ImportError:
    # OpenCV 3.x does not have cv2.cv submodule
    USE_CV2 = False

import sys
import numpy as np

from temporal_filters import IdealFilterWindowed, ButterBandpassFilter
import scipy.fftpack as fftpack

def rgb2ntsc(src):
    [rows,cols]=src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[0.114, 0.587, 0.298], [-0.321, -0.275, 0.596], [0.311, -0.528, 0.212]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst

#convert YIQ to RBG
def ntsc2rbg(src):
    [rows, cols] = src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[1, -1.108, 1.705], [1, -0.272, -0.647], [1, 0.956, 0.620]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst


def yiq_to_rgb_new(y, i, q):
    r = y + 0.948262 * i + 0.624013 * q
    g = y - 0.276066 * i - 0.639810 * q
    b = y - 1.105450 * i + 1.729860 * q
    # r[r < 0.0] = 0.0
    # r[r > 1.0] = 1.0
    # b[b < 0.0] = 0.0
    # b[b > 1.0] = 1.0
    # g[g < 0.0] = 0.0
    # g[g > 1.0] = 1.0
    return (r, g, b)


def flatten_pyr(coeff, steer):
    level = range(1, steer.height - 1)
    band = range(steer.nbands)
    arr = np.hstack([coeff[lvl][b].flatten() for lvl in level for b in band])
    arr = np.hstack((coeff[0].flatten(), arr, coeff[-1].flatten()))
    return arr


def reverse_arr(arr, steer, coeff):
    rev_coeff = []
    for item, i in zip(coeff, range(len(coeff))):
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


def MotionAttuenuate(vidFname, vidFnameOut):
    # initialize the steerable complex pyramid
    steer = Steerable(5)

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
    vidWriter = cv2.VideoWriter(vidFnameOut, fourcc, int(fps), (width, height), 1)
    print 'Writing:', vidFnameOut

    # how many frames
    nrFrames = min(vidFrames, maxFrames) - 250

    # read video
    refNum = 0

    cap = cv2.VideoCapture(vidFname)
    cap.set(1, refNum)
    ret, refIm = cap.read()
    yiqRef = np.empty((refIm.shape[0], refIm.shape[1], 3))

    [yiqRef[:, :, 0], yiqRef[:, :, 1], yiqRef[:, :, 2]] = colorsys.rgb_to_yiq(refIm[:, :, 0], refIm[:, :, 1],
                                                                              refIm[:, :, 2])
    channels = 3
    fixedPhase = []
    for i in range(channels):
        Ref = yiqRef[:, :, i]
        refcoeff = steer.buildSCFpyr(Ref)
        refcoeff_flat = flatten_pyr(refcoeff, steer)
        fixedPhase.append(np.angle(refcoeff_flat))

    print 'FrameNr:',
    for frameNr in range(nrFrames):
        print frameNr,
        sys.stdout.flush()



        if frameNr < nrFrames:
            # read frame
            _, im = vidReader.read()

            if im is None:
                # if unexpected, quit
                break

            #  change rgb image into yiq image

            yiqIm = np.empty((im.shape[0], im.shape[1], 3))
            [yiqIm[:, :, 0], yiqIm[:, :, 1], yiqIm[:, :, 2]] = colorsys.rgb_to_yiq(im[:, :, 0], im[:, :, 1],
                                                                                   im[:, :, 2])


            out = np.empty((im.shape[0], im.shape[1], 3))
            for i in range(channels):
                test_Im = yiqIm[:, :, i]
                # get coeffs for pyramid
                coeff = steer.buildSCFpyr(test_Im)
                arr = flatten_pyr(coeff, steer)
                new_arr = np.abs(arr) * np.exp(fixedPhase[i] * 1j)
                # print(np.exp(fixedPhase[i] * 1j))
                new_coeff = reverse_arr(new_arr, steer, coeff)

                # reconstruction
                out[:, :, i] = steer.reconSCFpyr(new_coeff)
            # make a RGB image
            rgbIm = np.empty((im.shape[0], im.shape[1], 3))
            [rgbIm[:, :, 0], rgbIm[:, :, 1], rgbIm[:, :, 2]] = yiq_to_rgb_new(out[:, :, 0], out[:, :, 1], out[:, :, 2])
            # rgbIm = ntsc2rbg(out)
            # print(rgbIm)
            rgbIm[rgbIm > 255] = 255
            rgbIm[rgbIm < 0] = 0


            # plt.imshow(rgbIm)
            # plt.show()
            # plt.subplot(321)
            # plt.imshow(out[:, :, 0])
            # # plt.show()
            # plt.subplot(322)
            # plt.imshow(yiqIm[:, :, 0])
            # # plt.show()
            # plt.subplot(323)
            # plt.imshow(out[:, :, 1])
            # # plt.show()
            # plt.subplot(324)
            # plt.imshow(yiqIm[:, :, 1])
            # # plt.show()
            # plt.subplot(325)
            # plt.imshow(out[:, :, 2])
            # # plt.show()
            # plt.subplot(326)
            # plt.imshow(yiqIm[:, :, 2])
            # plt.show()

            # write to disk
            res = cv2.convertScaleAbs(rgbIm)
            vidWriter.write(res)
            # colorsys.yiq_to_rgb(out[:, :, 0], out[:, :, 1], out[:, :, 2])
        # free the video reader/writer
    vidReader.release()
    vidWriter.release()

################# main script

vidFname = './media/face.mp4'

# maximum nr of frames to process
maxFrames = 60000

# output video filename
vidFnameOut = vidFname + 'Motion_Attuenuate.avi'

MotionAttuenuate(vidFname, vidFnameOut)
