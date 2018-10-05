import numpy as np
from pyPyrTools import SCFpyr
from skimage import color
from skimage import transform
import pyPyrTools as ppt
from scipy.signal import butter, lfilter, iirfilter
import matplotlib.pyplot as plt
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

def decompose(img):
    myPyr = ppt.SFpyr(img)   
    phase = []
    phase.append([np.angle(myPyr.pyr[level]) for level in range(1, len(myPyr.pyr) - 1)])
    
    shapes=[]
    shapes.append(myPyr.pyr[0].shape)
    bot=myPyr.pyr[0].flatten()
    f_phase = bot
    for i in range(len(phase)):
        print phase[i].shape
        shapes.append(phase[i].shape)
        item = np.array(phase[i]).flatten()
        f_phase = np.concatenate([f_phase,item])
       # print phase.shape,i
    top = myPyr.pyr[len(myPyr.pyr)-1]
    shapes.append(top.shape)
    phase = np.concatenate([phase,top.flatten()])
    return pyramids,myPyr,phase,shapes

def _decompose(img):
    height = 3
    order = 3
    twidth = 1
    lab = np.array(color.rgb2lab(img)) / 255.
    pyramids = {'pyramids': [], 'high_pass': [], 'low_pass': [], 'phase': [], 'amplitude': [], 'pind': 0} 
      # , height)  # , order, twidth)
            # - `height`(optional) specifies the number of pyramid levels to build.
            # - `order`(optional), int.Default value is 3. The number of orientation bands - 1.
            # - `twidth`(optional), int.Default value is 1. The width of the transition region of the radial
            # lowpass function, in octaves
    for i in range(img.shape[-1]):
        myPyr = ppt.SFpyr(lab[...,i])
        #print myPyr.pyr[i]
            # pyr = SCFpyr(lab[..., i], ht, n_orientations - 1, t_width, scale, n_scales, np)
        pyramids['pyramids'].append(myPyr)
        pyramids['high_pass'].append(myPyr.pyrHigh())
        pyramids['low_pass'].append(myPyr.pyrLow())
        pyramids['phase'].append([np.angle(myPyr.pyr[level]) for level in range(1, len(myPyr.pyr) - 1)])
        pyramids['amplitude'].append([np.abs(myPyr.pyr[level]) for level in range(1, len(myPyr.pyr) - 1)])

    return pyramids,myPyr


def compute_phase_difference(L, R, *args):
    #np = L['pyramids'][0].np
    phase_diff_out = []
    for i in range(len(L['phase'])):
        print L['phase'][i][1].shape

        for j in range(len(L['phase'][i])):
            print '------'
            #print R['phase'][i][j]

            ps = np.array(R['phase'][i][j]).shape
            arr = np.resize(R['phase'][i][j],(1,ps[0]*ps[1]))
            filterd_phase = butter_bandpass_filter(arr, 72, 92, 600)
            R['phase'][i][j] = np.resize(filterd_phase,ps)
            plt.imshow(R['phase'][i][j])

        phase_diff = [np.arctan2(np.sin(R['phase'][i][j] - L['phase'][i][j]), np.cos(R['phase'][i][j] - L['phase'][i][j]))
                      for j in range(len(L['phase'][i]))]

        #phase_diff_new = shift_correction(phase_diff, L['pyramids'][i], *args)
        #unwrapped_phase_diff = []
        #for j in range(len(phase_diff)):
         #   unwrapped_phase_diff.append(unwrap(np.stack([phase_diff_new[j], list(phase_diff)[j]], 0),
                      #                         np=L['pyramids'][0].np)[0])
        #phase_diff_out.append(unwrapped_phase_diff)
    return phase_diff


def shift_correction(pyr, pyramid, *args):
    n_high_elems = pyramid.pyrSize[0]
    n_low_elems = pyramid.pyrSize[-1]
    corrected_pyr = list(pyr)
    corrected_pyr.insert(0, np.zeros(n_high_elems))
    corrected_pyr.append(np.zeros(n_low_elems))
    n_levels = pyramid.spyrHt()
    n_bands = pyramid.numBands()
    for level in range(n_levels - 1, -1, -1):
        corrected_level = correct_level(corrected_pyr, pyramid, level, *args)
        start_ind = 1 + n_bands * level
        corrected_pyr[start_ind:start_ind+n_bands] = corrected_level
    corrected_pyr = corrected_pyr[1:len(corrected_pyr) - 1]
    return corrected_pyr


def correct_level(pyr, pyramid, level, *args):
    scale = args[0]
    limit = args[1]
    n_levels = pyramid.spyrHt()
    n_bands = pyramid.numBands()
    out_level = []
    if level < n_levels - 1:
        dims = pyramid.pyrSize[1+n_bands*level]
        for band in range(n_bands):
            index_lo = pyramid.bandIndex(level + 1, band)
            low_level_small = pyr[index_lo]
            if pyramid.np.__name__ == 'numpy':
                low_level = transform.resize(low_level_small, dims, mode='reflect').astype('float32')
            else:
                low_level = pyramid.np.array(transform.resize(pyramid.np.asnumpy(low_level_small),
                                                              dims, mode='reflect').astype('float32'))
            index_hi = pyramid.bandIndex(level, band)
            high_level = pyr[index_hi]
            unwrapped = pyramid.np.stack([low_level.reshape(-1) / scale, high_level.reshape(-1)], 0)
            unwrapped = unwrap(unwrapped, np=pyramid.np)
            high_level = unwrapped[1]
            high_level = pyramid.np.reshape(high_level, dims)
            angle_diff = pyramid.np.arctan2(pyramid.np.sin(high_level-low_level/scale),
                                            pyramid.np.cos(high_level-low_level/scale))
            to_fix = pyramid.np.abs(angle_diff) > (np.pi / 2)
            high_level[to_fix] = low_level[to_fix] / scale

            if limit > 0:
                to_fix = pyramid.np.abs(high_level) > (limit * np.pi / scale ** (n_levels - level))
                high_level[to_fix] = low_level[to_fix] / scale

            out_level.append(high_level)

    if level == n_levels - 1:
        for band in range(n_bands):
            index_lo = pyramid.bandIndex(level, band)
            low_level = pyr[index_lo]
            if limit > 0:
                to_fix = pyramid.np.abs(low_level) > (limit * np.pi / scale ** (n_levels - level))
                low_level[to_fix] = 0.
            out_level.append(low_level)
    return out_level


def unwrap(p, cutoff=np.pi, np=np):
    def local_unwrap(p, cutoff):
        dp = p[1] - p[0]
        dps = np.mod(dp + np.pi, 2 * np.pi) - np.pi
        dps[np.logical_and(dps == -np.pi, dp > 0)] = np.pi
        dp_corr = dps - dp
        dp_corr[np.abs(dp) < cutoff] = 0.
        p[1] += dp_corr
        return p
    shape = p.shape
    p = np.reshape(p, (shape[0], np.prod(shape[1:])))
    q = local_unwrap(p, cutoff)
    q = np.reshape(q, shape)
    return q


def interpolate_pyramid(L, R, phase_diff, alpha):
    new_pyr = []
    for i in range(len(phase_diff)):
        new_pyr.append([])
        high_pass = L['high_pass'][i] if alpha < 0.5 else R['high_pass'][i]
        low_pass = (1 - alpha) * L['low_pass'][i] + alpha * R['low_pass'][i]
        new_pyr[i].append(high_pass)
        for k in range(len(R['phase'][i])):
            new_phase = R['phase'][i][k] + (alpha - 1) * phase_diff[i][k]
            new_amplitude = (1 - alpha) * L['amplitude'][i][k] + alpha * R['amplitude'][i][k]
            mid_band = new_amplitude * np.e ** (1j * new_phase)
            new_pyr[i].append(mid_band)
        new_pyr[i].append(low_pass)
    return new_pyr


def reconstruct_image(pyr):
    #np = pyr['pyramids'][0].np
    out_img = np.zeros((pyr['pyramids'][0].pyrSize[0][0], pyr['pyramids'][0].pyrSize[0][1], 3))
    for i, pyr in enumerate(pyr['pyramids']):
        out_img[..., i] = pyr.reconPyr('all', 'all')

    out_img = color.lab2rgb(out_img * 255.)
    return out_img


def interpolate_frame(img1, img2, n_frames=1, n_orientations=8, t_width=1, scale=0.5, limit=.4, min_size=15,
                      max_levels=23, np=np):
    h, w, l = img1.shape
    n_scales = min(np.ceil(np.log2(min((h, w))) / np.log2(1. / scale) -
                           (np.log2(min_size) / np.log2(1 / scale))).astype('int'), max_levels)
    step = 1. / (n_frames + 1)

    L = decompose(img1, n_scales, n_orientations, t_width, scale, n_scales, np)
    R = decompose(img2, n_scales, n_orientations, t_width, scale, n_scales, np)

    phase_diff = compute_phase_difference(L, R, scale, limit)

    new_frames = []
    for j in range(n_frames):
        new_pyr = interpolate_pyramid(L, R, phase_diff, step * (j + 1))
        for i, pyr in enumerate(L['pyramids']):
            pyr.pyr = new_pyr[i]
        new_frames.append(reconstruct_image(L))
    return new_frames
