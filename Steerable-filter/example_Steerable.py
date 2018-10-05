from perceptual.filterbank import Steerable
import cv2

im = cv2.imread('images/02-112.png', cv2.IMREAD_GRAYSCALE)

# Build a complex steerable pyramid 
# with height 5 (including lowpass and highpass)
s = Steerable(5)
coeff = s.buildSCFpyr(im)

# coeff is an array and subbands can be accessed as follows:
# coeff[0] : highpass
# coeff[1][0], coeff[1][1], coeff[1][2], coeff[1][3] : bandpass of scale 1
# coeff[2][0], coeff[2][1], coeff[2][2], coeff[2][3] : bandpass of scale 2
# ...
# coeff[4]: lowpass. It can also be accessed as coeff[-1]
cv2.imwrite("subbband.png", coeff[1][0].real)

# or visualization of whole decomposition
# cv2.imwrite("coeff.png", visualize(coeff))

# reconstruction
out = s.reconSCFpyr(coeff)
cv2.imwrite("recon.png", out)
