import cv2
import sys


# Input image
src = cv2.imread(sys.argv[1])

# The -1 means to filp around both x and y axes.
dst = cv2.flip(src, -1)

# Output image
cv2.imwrite('ans2.png', dst)

