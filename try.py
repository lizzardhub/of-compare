from DE import DE

import cv2
import numpy as np
#from flowlib import flow_to_image, read_flow
from common.file_move import *
print(write_flow)
from skimage import io

from os.path import join
from glob import glob
from shutil import copyfile
from time import time

def show(img):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.show()

im1 = io.imread('frames/frame_0004.jpg')
im2 = io.imread('frames/frame_0005.jpg')
h, w, _c = im1.shape

processor = DE(w, h)
for i in range(1):
    start = time()
    processor.EstimateDisp(im1, im2)
    f = processor.GetRawDisparityMap()
    print(time() - start, 'seconds elapsed')

print(f[0].shape, f[1].shape)
u, v = f[1][0], f[1][1]
flow = np.stack([u, v], axis=2).astype(np.float32)
flow = cv2.resize(flow, (w, h))
write_flow(flow, 'a.flo')


fwd = 'a.flo'
flow_f = read_flo(fwd)
img = flow_to_png_middlebury(flow_f)
imsave('a.jpg', (im1 * 0.5 + img * 0.5).astype(np.uint8) )
del processor
