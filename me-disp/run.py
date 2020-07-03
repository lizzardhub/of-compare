import numpy as np
from glob import glob
import cv2

import os
from skimage import io
from time import time

import sys
sys.path.append("..") # Adds higher directory to python modules path
from common.utils.flowlib import read_flow, flow_to_image, write_flow


from DE import DE



'''
#from skimage.io import imsave
prefix = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(prefix, "test", "l.png.my"), "rb") as f:
    data = f.read()
    l = [i for i in data]
l = numpy.array(l).astype(numpy.uint8)[:3 * 1024 * 432].reshape((432, 1024, 3))

with open(os.path.join(prefix, "test", "r.png.my"), "rb") as f:
    data = f.read()
    r = [i for i in data]
r = numpy.array(r).astype(numpy.uint8)[:3 * 1024 * 432].reshape((432, 1024, 3))

processor = ME(l.shape[1], l.shape[0])
f = processor.EstimateME(l, r)
print(l.shape)
u, v = f
print(u.shape, v.shape)
print(np.sum(np.abs(u)))
print(np.sum(np.abs(v)))'''

def estimate_disp(processor, img_l, img_r):
    processor.EstimateDisp(img_l, img_r)
    f = processor.GetDisparityMap()
    u, v = f[0][0], f[0][1]
    flow = np.stack([u, v], axis=2).astype(np.float32)
    return flow

image_list = sorted(glob('/content/frames/*'))
img_l = io.imread(image_list[0]) # Fake read to determine dimensions
img_r = io.imread(image_list[1])

h, w = img_l.shape[:2]

max_h = int(h // 16 * 16)
max_w = int(w // 16 * 16)
if max_h < h: max_h += 16
if max_w < w: max_w += 16

processor = DE(max_w, max_h)

for i in range(0, len(image_list) - 1, 2):
    fname = image_list[i].split('/')[-1]
    img_l = io.imread(image_list[i])
    img_r = io.imread(image_list[i + 1])
    print('Image paths:', image_list[i], image_list[i + 1])

    img_l = cv2.resize(img_l,(max_w, max_h))
    img_r = cv2.resize(img_r,(max_w, max_h))

    tb = time()
    flow = estimate_disp(processor, img_l, img_r)
    flow = cv2.resize(flow, (w, h))
    write_flow(flow, './images/out/' + fname.split('.')[0] + '.flo')
    print('{:.3f} seconds elapsed'.format(time() - tb))

    tb = time()
    flow = estimate_disp(processor, img_r, img_l)
    flow = cv2.resize(flow, (w, h))
    write_flow(flow, './images/out/' + fname.split('.')[0] + '_b.flo')
    print('{:.3f} seconds elapsed'.format(time() - tb))
    #io.imsave('./images/out/' + fname, flow_to_image(flow))
