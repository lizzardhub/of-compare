from me import ME
import numpy
import numpy as np
from glob import glob
import cv2

import os
from skimage import io
from time import time

from skimage.color import rgba2rgb

import sys
sys.path.append("..") # Adds higher directory to python modules path
from common.utils.flowlib import read_flow, flow_to_image, write_flow

from common.my_flow import *


image_list = ['/content/0001.png', '/content/0002.png']
img_l = io.imread(image_list[0])
img_r = io.imread(image_list[1])

h, w = img_l.shape[:2]

max_h = int(h // 16 * 16)
max_w = int(w // 16 * 16)
if max_h < h: max_h += 16
if max_w < w: max_w += 16

processor = ME(max_w, max_h, max_len_hor=50, max_len_vert=50)

img_l = cv2.resize(img_l,(max_w, max_h))
img_r = cv2.resize(img_r,(max_w, max_h))

for i in range(0, len(image_list) - 1):
    fname = image_list[i].split('/')[-1]
    img_l = rgba2rgb(io.imread(image_list[i]))
    img_r = rgba2rgb(io.imread(image_list[i + 1]))
    print('shape', img_l.shape, img_r.shape)

    img_l = cv2.resize(img_l,(max_w, max_h))
    img_r = cv2.resize(img_r,(max_w, max_h))

    tb = time()
    f = processor.EstimateME(img_l, img_r)
    u, v = f
    flow = np.stack([u, v], axis=2).astype(np.float32)
    flow = cv2.resize(flow, (w, h))
    write_flow(flow, './images/out/' + fname.split('.')[0] + '.flo')

    f = processor.EstimateME(img_r, img_l)
    u, v = f
    flow = np.stack([u, v], axis=2).astype(np.float32)
    flow = cv2.resize(flow, (w, h))
    write_flow(flow, './images/out/' + fname.split('.')[0] + '_b.flo')
    #io.imsave('./images/out/' + fname, flow_to_image(flow))


















if (0):
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





    def estimate_disp(processor, img_l, img_r):
        processor.EstimateDisp(img_l, img_r)
        f = processor.GetDisparityMap()
        u, v = f[0][0], f[0][1]
        flow = np.stack([u, v], axis=2).astype(np.float32)
        u, v = f[1][0], f[1][1]
        flow_b = np.stack([u, v], axis=2).astype(np.float32)
        return (flow, flow_b)

    image_list = sorted(glob('/content/frames/*'))
    img_l = io.imread(image_list[0]) # Fake read to determine dimensions
    img_r = io.imread(image_list[1])

    h, w = img_l.shape[:2]

    max_h = int(h // 16 * 16)
    max_w = int(w // 16 * 16)
    if max_h < h: max_h += 16
    if max_w < w: max_w += 16

    processor = DE(max_w, max_h)

    for i in range(0, len(image_list) - 1):
        fname = image_list[i].split('/')[-1]
        img_l = io.imread(image_list[i])
        img_r = io.imread(image_list[i + 1])
        print('Image paths:', image_list[i], image_list[i + 1])

        img_l = cv2.resize(img_l,(max_w, max_h))
        img_r = cv2.resize(img_r,(max_w, max_h))

        tb = time()
        flow, flow_b = estimate_disp(processor, img_l, img_r)
        flow = cv2.resize(flow, (w, h))
        write_flow(flow, './images/out/' + fname.split('.')[0] + '.flo')
        print('{:.3f} seconds elapsed'.format(time() - tb))

        tb = time()
        #flow = estimate_disp(processor, img_r, img_l)
        flow = flow_b
        flow = cv2.resize(flow, (w, h))
        write_flow(flow, './images/out/' + fname.split('.')[0] + '_b.flo')
        print('{:.3f} seconds elapsed'.format(time() - tb))
        #io.imsave('./images/out/' + fname, flow_to_image(flow))
