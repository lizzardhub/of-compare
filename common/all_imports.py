from __future__ import absolute_import, division, print_function
from pathlib import Path
from glob import glob
import os
import subprocess
import shutil
import random
from time import time, clock

import numpy as np
import png

from skimage import io
from skimage.io import imread
from skimage.transform import rescale
import imageio
from PIL import Image, ImageFont, ImageDraw
import matplotlib as mpl
import matplotlib.colors as cl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from my_flow import *

def subp_run_str(cmd, output=True):
    print('RUN:', cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    if output:
        for line in process.stdout:
            print(line.decode(), end='')
    rc = process.poll()
    return rc

# Delete temp folders
def rm_tmp():
    subp_run_str('rm -rf frames/* frames_l/* frames_r/* \
        vcn/images/in/* vcn/images/out/* \
        pwc/images/in/* pwc/images/out/* \
        irr/saved_check_point/pwcnet/eval_temp/* \
        sintelall/MPI-Sintel-complete/training/frames/in \
        sintelall/MPI-Sintel-complete/training/frames/out \
        me/images/in/* me/images/out/*')

    subp_run_str('mkdir -p frames frames_l frames_r \
        sintelall/MPI-Sintel-complete/training/frames/in \
        sintelall/MPI-Sintel-complete/training/frames/out')

# Split frames
def frame_ffmpeg_split():
    subp_run_str(['ffmpeg -i cur_video.mkv -qscale:v 2 frames/frame_%04d.jpg']) # Split video into frames

def frame_ffmpeg_split_stereo():
    subp_run_str(['ffmpeg -i cur_video.mkv -qscale:v 2 frames_l/frame_%04d.jpg']) # Split video into frames
    subp_run_str(['ffmpeg -i cur_video2.mkv -qscale:v 2 frames_r/frame_%04d.jpg']) # Split video into frames

# Join outputs of neuronets
def load_and_caption(in_image, text):
    # Returns RGB image

    if len(in_image.shape) > 2:
        if in_image.shape[2] == 4:
            in_image = in_image[:, :, :3]
    else:
        in_image = in_image[:, :, np.newaxis]
        in_image = np.repeat(in_image, 3, 2)

    pil_img = Image.fromarray(in_image.astype(np.uint8))
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 40)
    draw = ImageDraw.Draw(pil_img)
    draw.text((0, 0), text, (255, 0, 0), font=font)
    draw = ImageDraw.Draw(pil_img)
    in_image = np.array(pil_img)

    return in_image

def flow_metrics(stereo=False):
    # Create video from frames located at
    # ./pwc/images/out ./vcn/images/out
    caption = ['CUR FRAME', 'PWC', 'ME', 'IRR-PWC']

    input_path = Path('frames')
    selflow_path = Path('pwc/images/out')
    vcn_path = Path('me/images/out')
    irr_path = Path('sintelall/MPI-Sintel-complete/training/frames/out')
    res_path = [input_path, selflow_path, vcn_path, irr_path]

    vid_path = 'flow_video.mp4'
    writer = imageio.get_writer(vid_path, fps=15)

    h, w, _c = io.imread(glob('frames/*')[0]).shape

    img_list = sorted(input_path.glob('*'))
    img1 = io.imread(input_path / Path(img_list[0]).name)
    img2 = io.imread(input_path / Path(img_list[1]).name)

    # FLOW METRICS:
    # For each method (INPUT, IRR-PWC, VCN, ME)
    # Photometric loss, gradient sum, image boundary-weighted gradient sum
    metrics_history = np.zeros((4, 3, len(img_list) - 1)) # 1000 frames is enough

    step = 1
    if stereo:
        step = 2
    for i in range(0, len(img_list) - 1, step):
        print(i, end=' ')
        if i % 10 == 0:
            print()
        fimg1 = img_list[i]
        fname1 = fimg1.name

        img = np.zeros((h * 2, w * 2, 3))
        res_canvas = [0] * 4
        res_canvas[1] = img[h:h * 2, :w, :]
        res_canvas[2] = img[h:h * 2, w:w * 2, :]
        res_canvas[3] = img[:h, w:w * 2, :]

        if stereo:
            img1 = io.imread(fimg1)
        img2 = io.imread(input_path / Path(img_list[i + 1]).name)
        img[:h, :w, :] = load_and_caption(img1, caption[0])

        # Compute metrics
        filename = fimg1.with_suffix('.flo')
        filename_b = Path(fimg1.stem + '_b').with_suffix('.flo')
        flows = [0] * 4
        flows_b = [0] * 4
        max_rad_me = 0
        for meth in range(1, 4): # Pre-calculate maximum flow
            if (res_path[meth] / filename).exists(): # Flow file with the image name?
                flows[meth] = read_flo(res_path[meth] / filename) # IMPORTANT: read_flo
                flows_b[meth] = read_flo(res_path[meth] / filename_b) # IMPORTANT: read_flo
                if meth != 2:
                    meth_max = np.max( np.sqrt(flows[meth][:, :, 0] ** 2 + flows[meth][:, :, 1] ** 2) )
                    max_rad_me = max(max_rad_me, meth_max)
        for meth in range(1, 4):
            if (res_path[meth] / filename).exists():
                flow = flows[meth]
                flow_b = flows_b[meth]
                img2_warped = warp(img2, flow)
                b_warped = warp(flow_b, flow)

                #diff = photometric_diff(img1, img2, flow)
                #flow_grad = grad_sum(flow)

                weighted_flow_grad = weighted_grad_sum(img1, flow)
                lrc = occlusion_area(flow, b_warped)
                lrc_photo = lrc_weighted_photo(img1, flow, b_warped, img2_warped)

                metrics_history[meth, 0, i] = weighted_flow_grad
                metrics_history[meth, 1, i] = lrc
                metrics_history[meth, 2, i] = lrc_photo

                res_canvas[meth][...] = load_and_caption( # IMPORTANT: flow_to_png_middlebury
                        flow_to_png_middlebury(flow, rad_clip=max_rad_me), caption[meth])
        #elif mode == 1: # Warp
        #    filename = filename.split('.')[0] + '.flo'
        #    if filename in selflow_images:
        #        flow = read_flo(selflow_path + '/' + filename)
        #        diff = round(photometric_diff(img1, img2, flow), 3)
        #        img[h:h * 2, :w, :] = load_and_caption(
        #                warp(img2, flow), caption[1] + ' ' + str(diff))

        rate = 1
        img = img.astype(np.uint8)
        for j in range(rate):
            writer.append_data(img)

        img1 = img2


    metrics = open('metrics.txt', 'w')
    print('FRAMES', metrics_history.shape[2], file=metrics)
    for i in range(1, 4):
        print(caption[i], end=' ', file=metrics)
    print(file=metrics)
    for i in range(metrics_history.shape[2]):
        for meth in range(1, 4):
            for metr in range(3):
                print(round(metrics_history[meth, metr, i], 3), end=' ', file=metrics)
        print(file=metrics)
    metrics.close()

    writer.close()
