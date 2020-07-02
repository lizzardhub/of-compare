from __future__ import absolute_import, division, print_function
from pathlib import Path
from glob import glob
import os
import sys
import subprocess
import shutil
import random
from time import time, clock
from datetime import datetime

import numpy as np

import png
import skimage
from skimage import io
from skimage.io import imread, imsave
from skimage.transform import rescale
from skimage.color import rgb2gray
import imageio
from PIL import Image, ImageFont, ImageDraw
from scipy import ndimage
import matplotlib as mpl
import matplotlib.colors as cl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

def subp_run_str(cmd, output=True):
    print('RUN:', cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    if output:
        for line in process.stdout:
            print(line.decode(), end='')
    rc = process.poll()
    return rc
