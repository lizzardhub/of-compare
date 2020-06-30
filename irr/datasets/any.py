from __future__ import absolute_import, division, print_function

import os
import torch.utils.data as data
from glob import glob
from time import time

from torchvision import transforms as vision_transforms
import numpy as np

from . import transforms
from . import common

import tools

import random
from datetime import datetime

random.seed(datetime.now())

class _Sintel(data.Dataset):
    def __init__(self,
                 args,
                 dir_root=None):

        self._args = args

        images_root = os.path.join(dir_root, "frames/in")

        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!")

        all_img_filenames = sorted(glob(os.path.join(images_root, "*.jpg")))

        # Remember base for substraction at runtime
        # e.g. subtract_base = "/home/user/.../MPI-Sintel-Complete/training/clean"
        self._substract_base = tools.cd_dotdot(images_root)

        # ------------------------------------------------------------------------
        # Get unique basenames
        # ------------------------------------------------------------------------
        # e.g. base_folders = [alley_1", "alley_2", "ambush_2", ...]
        substract_full_base = tools.cd_dotdot(all_img_filenames[0])

        self._image_list = []

        img_filenames = all_img_filenames

        for i in range(0, len(img_filenames) - 1):

            im1 = img_filenames[i]
            im2 = img_filenames[i + 1]

            self._image_list += [[im1, im2]]

        # ----------------------------------------------------------
        # photometric_augmentations
        # ----------------------------------------------------------

        self._photometric_transform = transforms.ConcatTransformSplitChainer([
            # uint8 -> FloatTensor
            vision_transforms.transforms.ToTensor(),
        ], from_numpy=True, to_numpy=False)

        self._size = len(self._image_list)

    def __getitem__(self, index):
        cu_id = int(random.random() * 1000)
        a = open('/content/a.txt', 'a')
        tb = time()
        print('Start load', 'id=' + str(cu_id), round(time() - tb, 3), file=a)
        a.flush()
        index = index % self._size

        im1_filename = self._image_list[index][0]
        im2_filename = self._image_list[index][1]

        # read float32 images and flow
        im1_np0 = common.read_image_as_byte(im1_filename)
        im2_np0 = common.read_image_as_byte(im2_filename)
        h, w, _c = im1_np0.shape
        flo_np0 = np.zeros((h, w, 2), dtype=np.float32)
        #flo_np0 = common.read_flo_as_float32(flo_filename)
        occ_np0 = np.zeros((h, w), dtype=np.float32)
        #occ_np0 = common.read_occ_image_as_float32(occ_filename)

        # possibly apply photometric transformations
        im1, im2 = self._photometric_transform(im1_np0, im2_np0)
        flo = common.numpy2torch(flo_np0)
        occ = common.numpy2torch(occ_np0)
        
        # e.g. "clean/alley_1/"
        basedir = os.path.splitext(os.path.dirname(im1_filename).replace(self._substract_base, "")[1:])[0]

        # example filename
        basename = os.path.splitext(os.path.basename(im1_filename))[0]

        example_dict = {
            "input1": im1,
            "input2": im2,
            "index": index,
            "basedir": basedir,
            "basename": basename,
            "target1": flo,
            "target_occ1": occ
        }
        print('End load', 'id=' + str(cu_id), round(time() - tb, 3), file=a)
        a.close()

        return example_dict

    def __len__(self):
        return self._size

class AnyValid(_Sintel):
    def __init__(self, args, root, photometric_augmentations=False):
        dir_root = os.path.join(root, "training")
        super(AnyValid, self).__init__(
            args,
            dir_root=dir_root)
