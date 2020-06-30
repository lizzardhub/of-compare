from time import time as clock
import os
from glob import glob
from skimage import io

print(os.getcwd())

from matplotlib import pyplot as plt
import utils
from utils.flowlib import read_flow, flow_to_image, write_flow
import cv2
import numpy as np
import skimage
from models.VCN import VCN
import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision import transforms as vision_transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import common
import transforms
import random
from time import time


def point_vec(img,flow):
    meshgrid = np.meshgrid(range(img.shape[1]),range(img.shape[0]))
    dispimg = cv2.resize(img, None,fx=4,fy=4)
    colorflow = flow_to_image(flow).astype(int)
    for i in range(img.shape[1]): # x
        for j in range(img.shape[0]): # y
            if flow[j,i,2] != 1: continue
            if j%10!=0 or i%10!=0: continue
            xend = int((meshgrid[0][j,i]+flow[j,i,0])*4)
            yend = int((meshgrid[1][j,i]+flow[j,i,1])*4)
            leng = np.linalg.norm(flow[j,i,:2])
            if leng<1:continue
            dispimg = cv2.arrowedLine(dispimg, (meshgrid[0][j,i]*4,meshgrid[1][j,i]*4),\
                                      (xend,yend),
                                      (int(colorflow[j,i,2]),int(colorflow[j,i,1]),int(colorflow[j,i,0])),5,tipLength=8/leng,line_type=cv2.LINE_AA)
    return dispimg

class _Sintel(data.Dataset):
    def __init__(self,
                 dir_root=None):


        images_root = os.path.join(dir_root, "in")

        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!")

        all_img_filenames = sorted(glob(os.path.join(images_root, "*.png")))

        # Remember base for substraction at runtime
        # e.g. subtract_base = "/home/user/.../MPI-Sintel-Complete/training/clean"
        #self._substract_base = tools.cd_dotdot(images_root)

        # ------------------------------------------------------------------------
        # Get unique basenames
        # ------------------------------------------------------------------------
        # e.g. base_folders = [alley_1", "alley_2", "ambush_2", ...]
        #substract_full_base = tools.cd_dotdot(all_img_filenames[0])

        self._image_list = []

        img_filenames = all_img_filenames

        for i in range(len(img_filenames) - 1):

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
        #basedir = os.path.splitext(os.path.dirname(im1_filename).replace(self._substract_base, "")[1:])[0]
        basedir = "lol"

        # example filename
        #basename = os.path.splitext(os.path.basename(im1_filename))[0]
        basename = "kek"

        example_dict = {
            "input1": im1,
            "input2": im2,
            "index": index,
            "basedir": basedir,
            "basename": self._image_list[index][0],
            "target1": flo,
            "target_occ1": occ
        }
        print('End load', 'id=' + str(cu_id), round(time() - tb, 3), file=a)
        a.close()

        return example_dict

    def __len__(self):
        return self._size

class AnyValid(_Sintel):
    def __init__(self, root):
        super(AnyValid, self).__init__(dir_root=root)

def init_inference(imgL_o, imgR_o, maxdisp, fac, modelpath):
    '''
    Return model and state dictionary
    '''

    # resize to 64X
    maxh = imgL_o.shape[0]
    maxw = imgL_o.shape[1]
    max_h = int(maxh // 64 * 64)
    max_w = int(maxw // 64 * 64)
    if max_h < maxh: max_h += 64
    if max_w < maxw: max_w += 64

    input_size = imgL_o.shape
    imgL = cv2.resize(imgL_o,(max_w, max_h))
    imgR = cv2.resize(imgR_o,(max_w, max_h))

    # load model
    model = VCN([1, max_w, max_h], md=[int(4*(maxdisp/256)),4,4,4,4], fac=fac)

    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    # load weights
    pretrained_dict = torch.load(modelpath)
    mean_L=pretrained_dict['mean_L']
    mean_R=pretrained_dict['mean_R']
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
    model.eval()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # for gray input images
    if len(imgL_o.shape) == 2:
        imgL_o = np.tile(imgL_o[:,:,np.newaxis],(1,1,3))
        imgR_o = np.tile(imgR_o[:,:,np.newaxis],(1,1,3))


    # flip channel, subtract mean
    imgL = imgL[:,:,::-1].copy() / 255. - np.asarray(mean_L).mean(0)[np.newaxis,np.newaxis,:]
    imgR = imgR[:,:,::-1].copy() / 255. - np.asarray(mean_R).mean(0)[np.newaxis,np.newaxis,:]
    imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
    imgR = np.transpose(imgR, [2,0,1])[np.newaxis]

    return model, pretrained_dict

def inference(imgL_o, imgR_o, maxdisp, fac, model, pretrained_dict):
    '''
    Return model
    '''
    tb = clock()

    # resize to 64X
    maxh = imgL_o.shape[0]
    maxw = imgL_o.shape[1]
    max_h = int(maxh // 64 * 64)
    max_w = int(maxw // 64 * 64)
    if max_h < maxh: max_h += 64
    if max_w < maxw: max_w += 64

    input_size = imgL_o.shape
    imgL = cv2.resize(imgL_o,(max_w, max_h))
    imgR = cv2.resize(imgR_o,(max_w, max_h))

    # load model
    #model = VCN([1, max_w, max_h], md=[int(4*(maxdisp/256)),4,4,4,4], fac=fac)

    #model = nn.DataParallel(model, device_ids=[0])
    #model.cuda()

    # load weights
    #pretrained_dict = torch.load(modelpath)
    mean_L=pretrained_dict['mean_L']
    mean_R=pretrained_dict['mean_R']
    #model.load_state_dict(pretrained_dict['state_dict'],strict=False)

    #print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # for gray input images
    if len(imgL_o.shape) == 2:
        imgL_o = np.tile(imgL_o[:,:,np.newaxis],(1,1,3))
        imgR_o = np.tile(imgR_o[:,:,np.newaxis],(1,1,3))


    # flip channel, subtract mean
    imgL = imgL[:,:,::-1].copy() / 255. - np.asarray(mean_L).mean(0)[np.newaxis,np.newaxis,:]
    imgR = imgR[:,:,::-1].copy() / 255. - np.asarray(mean_R).mean(0)[np.newaxis,np.newaxis,:]
    imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
    imgR = np.transpose(imgR, [2,0,1])[np.newaxis]

    # forward
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())
    with torch.no_grad():
        imgLR = torch.cat([imgL,imgR],0)

        rts = model(imgLR)
        torch.cuda.synchronize()
        pred_disp, entropy = rts

    # upsampling
    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()
    pred_disp = cv2.resize(np.transpose(pred_disp,(1,2,0)), (input_size[1], input_size[0]))
    pred_disp[:,:,0] *= input_size[1] / max_w
    pred_disp[:,:,1] *= input_size[0] / max_h
    flow = np.ones([pred_disp.shape[0],pred_disp.shape[1],3])
    flow[:,:,:2] = pred_disp
    entropy = torch.squeeze(entropy).data.cpu().numpy()
    entropy = cv2.resize(entropy, (input_size[1], input_size[0]))
    return flow, entropy




def main():
    #imgL_o = skimage.io.imread('./dataset/kitti_scene/testing/image_2/000042_10.png') # load two input images
    #imgR_o = skimage.io.imread('./dataset/kitti_scene/testing/image_2/000042_11.png')

    image_list = sorted(glob('./images/in/*'))
    imgL_o = io.imread(image_list[0])
    imgR_o = io.imread(image_list[1])

    maxdisp=256 # maximum disparity to search over (along each direction)
    fac=1       # shape of the search window -> 2 results in a horizontal rectangle

    modelpath = './weights/sintel-ft-trainval/finetune_67999.tar'
    model, pdict = init_inference(imgL_o, imgR_o, maxdisp, fac, modelpath)

    #validation_loader = DataLoader(
    #            AnyValid("images"),
    #            batch_size=1,
    #            shuffle=False,
    #            drop_last=False,
    #            **{"num_workers": 16, "pin_memory": False} )

    for i in range(0, len(image_list) - 1):
    #for example_dict in validation_loader:
        tb = clock()
        fname = image_list[i].split('/')[-1]
        #fname = example_dict['basename'][0].split('/')[-1]
        print(fname)
        #imgL_o = example_dict['input1']
        #print(imgL_o)
        #imgR_o = example_dict['input2']

        imgL_o = io.imread(image_list[i])
        imgR_o = io.imread(image_list[i + 1])
        flow, entropy = inference(imgL_o, imgR_o, maxdisp, fac, model, pdict)

        flow = flow.astype(np.float32)[:, :, :2]
        print("Iteration took", clock() - tb)
        write_flow(flow, './images/out/' + fname.split('.')[0] + '.flo')
        #io.imsave('./images/out/' + fname, flow_to_image(flow))

        imgL_o = imgR_o
        

main()
