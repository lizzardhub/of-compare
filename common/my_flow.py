from .all_imports import *
from .utils.flowlib import write_flow
import cv2

TAG_CHAR = np.array([202021.25], np.float32)
UNKNOWN_FLOW_THRESH = 1e7

random.seed(datetime.now())
np.random.seed(0)


def flow_to_png(flow_map, max_value=None):
    _, h, w = flow_map.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[:, :, 0] += normalized_flow_map[0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:, :, 2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)



def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def flow_to_png_middlebury(flow, rad_clip=999):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """

    #flow = flow.transpose([1, 2, 0])
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    #UNKNOWN_FLOW_THRESH = 100
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    maxrad = min(maxrad, rad_clip)

    clip_mask = rad > maxrad
    u[clip_mask] *= np.full(rad[clip_mask].shape, maxrad) / rad[clip_mask]
    v[clip_mask] *= np.full(rad[clip_mask].shape, maxrad) / rad[clip_mask]
    #print(maxrad)
    u = u / maxrad
    v = v / maxrad

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def read_flo(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=int(2*w*h))
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h[0], w[0],2))
            return data2D

# *** New functions

def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    grids_cuda = grid.float().requires_grad_(False)
    return grids_cuda

def warp(x, flow):
    height_im, width_im, _c = x.shape
    div_flow = 1
    x = torch.from_numpy(x[np.newaxis, ...].astype(np.float32)).transpose(2, 3).transpose(1, 2)
    flow = torch.from_numpy(flow[np.newaxis, ...]).transpose(2, 3).transpose(1, 2)

    flo_list = []
    flo_w = flow[:, 0] * 2 / max(width_im - 1, 1) / div_flow
    flo_h = flow[:, 1] * 2 / max(height_im - 1, 1) / div_flow
    flo_list.append(flo_w)
    flo_list.append(flo_h)
    flow_for_grid = torch.stack(flo_list).transpose(0, 1)
    grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)
    x_warp = tf.grid_sample(x, grid)

    mask = torch.ones(x.size(), requires_grad=False)
    mask = tf.grid_sample(mask, grid)
    mask = (mask >= 0.9999).float()

    return np.nan_to_num((x_warp * mask).transpose(1, 2).transpose(2, 3).numpy()[0])

def photometric_diff(im1, im2_warped):
    euclidian_dist = np.sqrt( np.sum(np.square(im1 - im2_warped), axis=2) )
    return np.sum(euclidian_dist) / (im1.shape[0] * im1.shape[1])

def grad_sum(flow):
    sx = ndimage.sobel(flow[:, :, 0], axis=0, mode='constant') # Vertical gradient
    sy = ndimage.sobel(flow[:, :, 1], axis=1, mode='constant') # Horizontal gradient
    return np.sum(np.sqrt(sx ** 2 + sy ** 2)) / (flow.shape[0] * flow.shape[1])

def weighted_grad_sum(im1, flow):
    # Flow gradient
    sx = ndimage.sobel(flow[:, :, 0], axis=0, mode='constant') # Vertical gradient
    sy = ndimage.sobel(flow[:, :, 1], axis=1, mode='constant') # Horizontal gradient
    flow_grad = np.sqrt(sx ** 2 + sy ** 2)
    # Image1 gradient
    gray_im1 = rgb2gray(im1)
    sx = ndimage.sobel(gray_im1, axis=0, mode='constant') # Vertical gradient
    sy = ndimage.sobel(gray_im1, axis=1, mode='constant') # Horizontal gradient
    im1_grad = np.sqrt(sx ** 2 + sy ** 2)

    return np.sum( flow_grad * np.exp(-im1_grad) ) / (flow.shape[0] * flow.shape[1])

def occlusion_area(flow_f, b_warped):
    res = flow_f + b_warped
    res = (res[:, :, 0] ** 2 + res[:, :, 1] ** 2) ** 0.5
    return np.sum(res) / (flow_f.shape[0] * flow_f.shape[1])

def lrc_weighted_photo(im1, flow_f, b_warped, im2_warped):
    res = flow_f + b_warped
    res = (res[:, :, 0] ** 2 + res[:, :, 1] ** 2) ** 0.5
    res = np.exp( -res ) # Regions of trust

    euclidian_dist = np.sqrt( np.sum(np.square(im1 - im2_warped), axis=2) )
    return np.sum(euclidian_dist * res) / (im1.shape[0] * im1.shape[1])



'''def weight_sum(flow, img):
    sx = ndimage.sobel(flow[:, :, 0], axis=0, mode='constant')
    sy = ndimage.sobel(flow[:, :, 1], axis=1, mode='constant')
    flow_w = np.sqrt(sx ** 2 + sy ** 2)
    img = rgb2gray(img)
    sx = ndimage.sobel(img, axis=0, mode='constant')
    sy = ndimage.sobel(img, axis=1, mode='constant')
    img_w = np.sqrt(sx ** 2 + sy ** 2)'''

def generate_perlin_noise_2d(shape, res):
    # Function author: https://github.com/pvigier/perlin-numpy

    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def perlin_noise(shape, scale):
    max_axis = (max(shape[0], shape[1]) // scale + 2) * scale
    noise = generate_perlin_noise_2d((max_axis, max_axis), (scale, scale))
    return noise[:shape[0], :shape[1]]

def distortion(img, params=[]):
    if not params: # Generate random parameters
        #mode = random.randrange(0, 3)
        mode = 2
        params.append(mode)
        pass_params = False
    else: # Use passed in parameters
        mode = params[0]
        pass_params = True

    if mode == 0: # Gaussian. params = [0, sigma]
        if pass_params:
            sigma = params[1]
        else:
            sigma = 1.5 + random.random() * 1.5
            params.append(sigma)
        img = filters.gaussian(img, sigma=sigma)
        return (img, params)
    elif mode == 1: # Perlin noise. params = [1, scale, intensity, random_noises]
        if pass_params:
            scale = params[1]
            intensity = params[2]
            random_noises = params[3]
        else:
            if random.randrange(0, 2) == 0:
                scale = 8 # Large spots
                intensity = 0.1
            else:
                scale = 512 # Fine-grained
                intensity = 0.2
            random_noises = [0] * 15
            for i in range(15):
                random_noises[i] = perlin_noise(img.shape, scale) * intensity

            params.append(scale)
            params.append(intensity)
            params.append(random_noises)
        if scale == 8:
            np.random.seed(0)
        noise = random_noises[np.random.randint(15)]
        img[:, :, 0] += noise
        noise = random_noises[np.random.randint(15)]
        img[:, :, 1] += noise
        noise = random_noises[np.random.randint(15)]
        img[:, :, 2] += noise
        img = np.clip(img, 0, 1)
        return (img, params)
    else: # No distortions
        return (img, [2])

def find_black_frame(img):
    img = rgb2gray(img)
    h, w = img.shape
    bounds = [0, h - 1, 0, w - 1] # y1:y2, x1:x2
    THRESH = 10.0

    for i in range(h // 2):
        if np.sum( img[i, :] ) < THRESH:
            bounds[0] = i + 1

    for i in range(h - 1, h // 2, -1):
        if np.sum( img[i, :] ) < THRESH:
            bounds[1] = i - 1

    for j in range(w // 2):
        if np.sum( img[:, j] ) < THRESH:
            bounds[2] = j + 1

    for j in range(w - 1, w // 2, -1):
        if np.sum( img[:, j] ) < THRESH:
            bounds[3] = j - 1

    return bounds

def split_frames(stereo=False):
    print("Hello!")
    max_area = 1240 * 436 # A Sintel-sized frame

    # Move from frames_l and frames_r to frames
    files_l = sorted(glob('frames_l/*'))
    files_r = sorted(glob('frames_r/*'))
    for i in range( 3, min(5, len(files_l)) ):
        target_fname_l = 'frames/frame_' + str(i * 2 + 1).zfill(4) + '.jpg'
        target_fname_r = 'frames/frame_' + str(i * 2 + 2).zfill(4) + '.jpg'
        shutil.copy(files_l[i], target_fname_l)
        shutil.copy(files_r[i], target_fname_r)

    # Process all frames
    for i, filepath in enumerate(sorted(glob('frames/*'))):
        if i >= 30:
            os.remove(filepath)
            continue
        if not stereo:
            if i < 3 or i >= 5:
                os.remove(filepath)
                continue

    img_test = sorted(glob('frames/*'))
    bounds = find_black_frame(imread(img_test[0]))

    params = []
    for i, filepath in enumerate(sorted(glob('frames/*'))):
        print(filepath)
        img = skimage.img_as_float(imread(filepath))[bounds[0]:bounds[1], bounds[2]:bounds[3]]

        #img = img[::2, ::2]
        h, w, _c = img.shape
        img = cv2.resize(img, (w // 2, h // 2))

        if not(stereo and i % 2 == 1):
            img, params = distortion(img, params)
        if i == 0:
            print('Distortion params after func:', params)

        print(Path(filepath).suffix)
        if Path(filepath).suffix == '.jpg':
            imsave(filepath, img, quality=100)
        else:
            imsave(filepath, img)

    # Copy frames into neuronets
    for i, filepath in enumerate(sorted(glob('frames/*'))):
        filename = filepath.split('/')[-1]
        #shutil.copy(filepath, 'vcn/images/in/' + filename)
        shutil.copy(filepath, 'pwc/images/in/' + filename)
        shutil.copy(filepath, 'sintelall/MPI-Sintel-complete/training/frames/in/' + filename)
