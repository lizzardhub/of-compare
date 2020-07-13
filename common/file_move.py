from .all_imports import *
from .my_flow import *

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

def vis(im1, im2, fwd, bwd, meth_str):
    #fwd = fwp
    #bwd = bwp
    #im1 = io.imread(im1p)
    #im2 = io.imread(im2p)
    h, w = im1.shape[:2]

    io.imsave(meth_str + 'im1.jpg', im1, quality=100)
    io.imsave(meth_str + 'im2.jpg', im2, quality=100)

    # Read forward
    flow_f = fwd
    #if method == 'me-disp':
    #    flow_f /= 4
    img = flow_to_png_middlebury(flow_f, rad_clip=25)
    io.imsave(meth_str + 'fw.jpg', img, quality=100)

    # Read backward
    flow_b = bwd
    #if method == 'me-disp':
    #    flow_b /= 4
    img = flow_to_png_middlebury(flow_b, rad_clip=25)
    io.imsave(meth_str + 'bw.jpg', img, quality=100)

    # Warp
    im2_warped = backwarp(im2, flow_f)
    io.imsave(meth_str + 'im2w.jpg', im2_warped.astype(np.uint8), quality=100)


    # FORWARD WARPING
    wf = warpforw(flow_f)
    wb = warpforw(flow_b)
    wb_crit = wb < 0.95

    # Consistency error
    b_warped = backwarp(flow_b, flow_f) # -> forward occlusions (closure areas)
    res = b_warped + flow_f
    mag = np.sqrt(res[:, :, 0] ** 2 + res[:, :, 1] ** 2)

    MIN_ERR = 0.5
    MAX_ERR = 5
    lrc_crit = np.clip((mag - MIN_ERR) / (MAX_ERR - MIN_ERR), 0, 1)

    mask = np.zeros((h, w, 3))
    mask[:, :, 0] = wb_crit*4
    img = im1 / 255 * 0.5 + mask * 0.5
    io.imsave(meth_str + 'wb.jpg', img, quality=100)

    mask = np.zeros((h, w, 3))
    mask[:, :, 0] = lrc_crit*4
    img = im1 / 255 * 0.5 + mask * 0.5
    io.imsave(meth_str + 'lrc.jpg', img, quality=100)

    lrc_crit[wb_crit] = 0
    prec_metric = np.sum(lrc_crit) / (h * w)

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
        fimg1 = input_path / Path(img_list[i]).name
        fname1 = Path(fimg1.name)

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
        filename = fname1.with_suffix('.flo')
        filename_b = Path(fname1.stem + '_b').with_suffix('.flo')
        flows = [0] * 4
        flows_b = [0] * 4
        max_rad_me = 1000
        for meth in range(1, 4): # Pre-calculate maximum flow
            if (res_path[meth] / filename).exists(): # Flow file with the image name?
                flows[meth] = read_flo(res_path[meth] / filename) # IMPORTANT: read_flo
                flows_b[meth] = read_flo(res_path[meth] / filename_b) # IMPORTANT: read_flo
                if meth == 2:
                    flows[meth] /= 4
                    flows_b[meth] /= 4

                meth_max = np.max( np.sqrt(flows[meth][:, :, 0] ** 2 + flows[meth][:, :, 1] ** 2) )
                max_rad_me = min(max_rad_me, meth_max)
            else:
                flows[meth] = np.zeros((h, w, 2), dtype=np.float32)
                flows_b[meth] = np.zeros((h, w, 2), dtype=np.float32)

        common_photo = confident_photo(img1, img2, flows[1:], flows_b[1:])
        for meth in range(1, 4):
            if (res_path[meth] / filename).exists():
                flow = flows[meth]
                flow_b = flows_b[meth]
                img2_warped = warp(img2, flow)
                b_warped = warp(flow_b, flow)

                #diff = photometric_diff(img1, img2, flow)
                #flow_grad = grad_sum(flow)

                photo = photometric_diff(img1, img2_warped)
                #lrc = occlusion_area(flow, b_warped)
                precision = occ_precision(flow, flow_b)

                metrics_history[meth, 0, i] = photo
                metrics_history[meth, 1, i] = common_photo[meth - 1]
                metrics_history[meth, 2, i] = precision

                res_canvas[meth][...] = load_and_caption( # IMPORTANT: flow_to_png_middlebury
                        flow_to_png_middlebury(flow, rad_clip=max_rad_me), caption[meth])
                #if i == (len(img_list) // 2) // step * step: # So that stereo frames also getcopied
                vis(img1, img2, flow, flow_b, caption[meth] + '_' + str(i).zfill(4) + '_')

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
