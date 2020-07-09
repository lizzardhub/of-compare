from common.file_move import *

arg_pass = open('arg_pass.txt', 'r')
START_IDX, STOP_IDX = map(int, arg_pass.readline().split())
print(START_IDX, STOP_IDX)
arg_pass.close()

optical_flow = True
disp = True

#lst = sorted(glob('vids/*_l*'))
#for i in [1, 11, 15, 20, 24, 25, 38, 82]:
for id in ['11_l1', '11_l2', '11_l3', '11_l4', '12_l1', '12_l2', '12_l3', '12_l4', , '13_l2', '16_l4']:
#for video_path in sorted(glob('vids/*_l*'))[START_IDX:STOP_IDX]:
    #video_path = lst[i]
    video_path = 'vids/' + id + '.mkv'
    # OPTICAL FLOW

    # Copy video
    subp_run_str('cp ' + video_path + ' cur_video.mkv')
    video_name = Path(video_path).name
    video_id = video_name[:3] + video_name[4] # 01_l0.mkv -> 01_0
    print("************************\n" * 3)
    print('Video id:', video_id, 'OPTICAL FLOW')
    if not os.path.exists(video_id): # Create folder for video results
        print('create dir', video_id)
        os.mkdir(video_id)
    else:
        shutil.rmtree(video_id)
        os.mkdir(video_id)

    #subp_bash('mkdir -p ' + video_id + '/{mono,irr,pwc,me,stereo,irr-disp,pwc-disp,me-disp}') # Debug
    subp_bash('mkdir -p ' + video_id + '/{mono,stereo}')

    if optical_flow:
        rm_tmp() # Delete previous video images
        frame_ffmpeg_split() # Split new video
        split_frames() # Copy frames to neural networks' inputs

        # Launch neuronets
        subp_bash('cd pwc; python run.py')
        subp_bash('cd irr; python run.py')
        subp_bash('cd me; python run.py')
        #!cd pwc; python run.py
        #!cd irr; python run.py
        #!cd me; python run.py

        flow_metrics()
        subp_run_str('mv flow_video.mp4 ' + video_id + '/flow.mp4')
        subp_run_str('mv metrics.txt ' + video_id + '/flow.txt')
        subp_bash('mv *.jpg ' + video_id + '/mono/') # Debug


        #subp_run_str('cp frames/* ' + video_id + '/mono') # Debug
        #subp_run_str('cp sintelall/MPI-Sintel-complete/training/frames/out/* ' + video_id + '/irr')
        #subp_run_str('cp pwc/images/out/* ' + video_id + '/pwc')
        #subp_run_str('cp me/images/out/* ' + video_id + '/me')
    if disp:
        # Swap flow and disparity folders
        #subp_run_str('mv selflow selflow-of')
        subp_run_str('mv pwc pwc-of')
        subp_run_str('mv me me-of')
        subp_run_str('mv irr irr-of')
        #subp_run_str('mv selflow-disp selflow')
        subp_run_str('mv pwc-disp pwc')
        subp_run_str('mv me-disp me')
        subp_run_str('mv irr-disp irr')

        video_name = video_name[:3] + 'r' + video_name[4:] # 01_l0.mkv -> 01_r0.mkv
        subp_run_str('cp vids/' + video_name + ' cur_video2.mkv')
        print("************************\n" * 3) # DISPARITY
        print('Video id:', video_id, 'DISPARITY')
        if not Path(video_id).exists():
            print('create dir', video_id)
            os.mkdir(video_id)
        #else: # Delete previously copied work
        #    shutil.rmtree(video_id)
        #    os.mkdir(video_id)



        rm_tmp() # Delete previous video images
        frame_ffmpeg_split_stereo() # Split new video
        split_frames(stereo=True) # Copy frames to neural networks' inputs

        # Launch neuronets
        subp_bash('cd pwc; python run.py')
        subp_bash('cd irr; python run.py')
        subp_bash('cd me; python run.py')
        #!cd pwc; python run.py
        #!cd irr; python run.py
        #!cd me; python run.py

        flow_metrics(stereo=True)
        subp_run_str('mv flow_video.mp4 ' + video_id + '/disp.mp4')
        subp_run_str('mv metrics.txt ' + video_id + '/disp.txt')
        subp_bash('mv *.jpg ' + video_id + '/stereo/') # Debug

        #subp_run_str('cp frames/* ' + video_id + '/stereo') # Debug
        #subp_run_str('cp sintelall/MPI-Sintel-complete/training/frames/out/* ' + video_id + '/irr-disp')
        #subp_run_str('cp pwc/images/out/* ' + video_id + '/pwc-disp')
        #subp_run_str('cp me/images/out/* ' + video_id + '/me-disp')

        # Swap flow and disparity folders again
        #subp_run_str('mv selflow selflow-disp')
        subp_run_str('mv pwc pwc-disp')
        subp_run_str('mv me me-disp')
        subp_run_str('mv irr irr-disp')
        #subp_run_str('mv selflow-of selflow')
        subp_run_str('mv pwc-of pwc')
        subp_run_str('mv me-of me')
        subp_run_str('mv irr-of irr')
    subp_run_str('zip -r ' + video_id + '.zip ' + video_id)
    subp_run_str('curl -T ' + video_id + '.zip ftp://staro.drevo.si:8021/')
