import subprocess
from glob import glob
import shutil

process = subprocess.Popen('cd scripts/validation; bash IRR-PWC_sintel.sh', shell=True, stdout=subprocess.PIPE)
process.wait()

print(process.returncode)

for filepath in sorted(glob('/content/irr/saved_check_point/pwcnet/eval_temp/IRR_PWC/flo/frames/in/*')):
    filename = filepath.split('/')[-1]
    #shutil.copyfile(filepath, '/content/sintelall/MPI-Sintel-complete/training/frames/out/frame_' + filename.split('_')[1] + '.png')
    shutil.copyfile(filepath, '/content/sintelall/MPI-Sintel-complete/training/frames/out/' + filename)
