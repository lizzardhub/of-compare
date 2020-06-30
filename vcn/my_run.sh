modelname=sintel-ft-trainval
i=67999
CUDA_VISIBLE_DEVICES=0 python submission.py --dataset sintel --datapath /content/vcnsintel/   --outdir ./weights/sintel-ft-trainval/ --loadmodel ./weights/sintel-ft-trainval/finetune_67999.tar  --maxdisp 448 --fac 1.4
python eval_tmp.py --path ./weights/sintel-ft-trainval/ --vis no --dataset sintel