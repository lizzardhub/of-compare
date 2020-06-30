modelname=sintel-ft-trainval
i=67999
CUDA_VISIBLE_DEVICES=0 python submission.py --dataset sintel --datapath /content --outdir ./weights/$modelname/ --loadmodel ./weights/$modelname/finetune_$i.tar  --maxdisp 448 --fac 1.4
python eval_tmp.py --path ./weights/$modelname/ --vis no --dataset sintel
