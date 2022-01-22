python python -m torch.distributed.launch --nproc_per_node=8 main_moco_autoview.py   -a vit_small -b 256   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2   --multiprocessing-distributed --world-size 2 --rank 0   /mnt/data1/tangtao/imagenet


python -m torch.distributed.launch --nproc_per_node=8 main_moco_autoview.py  -a vit_small -b 32   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1  --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2   /mnt/data1/tangtao/imagenet