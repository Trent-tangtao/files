# pkill python
# sleep 30s

#python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500\
#        main_ibot.py \
#        --arch vit_small \
#        --output_dir ibot_test \
#        --data_path /mnt/data1/tangtao/imagenet/train \
#        --batch_size_per_gpu 64 \
#        --local_crops_number 8 \
#        --saveckp_freq 10\
#        --shared_head true \
#        --out_dim 8192

#python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500\
#        main_ibot_autoview.py \
#        --arch vit_small \
#        --output_dir ibot_test \
#        --data_path /mnt/data1/tangtao/imagenet/train \
#        --batch_size_per_gpu 32 \
#        --local_crops_number 8 \
#        --saveckp_freq 10\
#        --shared_head true \
#        --out_dim 8192


python -m torch.distributed.launch --nproc_per_node=8  --master_port=29500 \
                 run_class_finetuning.py \
                --finetune backbone_weights.pth\
                --model vit_small \
                --epochs 200 \
                --warmup_epochs 20 \
                --layer_decay 0.75 \
                --mixup 0.8 \
                --cutmix 1.0 \
                --layer_scale_init_value 0.0 \
                --disable_rel_pos_bias \
                --abs_pos_emb \
                --use_cls \
                --imagenet_default_mean_and_std \
                --output_dir finetune-300-base \
                --data_path /data/ImageNet\
                --batch_size 256\
                --lr 1e-3


python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500\
                run_class_finetuning.py \
                --finetune $WEIGHT_FILE \
                --model vit_base \
                --epochs 100 \
                --warmup_epochs 20 \
                --layer_decay 0.65 \
                --mixup 0.8 \
                --cutmix 1.0 \
                --layer_scale_init_value 0.0 \
                --disable_rel_pos_bias \
                --abs_pos_emb \
                --use_cls \
                --imagenet_default_mean_and_std \
                --output_dir $SUB_OUTPUT_DIR \
                --data_path data/imagenet \



python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
        --data-path /data/ILSVRC/Data/CLS-LOC/train \
        --output_dir /data/output_for_dino_finetuning \
        --model deit_base_patch16_224 --lr 5e-4 --init_scale 0.001 --batch-size 128 \
        --finetune /data/ckpts/dino_vitbase16_pretrain.pth --no-model-ema


python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --batch-size 128 --data_path /data/ImageNet --output_dir /data/tangtao/datasets/deit/output --lr 5e-4 --finetune /data/tangtao/datasets/checkpoint300-0.7.pth --no-model-ema