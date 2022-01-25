python3 extract_backbone_weights.py /mnt/data1/tangtao/ibot/dino_vitsmall16_300ep_pretrain.pth  /mnt/data1/tangtao/ibot/backbone_weights.pth --checkpoint_key teacher

python3 -m torch.distributed.launch --nproc_per_node=8 \
                --master_port=29500 \
                evaluation/object_detection/train.py \
                evaluation/object_detection/configs/cascade_rcnn/vit_small_giou_4conv1f_coco_3x.py \
                --launcher pytorch \
                --work-dir coco_det_test\
                --deterministic \
                --cfg-options model.backbone.use_checkpoint=True \
                data.samples_per_gpu=4 \
                lr_config.step=8,11 \
                runner.max_epochs=12 \
                optimizer.paramwise_cfg.layer_decay_rate=0.8\
                model.pretrained=/mnt/data1/tangtao/ibot/backbone_weights.pth


python3 -m torch.distributed.launch --nproc_per_node=8 \
                --master_port=29500 \
                evaluation/object_detection/train.py \
                evaluation/object_detection/configs/cascade_rcnn/vit_small_giou_4conv1f_coco_3x.py \
                --launcher pytorch \
                --work-dir coco_det_test\
                --deterministic \
                --cfg-options model.backbone.use_checkpoint=True \
                model.pretrained=/mnt/data1/tangtao/ibot/backbone_weights.pth

python3 -m torch.distributed.launch --nproc_per_node=8 \
                --master_port=29500 \
                evaluation/object_detection/test.py \
                evaluation/object_detection/configs/cascade_rcnn/vit_small_giou_4conv1f_coco_3x.py \
                coco_det_test/latest.pth \
                --launcher pytorch \
                --eval bbox segm \
                --cfg-options model.backbone.use_checkpoint=True


python3 -m torch.distributed.launch --nproc_per_node=8 \
                --master_port=29500 \
                evaluation/object_detection/train.py \
                evaluation/object_detection/configs/cascade_rcnn/vit_small_giou_4conv1f_coco_3x.py \
                --launcher pytorch \
                --work-dir coco_det_test\
                --deterministic \
                --cfg-options model.backbone.use_checkpoint=True \
                --resume-from coco_det_test/latest.pth


'
git clone -b v0.12.0 https://github.com/open-mmlab/mmsegmentation
cd mmsegmentation
pip3 install -v -e .
cd ..
or simply:

pip3 install mmsegmentation==0.12.0
'

python3 extract_backbone_weights.py /mnt/data1/tangtao/ibot/dino_vitsmall16_300ep_pretrain.pth  /mnt/data1/tangtao/ibot/backbone_weights.pth --checkpoint_key teacher

python3 -m torch.distributed.launch --nproc_per_node=8 \
                --master_port=29500 \
                train.py \
                configs/upernet/vit_small_512_ade20k_160k.py \
                --launcher pytorch \
                --work-dir ade20k_seg_test \
                --deterministic \
                --options model.pretrained=/data/tangtao/datasets/ibot/backbone_weights.pth

python3 -m torch.distributed.launch --nproc_per_node=8 \
                --master_port=29500 \
                semantic_segmentation/test.py \
                semantic_segmentation/configs/upernet/vit_small_512_ade20k_160k.py \
                ade20k_seg_test/iter_160000.pth \
                --launcher pytorch \
                --eval mIoU


python3 -m torch.distributed.launch --nproc_per_node=8 \
                --master_port=29500 \
                evaluation/semantic_segmentation/test.py \
                evaluation/semantic_segmentation/configs/upernet/vit_small_512_ade20k_160k.py \
                evaluation/semantic_segmentation/ade20k_seg_test/iter_50.pth \
                --launcher pytorch \
                --eval mIoU




# 800
python -m torch.distributed.launch main_ibot.py \
        --arch vit_small \
        --output_dir ibot_test \
        --data_path data/imagenet/train \
        --teacher_temp 0.07 \
        --warmup_teacher_temp_epochs 30 \
        --norm_last_layer false \
        --epochs 800 \
        --batch_size_per_gpu 64 \
        --shared_head true \
        --out_dim 8192 \
        --local_crops_number 10 \
        --global_crops_scale 0.25 1 \
        --local_crops_scale 0.05 0.25 \
        --pred_ratio 0 0.3 \
        --pred_ratio_var 0 0.2
# 100
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500\
        main_ibot.py \
        --arch vit_small \
        --output_dir ibot_test \
        --data_path /mnt/data1/tangtao/imagenet/train \
        --batch_size_per_gpu 64 \
        --local_crops_number 8 \
        --saveckp_freq 10\
        --shared_head true \
        --out_dim 8192


python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500\
        main_ibot_autoview.py \
        --arch vit_small \
        --output_dir ibot_test \
        --data_path /mnt/data1/tangtao/imagenet/train \
        --batch_size_per_gpu 32 \
        --local_crops_number 8 \
        --saveckp_freq 10\
        --shared_head true \
        --out_dim 8192

