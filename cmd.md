### Vit-B

```
{"arch": "vit_base", "patch_size": 16, "out_dim": 65536, "norm_last_layer": true, "warmup_teacher_temp": 0.04, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 50, "use_fp16": false, "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 0.3, "batch_size_per_gpu": 32, "epochs": 400, "freeze_last_layer": 3, "lr": 0.00075, "warmup_epochs": 10, "min_lr": 2e-06, "global_crops_scale": [0.25, 1.0], "local_crops_scale": [0.05, 0.25], "local_crops_number": 10, "seed": 0, "num_workers": 10, "world_size": 32, "ngpus": 8, "nodes": 4, "optimizer": "adamw", "momentum_teacher": 0.996, "use_bn_in_head": false, "drop_path_rate": 0.1}


4x8
python -m torch.distributed.launch --nproc_per_node=8 main_dino_autoview.py --arch vit_base --epochs 400 --teacher_temp 0.07 --warmup_teacher_temp_epochs 50 --local_crops_number 10 --global_crops_scale 0.25 1.0 --local_crops_scale 0.05 0.25 --batch_size_per_gpu 32 --clip_grad 0.3 --min_lr 2e-06  --freeze_last_layer 3 --lr 0.00075 --use_fp16 false


```


