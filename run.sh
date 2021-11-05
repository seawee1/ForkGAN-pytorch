# Standard Experiment
#python -m torch.distributed.launch --nproc_per_node=8 train.py --dataroot ./datasets/dataset --name model_name --model fork_gan --load_size 512 --crop_size 512 --preprocess scale_height_and_crop --input_nc 1 --output_nc 1 --display_freq 100 --batch_size 8 --netD ms3 --lambda_identity 0.0 --num_threads 16 --lr 0.0006 --n_epochs 10 --n_epochs_decay 10 --save_latest_freq 1000 --display_freq 100 --display_id -1 --continue_train
#python -m torch.distributed.launch --nproc_per_node=8 train.py --dataroot ./datasets/dataset --name model_name --model fork_gan --load_size 512 --crop_size 512 --preprocess scale_height_and_crop --input_nc 1 --output_nc 1 --display_freq 100 --batch_size 8 --netD ms3 --lambda_identity 0.0 --num_threads 16 --lr 0.0001 --n_epochs 10 --n_epochs_decay 10 --save_latest_freq 1000 --display_freq 100 --display_id -1 --norm none --save_epoch_freq 1 --continue_train

# Train (no inst)
#python -m torch.distributed.launch --nproc_per_node=8 train.py --dataroot ./datasets/dataset --name model_name --model fork_gan --load_size 512 --crop_size 512 --preprocess scale_height_and_crop --input_nc 1 --output_nc 1 --display_freq 100 --batch_size 8 --netD ms3 --lambda_identity 0.0 --num_threads 16 --lr 0.0001 --n_epochs 10 --n_epochs_decay 10 --save_latest_freq 1000 --display_freq 100 --display_id -1 --norm none --save_epoch_freq 1 --continue_train

# Train (inst basic)
#python -m torch.distributed.launch --nproc_per_node=1 train.py --dataroot ./datasets/dataset --name model_name --model fork_gan --load_size 512 --crop_size 512 --preprocess scale_height_and_crop --input_nc 1 --output_nc 1 --display_freq 100 --batch_size 1 --netD ms3 --lambda_identity 0.0 --num_threads 4 --lr 0.0001 --n_epochs 10 --n_epochs_decay 10 --save_latest_freq 1000 --display_freq 100 --display_id -1 --norm none --save_epoch_freq 1 --continue_train --dataset_mode unaligned_coco --instance_level --continue_train --epoch latest --coco_imagedir path/to/coco/imagedir

# Test
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 test.py --dataroot ./datasets/dataset --name model_name --model fork_gan --load_size 512 --crop_size 512 --preprocess scale_height --input_nc 1 --output_nc 1 --netD ms3 --norm none --coco_imagedir path/to/coco/imagedir --dataset_mode unaligned_coco --batch_size 8 --epoch latest --num_threads 8

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 test.py --dataroot ./datasets/dataset --name model_name --model fork_gan --load_size 512 --crop_size 512 --preprocess scale_height --input_nc 1 --output_nc 1 --netD ms3 --norm none --coco_imagedir path/to/coco/imagedir --dataset_mode unaligned_coco --batch_size 8 --epoch latest --num_threads 8 --results_dir results_inst
