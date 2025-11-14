#!/bin/bash
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -W group_list=gj40
#PBS -j oe
#------- Program execution -------#
export HOME=/work/gj40/j40001
source $HOME/anaconda3/bin/activate diffusion
# Change to the submission directory
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
# Launch the worker script on each allocated node using pbsdsh
python train_w_pipe.py   --task data_process   --dataset_path /work/gj40/j40001/code/image-to-video-edit/dataset_root_image_path   --output_path ./models   --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"   --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"   --tiled   --num_frames 81   --height 480   --width 832 --dataloader_num_workers 2
# python senorita_loader.py   --task data_process --output_path ./data/sytle --tiled   --num_frames 10   --height 480   --width 832 --dataloader_num_workers 0