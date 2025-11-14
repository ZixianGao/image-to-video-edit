#!/bin/bash
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gj40
#PBS -j oe
#------- Program execution -------#
export HOME=/work/gj40/j40001
source $HOME/anaconda3/bin/activate diffusion
# Change to the submission directory
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
# Launch the worker script on each allocated node using pbsdsh
python inference_w_qwen.py