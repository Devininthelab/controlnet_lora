#!/bin/bash
## Set job parameters

## Job Name
#PBS -N VOX

## Project Funding Code 
#PBS -P gs_slab_h100

## Queue Name
#PBS -q gpu_a40

## Output and error log files
#PBS -o output_vox.log
#PBS -e error_vox.log

## Specify walltime in HH:MM:SS (Set based on your expected runtime)
#PBS -l walltime=72:00:00

## Request 1 node with 1 A40 GPU
#PBS -l select=1:ncpus=8:ngpus=1:mem=100GB

## Load necessary modules (Modify if your cluster uses a different setup)
# module load anaconda # Ensure Anaconda is available on your system

## Activate Conda environment
source activate myenv2

## Move to the project directory
cd $PBS_O_WORKDIR

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DATA_DIR="./sample_data/artistic-custom"
export OUTPUT_DIR="./runs/artistic_custom"

accelerate launch --mixed_precision="no" train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$TRAIN_DATA_DIR \
  --caption_column="text" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 \
  --validation_epochs 1 \
  --checkpointing_steps=2000 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --checkpoints_total_limit 2 \
  --validation_prompt="a house" \
  --report_to="wandb" \