#!/bin/bash
## Set job parameters

## Job Name
#PBS -N VOX_2

## Project Funding Code 
#PBS -P gs_slab_h100

## Queue Name
#PBS -q gpu_a40

## Output and error log files
#PBS -o output_vox_2.log
#PBS -e error_vox_2.log

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
export INSTANCE_DIR="./sample_data/dreambooth-cat"
export OUTPUT_DIR="./runs/dreambooth_cat"

accelerate launch --mixed_precision="no" train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a sks cat" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks cat in a bucket" \
  --validation_epochs=50 \
  --checkpoints_total_limit 2 \
  --seed="0"