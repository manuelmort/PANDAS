#!/bin/bash
#$ -l h_rt=48:00:00
#$ -pe omp 4
#$ -P ec500kb
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=40G
#$ -N GTP_train
#$ -j y
#$ -o logs/train_$JOB_ID.log

# Load modules
module load python3/3.10.12
module load cuda/11.8
module load gcc/9.3.0

# Activate environment
source /projectnb/ec500kb/projects/Project_1_Team_1/panda_env/bin/activate

# Go to working directory
cd /projectnb/ec500kb/projects/Project_1_Team_1/Official_GTP_PANDAS/PANDAS

mkdir -p logs

echo "=== JOB INFO ==="
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# DO NOT override CUDA_VISIBLE_DEVICES - let scheduler handle it
export PYTHONUNBUFFERED=1

# Run training
python -u main.py \
    --n_class 3 \
    --n_features 512 \
    --data_path './feature_extractor/graphs_simclr/panda' \
    --train_set './scripts/train_set.txt' \
    --val_set './scripts/val_set.txt' \
    --model_path './graph_transformer/saved_models/' \
    --log_path './graph_transformer/runs/' \
    --task_name 'simclr_transformer_model' \
    --batch_size 8 \
    --train \
    --site panda

echo ""
echo "=== JOB COMPLETE ==="
echo "Finished at: $(date)"