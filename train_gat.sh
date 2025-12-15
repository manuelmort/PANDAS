#!/bin/bash
#$ -l h_rt=48:00:00
#$ -pe omp 4
#$ -P ec500kb
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=16G
#$ -N GAT_train
#$ -j y
#$ -o logs/gat_train_$JOB_ID.log

# Load modules
module load python3/3.10.12
module load cuda/11.8
module load gcc/9.3.0

# Activate environment
source /projectnb/ec500kb/projects/Project_1_Team_1/panda_env/bin/activate

# Go to working directory
cd /projectnb/ec500kb/projects/Project_1_Team_1/Official_GTP_PANDAS/PANDAS

mkdir -p logs
mkdir -p ./graph_transformer/saved_models/gat

echo "=== JOB INFO ==="
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# DO NOT override CUDA_VISIBLE_DEVICES - let scheduler handle it
export PYTHONUNBUFFERED=1

# Run GAT training with Phikon featuresscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
python -u main_gat.py \
    --n_class 3 \
    --n_features 768 \
    --hidden_dim 64 \
    --heads 4 \
    --dropout 0.1 \
    --data_path './feature_extractor/graphs_phikon/panda' \
    --train_set './scripts/train_set.txt' \
    --val_set './scripts/val_set.txt' \
    --model_path './graph_transformer/saved_models/gat/' \
    --log_path './graph_transformer/runs/' \
    --task_name 'gat_phikon_model' \
    --batch_size 8 \
    --num_epochs 50 \
    --lr 0.0001 \
    --train \
    --site panda

echo ""
echo "=== JOB COMPLETE ==="
echo "Finished at: $(date)"