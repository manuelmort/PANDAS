#!/bin/bash
# GTP/GAT Docker Helper Script
# Usage: ./docker_run.sh [command]

set -e

IMAGE_NAME="gtp-panda"
IMAGE_TAG="latest"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_help() {
    echo "GTP/GAT Docker Helper Script"
    echo ""
    echo "Usage: ./docker_run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  inference   Run inference with GAT model"
    echo "  train-gat   Train GAT model"
    echo "  train-gtp   Train Graph Transformer model"
    echo "  evaluate    Evaluate GAT model"
    echo "  shell       Open interactive shell in container"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./docker_run.sh build"
    echo "  ./docker_run.sh inference"
    echo "  ./docker_run.sh shell"
}

build() {
    echo -e "${GREEN}Building Docker image...${NC}"
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    echo -e "${GREEN}Done! Image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
}

run_inference() {
    echo -e "${GREEN}Running inference with GAT...${NC}"
    docker run --gpus all \
        -v $(pwd)/data/graphs:/data/graphs:ro \
        -v $(pwd)/data/output:/data/output \
        -v $(pwd)/weights:/app/weights:ro \
        -v $(pwd)/scripts:/app/scripts:ro \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        python inference.py \
            --input /data/graphs \
            --output /data/output/predictions.csv \
            --model gat \
            --backbone phikon
}

train_gat() {
    echo -e "${GREEN}Training GAT model...${NC}"
    docker run --gpus all \
        -v $(pwd)/data/graphs:/data/graphs:ro \
        -v $(pwd)/weights:/app/weights \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/scripts:/app/scripts:ro \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        python main_gat.py \
            --n_class 3 \
            --n_features 768 \
            --hidden_dim 64 \
            --heads 4 \
            --data_path /data/graphs \
            --train_set /app/scripts/train_set.txt \
            --val_set /app/scripts/val_set.txt \
            --model_path /app/weights \
            --log_path /app/logs \
            --task_name gat_phikon \
            --batch_size 8 \
            --num_epochs 50 \
            --train
}

train_gtp() {
    echo -e "${GREEN}Training Graph Transformer...${NC}"
    docker run --gpus all \
        -v $(pwd)/data/graphs:/data/graphs:ro \
        -v $(pwd)/weights:/app/weights \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/scripts:/app/scripts:ro \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        python main.py \
            --n_class 3 \
            --n_features 768 \
            --data_path /data/graphs \
            --train_set /app/scripts/train_set.txt \
            --val_set /app/scripts/val_set.txt \
            --model_path /app/weights \
            --log_path /app/logs \
            --task_name gtp_phikon \
            --batch_size 8 \
            --train
}

evaluate() {
    echo -e "${GREEN}Evaluating GAT model...${NC}"
    docker run --gpus all \
        -v $(pwd)/data/graphs:/data/graphs:ro \
        -v $(pwd)/data/output:/data/output \
        -v $(pwd)/weights:/app/weights:ro \
        -v $(pwd)/scripts:/app/scripts:ro \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        python evaluate_gat.py
}

shell() {
    echo -e "${GREEN}Opening interactive shell...${NC}"
    docker run -it --gpus all \
        -v $(pwd)/data/graphs:/data/graphs \
        -v $(pwd)/data/output:/data/output \
        -v $(pwd)/weights:/app/weights \
        -v $(pwd)/scripts:/app/scripts \
        -v $(pwd)/logs:/app/logs \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        /bin/bash
}

# Main
case "${1}" in
    build)
        build
        ;;
    inference)
        run_inference
        ;;
    train-gat)
        train_gat
        ;;
    train-gtp)
        train_gtp
        ;;
    evaluate)
        evaluate
        ;;
    shell)
        shell
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo -e "${RED}Unknown command: ${1}${NC}"
        echo ""
        print_help
        exit 1
        ;;
esac