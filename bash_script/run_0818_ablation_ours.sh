# #!/bin/bash
# # Training script for GAT / GCN / GATCN ablation

# # Default args
# DATA_PATH="/maiziezhou_lab2/yunfei/Projects/cellClustering_GNN/data"
# LABELS_PATH="../data/T10_groups.tsv"   # set to "" if no labels
# ARCH="16"
# N_CLUSTERS=4
# EPOCHS=1000
# LR=0.001
# DROPOUT=0.5
# COLLAPSE=1.0
# MODEL_TYPE="gatcn"   # options: gat, gcn, gatcn

# # Parse optional args
# while [[ $# -gt 0 ]]; do
#   case $1 in
#     --model_type) MODEL_TYPE="$2"; shift 2;;
#     --epochs)     EPOCHS="$2"; shift 2;;
#     --lr)         LR="$2"; shift 2;;
#     --dropout)    DROPOUT="$2"; shift 2;;
#     --clusters)   N_CLUSTERS="$2"; shift 2;;
#     --arch)       ARCH="$2"; shift 2;;
#     *) echo "Unknown argument: $1"; exit 1;;
#   esac
# done

# echo "Running training with:"
# echo "  Data path: $DATA_PATH"
# echo "  Labels: $LABELS_PATH"
# echo "  Model: $MODEL_TYPE"
# echo "  Epochs: $EPOCHS, LR: $LR, Dropout: $DROPOUT, Clusters: $N_CLUSTERS, Arch: $ARCH"

# python ../train_yunfei0818.py \
#   --data_path "$DATA_PATH" \
#   --labels_path "$LABELS_PATH" \
#   --architecture "$ARCH" \
#   --n_clusters "$N_CLUSTERS" \
#   --n_epochs "$EPOCHS" \
#   --learning_rate "$LR" \
#   --dropout_rate "$DROPOUT" \
#   --collapse_regularization "$COLLAPSE" \
#   --model_type "$MODEL_TYPE"


#!/bin/bash
# Training script for GAT / GCN / GATCN ablation across multiple simulation reps

# Default args
BASE_PATH="/maiziezhou_lab/yunfei/Projects/cellClustering_GNN/simulation_01172025/simulation"
LOG_FILE="../simulation_benchmarking_0816_ours.log"

ARCH="99,99,16" 
N_CLUSTERS=4
EPOCHS=1000
LR=0.001
DROPOUT=0.5
COLLAPSE=1.0
MODEL_TYPE="gatcn"   # options: gat, gcn, gatcn

# Parse optional args for training parameters
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_type) MODEL_TYPE="$2"; shift 2;;
    --epochs)     EPOCHS="$2"; shift 2;;
    --lr)         LR="$2"; shift 2;;
    --dropout)    DROPOUT="$2"; shift 2;;
    --clusters)   N_CLUSTERS="$2"; shift 2;;
    --architecture) ARCH="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

# Initialize log file
echo "Simulation Processing Log - $(date)" > "$LOG_FILE"

# Loop through all folders in the simulation directory
for folder in "$BASE_PATH"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        echo "Found subfolder: $folder_name" | tee -a "$LOG_FILE"
    fi

    # Loop through rep1 to rep5 in each folder
    for rep in {1..5}; do
        rep_path="${folder}/rep${rep}"

        if [ -d "$rep_path" ]; then
            echo "Processing: $rep_path" | tee -a "$LOG_FILE"

            # Retry mechanism (currently 1 attempt, can increase)
            for attempt in {1}; do
                echo "Attempt $attempt for $rep_path" | tee -a "$LOG_FILE"

                # Run training for this rep
                python ../train_yunfei0818.py \
                  --data_path "$rep_path/cell_adj_cosine.tsv" \
                  --labels_path "$rep_path/cells_groups.tsv" \
                  --architecture "$ARCH" \
                  --n_clusters "$N_CLUSTERS" \
                  --n_epochs "$EPOCHS" \
                  --learning_rate "$LR" \
                  --dropout_rate "$DROPOUT" \
                  --collapse_regularization "$COLLAPSE" \
                  --model_type "$MODEL_TYPE" \
                  >> "$LOG_FILE" 2>&1

                exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    echo "Successfully processed $rep_path on attempt $attempt" | tee -a "$LOG_FILE"
                else
                    echo "Failed attempt $attempt for $rep_path (exit code $exit_code)" | tee -a "$LOG_FILE"
                fi
            done
        else
            echo "Warning: $rep_path does not exist." | tee -a "$LOG_FILE"
        fi
    done
done
