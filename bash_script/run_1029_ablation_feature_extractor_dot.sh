#!/bin/bash
# Training script for feature extractor ablation across multiple simulation reps

# Default args
BASE_PATH="/gpfs/home/tp22o/simulation_01172025/"
# BASE_PATH="/gpfs/home/tp22o/Projects/cellClustering_GNN/simulation_01172025"

FEATURE_TYPES=(
    "dot"
)


# Loop over feature types
for FEATURE_TYPE in "${FEATURE_TYPES[@]}"; do
    FEATURE_TYPE_SAFE=$(echo "$FEATURE_TYPE" | tr ',' '_')
    LOG_FILE="/gpfs/home/tp22o/Projects/cellClustering_GNN/results_10292025/simulation_benchmarking_1029_fe_${FEATURE_TYPE_SAFE}.log"

    # Initialize log file
    echo "Simulation Processing Log - $(date)" > "$LOG_FILE"
    echo "Feature Type: $FEATURE_TYPE" | tee -a "$LOG_FILE"

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
                    python3 /gpfs/home/tp22o/Projects/cellClustering_GNN/train_10292025.py \
                      --data_path "$rep_path" \
                      --feature_type "$FEATURE_TYPE" \
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
done
