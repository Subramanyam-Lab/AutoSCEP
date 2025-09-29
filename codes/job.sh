#!/bin/bash
TOTAL_FILES=5000 
TOTAL_PERIODS=8 # fixed
TASK_FILE="tasks.txt"
LOG_FILE="dataset_generation_log_adaptive.csv"
NUMSAM=$TOTAL_FILES
SEED=${SEED:-1}
export NUMSAM SEED

echo "====================================="
echo "STEP 1: Starting Sampling Process"
echo "====================================="
sampling_start_time=$(date +%s)

sbatch --export=NUMSAM,SEED --wait sampling_script.sh

sampling_end_time=$(date +%s)
sampling_duration=$((sampling_end_time - sampling_start_time))
echo "Sampling process finished in $sampling_duration seconds."


echo "====================================="
echo "STEP 2: Starting Labeling Process"
echo "====================================="

echo "Creating the task list file: $TASK_FILE"
rm -f $TASK_FILE
for (( file=1; file<=$TOTAL_FILES; file++ )); do
    for (( period=1; period<=$TOTAL_PERIODS; period++ )); do
        echo "$file $period" >> $TASK_FILE
    done
done
echo "Task list created with $(wc -l < $TASK_FILE) tasks."


start_time=$(date +%s)
echo "Job array submitted at $(date)"


echo "Submitting job array and waiting for completion..."
sbatch --export=NUMSAM,SEED --wait worker_script.sh

end_time=$(date +%s)
echo "Job array finished at $(date)"
duration=$((end_time - start_time))
echo "Total Wall Clock Time: $duration seconds."

echo "Logging total wall clock time to $LOG_FILE"
if [ ! -f "$LOG_FILE" ]; then
    echo "seed,num_samples,sampling(s),labeling(s)" > $LOG_FILE
fi

echo "====================================="
echo "STEP 3: Logging Results"
echo "====================================="


awk -v seed="$SEED" -v ns="$NUMSAM" -v dur="$duration" 'BEGIN {FS=OFS=","} {
    if ($1 == seed && $2 == ns) {
        $4 = dur;
        found=1;
    }
    print;
} END {
    if (!found) {
        print seed,ns,0,dur;
    }
}' $LOG_FILE > tmp_log.csv && mv tmp_log.csv $LOG_FILE


echo "Logging complete."

echo "====================================="
echo "STEP 4: Starting Data Preprocessing"
echo "====================================="


python data_preprocessing.py --numsam "$NUMSAM" --seed "$SEED"
echo "Data preprocessing finished."

echo "====================================="
echo "STEP 5: ML models training"
echo "====================================="

sbatch --export=NUMSAM,SEED --wait ml_train.sh


echo "====================================="
echo "STEP 6: ML Embedding and Solving"
echo "====================================="

sbatch --export=NUMSAM,SEED --wait embedding.sh


echo "Script finished."