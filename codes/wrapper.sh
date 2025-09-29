#!/bin/bash

for seed_value in {1..10}
do
  echo "=========================================="
  echo "   STARTING JOB FOR SEED = $seed_value"
  echo "=========================================="
  
  export SEED=$seed_value
  ./labeling_job.sh
  
  echo "Finished job for SEED = $seed_value."
  echo ""
done

echo "All seed jobs have been submitted and completed."