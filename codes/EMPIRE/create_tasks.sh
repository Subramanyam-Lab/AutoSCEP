#!/bin/bash
TOTAL_FILES=10000
TOTAL_PERIODS=8
TASK_FILE="tasks.txt"

rm -f $TASK_FILE

echo "Creating task list file: $TASK_FILE"
for (( file=1; file<=$TOTAL_FILES; file++ )); do
    for (( period=1; period<=$TOTAL_PERIODS; period++ )); do
        echo "$file $period" >> $TASK_FILE
    done
done
echo "Task list created with $(wc -l < $TASK_FILE) tasks."