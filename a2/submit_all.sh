#!/bin/bash

# Directory containing generated batch scripts
JOB_DIR="generated_jobs"

# Your username (can also be detected automatically)
USER_NAME=$(whoami)

# Function to check if you have jobs running or pending
has_active_jobs() {
    squeue -u "$USER_NAME" | grep -qv "JOBID"
}

# Sort job files to ensure consistent order
job_files=("$JOB_DIR"/*.sh)
total_jobs=${#job_files[@]}
echo "Total jobs to submit: $total_jobs"

job_index=1
for job in "${job_files[@]}"; do
    echo "Waiting to submit job $job_index/$total_jobs: $job"

    # Wait until no jobs are in the queue
    while has_active_jobs; do
        echo "Job(s) still running. Waiting..."
        sleep 30  # Wait 30 seconds before checking again
    done

    echo "Submitting $job"
    sbatch "$job"
    ((job_index++))
done

echo "All jobs submitted."
