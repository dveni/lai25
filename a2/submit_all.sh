#!/bin/bash

# Directory containing generated batch scripts
JOB_DIR="generated_jobs"

# Submit all *.sh files in that directory
for job in "$JOB_DIR"/*.sh; do
    echo "Submitting $job"
    sbatch "$job"
done
