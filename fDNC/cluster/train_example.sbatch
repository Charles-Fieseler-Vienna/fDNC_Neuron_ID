#!/usr/bin/env bash

#SBATCH --job-name=fdnc_training   # Job name
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=short             # Time limit hrs:min:sec
#SBATCH --output=fdnc_training.out    # Standard output and error log
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
# # SBATCH --gpus-per-task=1

pwd; hostname; date

module load cudnn/7.6.5

while getopts t:n: flag
do
    case "${flag}" in
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

CMD="/scratch/zimmer/Charles/github_repos/fDNC_Neuron_ID/fDNC/src/fDNC_train.py"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD"
else
  python $CMD --train_path ./Data/train_synthetic --eval_path ./Data/validation_synthetic
fi
