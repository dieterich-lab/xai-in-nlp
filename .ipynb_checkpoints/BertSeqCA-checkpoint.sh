#!/bin/bash

# Settings to be modified:

# gres setup:
# - gpu:tesla is currently fixed as we only have GPUs as extract resource
# - Moreover we only have cards of the tesla type
# - The number after the clon specifies the number of GPUs required,
# e.g. something between 1 and 4

#SBATCH --gres=gpu:pascal:2

# Modifiy other SLURM variables as needed

#SBATCH --job-name=BertSeqCA
#SBATCH --output=BertSeqCA.txt
#SBATCH --partition=gpu
#SBATCH --mail-user=sari@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G

# Template for SLURM GPU handling scripts
# From https://techfak.net/gpu-cluster

# please do not modify the following parts

################################################################################################
echo "==== Start of GPU information ===="

CUDA_DEVICE=$(echo "$CUDA_VISIBLE_DEVICES," | cut -d',' -f $((SLURM_LOCALID + 1)) );
T_REGEX='^[0-9]$';
if ! [[ "$CUDA_DEVICE" =~ $T_REGEX ]]; then
        echo "error no reserved gpu provided"
        exit 1;
fi

# Print debug information

echo -e "SLURM job:\t$SLURM_JOBID"
#echo -e "SLURM process:\t$SLURM_PROCID"
#echo -e "SLURM GPU ID:\t$SLURM_LOCALID"
#echo -e "CUDA_DEVICE ID:\t$CUDA_DEVICE"
echo -e "CUDA_VISIBLE_DEVICES:\t$CUDA_VISIBLE_DEVICES"
echo "Device list:"
echo "$(nvidia-smi --query-gpu=name,gpu_uuid --format=csv -i $CUDA_VISIBLE_DEVICES | tail -n +2)"
echo "==== End of GPU information ===="
echo ""
#################################################################################################i

source MIEdeep/bin/activate

python3 BertSeqCardio.py
