#!/bin/bash
#
# Create jobs for a specific sweep.

NAME="topo_shapr_sweep"
SWEEP=$1

if [ -z "$1" ]; then
  exit -1
fi

if [ -z "$2" ]; then
  N_JOBS=48;
else
  N_JOBS=$2;
fi

for i in `seq ${N_JOBS}`; do
  sbatch -p gpu_p               \
         -J ${NAME}             \
         -o "${NAME}_%j.out"    \
         -x supergpu[05-08]     \
         --gres=gpu:1           \
         --qos=gpu              \
         --cpus-per-task=4      \
         --mem=4G               \
         --nice=10000           \
         --wrap "poetry run wandb agent --count 1 $1"
done
