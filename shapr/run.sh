#!/bin/bash

if [ -z "$1" ]; then
  PARAMS="";
else
  PARAMS=" --params $1";
fi

sbatch -p gpu_p               \
       -J "topo_shapr"        \
       -o "topo_shapr%j.out"  \
       -x supergpu[05-08]     \
       --gres=gpu:1           \
       --qos=gpu              \
       --cpus-per-task=4      \
       --mem=4G               \
       --nice=10000           \
       --wrap "poetry run python run_train_script.py $PARAMS"
