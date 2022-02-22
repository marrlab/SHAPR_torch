#!/bin/bash

if [ -z "$1" ]; then
  PARAMS="";
else
  PARAMS=" --params $1";
fi

# Shift remainder of all parameters; we can drop $1 since we already
# handled it above.
shift

PARAMS="${PARAMS} $@"

sbatch -p gpu_p               \
       -J "topo_shapr"        \
       -o "topo_shapr%j.out"  \
       -x supergpu[05-08]     \
       --gres=gpu:1           \
       --qos=gpu              \
       --cpus-per-task=4      \
       --mem=4G               \
       --nice=10000           \
       --wrap "poetry run python run.py $PARAMS"
