#!/bin/bash

if [[ "$#" -eq  "0" ]]; then
  echo "Submitting job with default values"
  SCRIPT="poetry run python run_train_script.py"
 else
  echo "Submitting with parameters from $1"
  SCRIPT="poetry run python run_train_script.py --params $1"
fi

srun -p interactive_gpu_p  \ 
     -J "topo_shapr"       \
     -o "topo_shapr%j.out" \
     --gres=gpu:1          \ 
     --qos=interactive_gpu \
     --cpus-per-task=4     \ 
     --mem=4G              \
     --nice=10000          \
     --wrap ${SCRIPT}
