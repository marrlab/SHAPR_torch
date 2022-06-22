#!/usr/bin/env zsh

for i in `seq 64`; do
  python analyse_interpolation.py -s $i -p config/small-0D.json
done

