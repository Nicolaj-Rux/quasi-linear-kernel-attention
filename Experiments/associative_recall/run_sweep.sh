#!/bin/bash
# Every kernel at its default tau x N = 20, 40, ..., 600 x seeds 0, 1, 2; one row per run in results.csv.
#   associative_recall/run_sweep.sh
set -eo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

for seed in 0 1 2; do
  for k in softmax gauss laplace riesz add_laplace add_riesz add_bump tri elu relu dpfp; do
    for N in $(seq 20 20 600); do
      python -m associative_recall.train --kernel $k --N $N --seed $seed
    done
  done
done
