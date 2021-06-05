#!/bin/bash
if [ -f /.dockerenv ]; then
    docker exec -it colab $0 $*
else
    set -euo pipefail
    SEED=42
    for i in $(seq $1 $2); do
        python fit_xgb.py --num $i --seed $SEED
    done
fi

