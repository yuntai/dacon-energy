#!/bin/bash
if [ -f /.dockerenv ]; then
    set -euo pipefail
    SEED=42
    for i in $(seq $1 $2); do
        python fit_xgb.py --num $i --seed $SEED
    done
else
    set -x
    docker exec -it colab bash $0 $*
    set +x
fi

