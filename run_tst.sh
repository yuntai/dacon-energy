#!/bin/bash
if [ -f /.dockerenv ]; then
    python tst.py
else
    set -x
    docker exec -it colab bash $0 $*
    set +x
fi

