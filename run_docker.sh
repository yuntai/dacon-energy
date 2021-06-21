#!/bin/bash

set -euxo pipefail

docker build . -t colab

dataroot="/mnt/data"
if [[ $(hostname) == "nipa2020-0909" ]]; then
  echo "running on $(hostname)"
  dataroot="/home/ua/data"
fi
echo "dataroot($dataroot)"

docker stop colab || true && docker rm colab || true

#docker run --gpus all --shm-size=1g --rm --name colab --ulimit memlock=-1 --ulimit stack=67108864 -d --ipc=host --ip 0.0.0.0 -p 9999:8888 -v /mnt/tmp:/mnt/tmp -v $(pwd):/workspace -v ${dataroot}:${dataroot} colab
docker run --gpus all --shm-size=1g --rm --name colab --ulimit memlock=-1 --ulimit stack=67108864 -d --ipc=host --ip 0.0.0.0 -p 9999:8888 -v /mnt/tmp:/mnt/tmp -v /mnt/datasets:/mnt/datasets -v $(pwd):/workspace -v ${dataroot}:${dataroot} colab
