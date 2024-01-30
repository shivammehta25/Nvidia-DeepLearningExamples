#!/usr/bin/env bash

PORT=${PORT:-8898}

docker run --gpus=all -it --rm -e CUDA_VISIBLE_DEVICES --ipc=host -p $PORT:$PORT -v $PWD:/workspace/fastpitch/ -v /home/smehta/Projects/Speech-Backbones/Grad-TTS/data:/workspace/fastpitch/data --name fp_det fastpitch:latest bash 
