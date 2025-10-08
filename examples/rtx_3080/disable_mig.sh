#!/bin/sh

sudo nvidia-smi -i 0 -mig 0
sudo nvidia-smi --gpu-reset
nvidia-smi