#!/bin/sh

sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -cgi 14,14,14,14 -C
sudo nvidia-smi mig -lgi
nvidia-smi -L