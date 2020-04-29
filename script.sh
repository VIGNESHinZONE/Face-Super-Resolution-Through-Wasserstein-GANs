#!/bin/bash

#SBATCH --job-name="output"
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 06:00:00
#SBATCH --output=WGAN_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
nvidia-docker run -v /home/$USER:/home/$USER mustang/wgan:1.0 python -u ../home/mvenkataraman_ph/Face-Super-Resolution-Through-Wasserstein-GANs/main.py