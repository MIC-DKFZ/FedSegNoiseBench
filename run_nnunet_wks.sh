#!/bin/bash

# load default .bashrc.
source /home/m391k/.bashrc

shopt -s expand_aliases

# activate virtual env
source .venv/bin/activate

#set environment variables for nnunet
export nnUNet_raw="/home/m391k/E132-Rohdaten/nnUNetv2"
export nnUNet_preprocessed="/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/nnunet-preprocessed-blosc"
export nnUNet_results="/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/nnunet-results"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running nnUNetv2_train with args:"	
echo "${@:1}"
nnUNetv2_train "${@:1}"

if compgen -G "$PWD/core*"  > /dev/null; then
    rm core*
fi