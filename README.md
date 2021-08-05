# tensorflow-image-acp
Principal Components Analysis on images pixels via tensorflow

## conda environment

without CUDA:

    conda create -n pca python=3.9 tensorflow

with CUDA

    conda create -n pca-gpu python=3.9 tensorflow-gpu cudatoolkit cudnn

    pip install tensorflow-gpu

MAKE SURE THE ENVIRONMENT IS ACTIVATED BEFORE RUNNING THE CODE !

## usage

    python pca.py -I path/to/input/image -o path/to/output/file
