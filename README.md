# BFPy
Collection of software developed during my Master Thesis.
For further information, refer to my Master Thesis

## IMPORTANT: updated installation procedure below

The specified installation instructions in the printed vesion of the Master Thesis are no longer supported.
Note: Anaconda is required for this procedure. You can obtain your copy here: https://www.anaconda.com/products/distribution#Downloads

Then, follow these steps:

1. navigate to lcoal directory on you machine, in whoch the repository is to be cloned into
2. If you have git installed, skip this step. Otherwise, in the Anaconda prompt: conda install git
4. git clone https://github.com/nfb2021/PlasmonicalPy.git
5. cd PlasmonicalPy
6. conda env create -f environment.yml
7. conda activate PlasmonicalPy
8. cd opencv_superres_models
9. git clone https://github.com/Saafke/FSRCNN_Tensorflow.git
10. git clone https://github.com/Saafke/EDSR_Tensorflow.git
11. Move the pre-trained models from their correponsing directories "models" into "opencv_superres_models"
