# BFPy Library
Collection of software developed during my Master Thesis.
For further information, refer to my Master Thesis

## IMPORTANT: updated installation procedure below

The specified installation instructions in the printed vesion of the Master Thesis are no longer supported.
Note: Anaconda is required for this procedure. You can obtain your copy here: https://www.anaconda.com/products/distribution#Downloads

Then follow these steps:

1. navigate to lcoal directory on you machine, in which the repository is to be cloned into (where it will be stored)
2. If you have git installed, skip this step. Otherwise, in the Anaconda prompt (Windows) or terminal (Linux, MacOS) type the following, then hit return: conda install -c anaconda git
4. type the following, then hit return: git clone https://github.com/nfb2021/PlasmonicalPy.git
5. type the following, then hit return: cd PlasmonicalPy
6. type the following, then hit return: conda env create -f environment.yml
7. type the following, then hit return: conda activate PlasmonicalPy
8. type the following, then hit return: cd opencv_superres_models
9. type the following, then hit return: git clone https://github.com/Saafke/FSRCNN_Tensorflow.git
10. type the following, then hit return: git clone https://github.com/Saafke/EDSR_Tensorflow.git
11. type the following, then hit return: cd EDSR_Tensorflow/models
12. type the following, then hit return: cp * ../..
13. type the following, then hit return: cd ../../FSRCNN_Tensorflow/models/models
14. type the following, then hit return: cp * ../..

Now, everything is set up and ready to be used.
The minimum working example "BFPy_MWE.py" might take some time to run, depending on your machine's computational power.
To run in from within the terminal:
15. type the following, then hit return: cd ../../..
16. type the following, then hit return to run the minimum working example: python BFPy_MWE.py

Alternatively, an IDE like VisualStudioCode or PyCharm is recommended.
