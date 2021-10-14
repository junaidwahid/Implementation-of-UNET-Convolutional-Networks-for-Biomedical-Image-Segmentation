
# Implementation of UNET: Convolutional Networks for Biomedical Image Segmentation

This repository contains the implementation of a paper by  Dr. Thomas Brox and his team, "U-Net: Convolutional Networks for Biomedical Image Segmentation".

In this paper, authours present a elegant architecture for segmentation task, the so-called “fully convolutional network”.  This architecture designed in a way such that it works with very few training images and yields more precise segmentations. The main idea in paper is to supplement a usual contracting network by successive layers, where pooling operators are replaced by upsampling operators. Hence, these layers increase the resolution of the output. In order to localize, high resolution features from the contracting path are combined with the upsampled output. A successive convolution layer can then learn to assemble a more precise output based on this information. 

One important modification in this architecture is that in the upsampling part have also a large number of feature channels, which allow the network to propagate context information to higher resolution layers. As a consequence, the expansive path is more or less symmetric to the contracting path, and yields a u-shaped architecture. 

## Repository:
The repositories contains following files
* helper_functions. py
	* This file contains small functions that were use in other files.
*  unet_blocks .py
	* This file contains all necessary blocks for UNET architectures.
* unet. py
	* The file utlizes all UNET blocks and made the complete architecture.
*  train. py
	* This file utlizes the UNET archtecture and run the complete training on the given example data.


In order to run the training and visualize the results, you need to run this command. 
```
python train.py
```
