<img src=https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjo1NywidyI6MTAwMCwiZnMiOjU3LCJmZ2MiOiIjMzFBMEVCIiwiYmdjIjoiI0ZGRkZGRiIsInQiOjF9/QXV0b1NvdXJjZUlELUNsYXNzaWZpY2F0aW9u/kg-second-chances-sketch.png>


<!--
<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/OpticalImagePatch.png " width=50% height=50%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGOnOptical.png " width=50% height=50%> 
-->


# Description
ASID-C is an algorithm for the classification of stars and galaxies in single band optical images. 
ASID-C uses cutouts of 32x32 pixels of sources localized by AutoSourceID-Light (ASID-L) and assignes a calibrated probability to belong to either class.
Multiple Deep Learning networks have been applied and compared, leading to a rather surpirsing result: a vanilla convolutional neural network (CNN) is the best suited for the task.

## Table of Contents 
- [Work in progress](#work_in_progress)


# Work in progress

The hyperparameter space is being explored with the help of the great Weights & Biases software and soon a paper will be released to show the results and compare it to the available methods.

<img src="https://github.com/FiorenSt/AutoSourceID-Classification/blob/main/Plots/Hyper.png" width=100% height=100%> 
