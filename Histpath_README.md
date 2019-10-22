
# Histopathologic Cancer Detection 


#### -- Project Status: [Completed]

## Project Intro/Objective
The purpose of this project is to create algorithm to identify metastatic cancer from small image patches taken from the larger digital pathology scans. 


### Methods Used
* Data Augmentation
* Data Visualization
* Convolutional Neural Networks
* Transfer learning
* Adam Optimizer
* Binary Classification
* etc.

### Technologies
* Python, matplotlib
* Pandas, jupyter
* Sklearn
* Keras
* Tensorflow

## Project Description
A complete and accurate pathology report is crucial to getting a precise diagnosis and deciding on the best treatment plan for cancer patients. 

Doctors will often recommend a biopsy after a physical examination or a diagnostic test has identified a possible cancer.
During a biopsy, a doctor removes a small amount of tissue from the area of the body in question so it can be examined by a pathologist.
Histology refers to the study of the anatomy of cells and tissues at the microscopic level. The pathologist analyzes the appearance of cells under a microscope
and determines whether the tissue that was removed is benign (noncancerous) or malignant (cancerous). The pathology report includes information on the lymph node
status which documents whether the cancer has spread to nearby lymph nodes. 

The pathologist sends a pathology report to the doctor within **10 days** after the biopsy or surgery is performed. This is a long wait 
for those waiting for results. Is there a way to make fast and accurate predictions using deep learning tools?

The dataset used here is a slightly modified version of the PatchCamelyon (PCam) [benchmark dataset](https://github.com/basveeling/pcam)
It consists of 327.680 color images (96 x 96px) extracted from histopathologic scans of lymph node sections. Each image is annoted with a binary label indicating presence of metastatic tissue.. 
There exists 220024 images whose ground truth is known. A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. T
The original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/data) does not contain duplicates. 
I start by reviewing the provided input datasets, specifically ``` train.zip``` and ```train_labels.csv```. I split the "train" dataset
into a training and validation set. 

Using ```matplotlib.pyplot``` and ```keras``` I load a random sample of 6 images from the thus created training set. 

<img src="training_set_pos_neg_6.png.png" style="width:200px;height:450px;">
<caption><center> <u> <font color='purple'> Figure 1 </u><font color='purple'>  : Histopathologic scans of lymph node sections </center></caption>

Next I use Keras to define the Convolutional Neural Network to be used as the convolutional base 
