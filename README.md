# Focus, Fusion, and Rectify: Context-Aware Learning for COVID 19 Lung Infection Segmentation

## Introduction

The coronavirus disease 2019 (COVID-19) pandemic is spreading worldwide. Considering the limited clinicians and resources, and the evidence that computed tomography (CT) analysis can achieve comparable sensitivity, specificity and accuracy with reverse-transcription polymerase chain reaction, the automatic segmentation of lung infection from CT scans supplies a rapid and effective strategy for COVID-19 diagnosis, treatment and follow-up. It is challenging because the infection appearance has high intra-class variation and inter-class indistinction in CT slices. Therefore, a new context-aware neural network is proposed for lung infection segmentation. Specifically, the Autofocus and Panorama modules are designed for extracting fine details and semantic knowledge and capturing the long-range dependencies of the context from both peer-level and cross-level. Also, a novel structure consistency rectification is proposed for calibration by depicting the structural relationship between foreground and background.

![image](img/overview.png)

## Update

2021/8: the code released.

## Usage

1. Install pytorch 

   - The code is tested on python 3.7 and pytorch 1.2.0.

2. Dataset
   - Download the [Covid-19](https://medicalsegmentation.com/covid19/) dataset.
   - Please put dataset in folder `../Data/covid_19_seg/`

3. Train and test

   - please run the following code for a quick start:

   - ```shell
     CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset covid_19_seg --model ffrnet --bcakbone resnet50 --checkname exp-0804_ffrnet --bath-size 8
     ```


