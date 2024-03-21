# IoT Intrusion Detection System Performance in Imbalanced Botnet Datasets

This repository contains the research work conducted by Luis David Garcia and Nicholas Zarate. The focus of our study is on enhancing Intrusion Detection Systems (IDS) for the Internet of Things (IoT) by analyzing the impact of dataset imbalance on the efficacy of machine learning models in detecting security threats such as DoS and DDoS attacks.
Abstract

With the increasing adoption of IoT in sectors like energy, healthcare, and automotive, there's a rising productivity coupled with notable security vulnerabilities. Our study explores how the imbalance in datasets affects machine learning model performance in identifying these threats. We apply Synthetic Minority Over-sampling Technique (SMOTE) to balance the data and evaluate the performance enhancements in detection models, with the XGBoost model achieving an F1-score of 0.999983 under adjusted distributions.

## Research Paper

The detailed report, "Quantifying Feature Impact on IoT Intrusion Detection Performance in Imbalanced Botnet Datasets with Explainable AI", is included in this repository as `Quantifying_Feature_Impact_on_IoT_Intrusion_Detection_Performance_in_Imbalanced_Botnet_Datasets_with_Explainable_AI.pdf`.

## System Diagram

The system diagram provided outlines the methodology of our research, detailing the process from dataset preprocessing to the running of various classifiers and feature selection techniques.

<img width="487" alt="Screenshot 2024-03-21 at 10 37 14 AM" src="https://github.com/luisdavidgarcia/Botnet_ML_Feature_Analysis/assets/87344382/32e9272e-c1c7-40b7-a8fc-d7cf958ae92a">

Figure 1: System Architecture Diagram

## Models and Performance

We used the following classifiers in our study:

- XGBoost (XGB)
- Random Forest (RF)
- Decision Tree (DT)
- Logistic Regression (LR)

## Dataset

The dataset used is the [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) dataset, which includes multiple types of malicious attacks. The distribution was adjusted from a 60/40 benign-to-DDoS/DoS split to both 50/50 and 80/20 splits to evaluate the model performance.
Contributions

Our research contributes to the field by providing:

- An in-depth analysis of DoS and DDoS attack behaviors in IoT networks.
- An assessment of machine learning DoS detection across balanced and imbalanced datasets.
- An evaluation of dataset balancing techniques like SMOTE and their impact on predictive accuracy.

## Usage

To replicate our study or use the system as a foundation for further research, please refer to the following instructions:

- Ensure you have the required computational environment to handle machine learning workflows.
- Follow the dataset preprocessing steps outlined in our study to ensure data compatibility.
- Train the machine learning models using the provided scripts in the src directory.
- Evaluate the models using the performance metrics as described in the report.

## Authors

- Luis David Garcia - lgarc120@calpoly.edu
- Nicholas Zarate - nezarate@calpoly.edu

## Acknowledgments

We extend our gratitude to Dr. Dongfeng Fang for her guidance and support, and California Polytechnic State University for the resources provided for our research.
License

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE.md) file for details.
